from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:  # pragma: no cover
    Conv1D = None

from .compressors import MPOCompressor, MPOLinear, _partition_factors, _product
from .config import CompressionPolicy


@dataclass(slots=True)
class AdaptiveLayerCompressionReport:
    """Detailed compression and reconstruction report for an adaptive MPO layer."""
    module_path: str
    original_parameters: int
    compressed_parameters: int
    parameter_ratio: float
    reduction_percent: float
    selected_ranks: tuple[int, ...]
    retained_energies: tuple[float, ...]
    discarded_energies: tuple[float, ...]
    energy_threshold_met: bool
    relative_frobenius_error: float
    actual_error_squared: float
    theoretical_error_bound_squared: float
    target_reduction: float
    target_reduction_achieved: bool


class AdaptiveMPOLinear(MPOLinear):
    """MPO Linear layer whose TT-SVD bond dimensions are determined adaptively at each cut.

    Uses standard left-canonical TT-SVD (Oseledets, 2011). Non-terminal cores are strictly
    left-orthogonal (U_k^T U_k = I), while the remaining singular-value spectrum is absorbed
    into the remainder tensor. The terminal core carries the remaining singular values.
    This guarantees that the per-cut discarded energy corresponds strictly to the Frobenius
    distortion and satisfies the quasi-optimal global TT-SVD error bound:
        ||W - W_hat||_F^2 <= sum_{k=0}^{N-2} epsilon_k^2
    """

    selected_ranks: tuple[int, ...]
    retained_energies: tuple[float, ...]
    discarded_energies: tuple[float, ...]
    energy_threshold_met: bool
    relative_frobenius_error: float
    actual_reconstruction_error_squared: float
    theoretical_error_bound_squared: float

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        *,
        energy_threshold: float = 0.995,
        min_bond_dim: int = 1,
        max_bond_dim: int | None = None,
        mpo_sites: int = 3,
    ) -> "AdaptiveMPOLinear":
        return cls._from_weight_adaptive(
            layer.weight.detach().cpu(),
            layer.bias,
            energy_threshold=energy_threshold,
            min_bond_dim=min_bond_dim,
            max_bond_dim=max_bond_dim,
            mpo_sites=mpo_sites,
        )

    @classmethod
    def from_conv1d(
        cls,
        layer: nn.Module,
        *,
        energy_threshold: float = 0.995,
        min_bond_dim: int = 1,
        max_bond_dim: int | None = None,
        mpo_sites: int = 3,
    ) -> "AdaptiveMPOLinear":
        # GPT-2 Conv1D stores weights as [in_features, out_features],
        # whereas standard Linear uses [out_features, in_features].
        weight = layer.weight.detach().cpu().t()
        return cls._from_weight_adaptive(
            weight,
            getattr(layer, "bias", None),
            energy_threshold=energy_threshold,
            min_bond_dim=min_bond_dim,
            max_bond_dim=max_bond_dim,
            mpo_sites=mpo_sites,
        )

    @classmethod
    def _from_weight_adaptive(
        cls,
        weight: torch.Tensor,
        bias_parameter: torch.Tensor | nn.Parameter | None,
        *,
        energy_threshold: float = 0.995,
        min_bond_dim: int = 1,
        max_bond_dim: int | None = None,
        mpo_sites: int = 3,
    ) -> "AdaptiveMPOLinear":
        if mpo_sites < 2:
            raise ValueError("mpo_sites must be at least 2")
        if not 0.0 < energy_threshold <= 1.0:
            raise ValueError("energy_threshold must be in (0, 1]")
        if min_bond_dim < 1:
            raise ValueError("min_bond_dim must be >= 1")
        if max_bond_dim is not None and max_bond_dim < min_bond_dim:
            raise ValueError(f"max_bond_dim ({max_bond_dim}) cannot be less than min_bond_dim ({min_bond_dim})")

        # Use CPU float32 for deterministic, numerically stable SVD
        weight_fp32 = weight.detach().cpu().float()
        out_features, in_features = weight_fp32.shape
        out_factors = _partition_factors(out_features, mpo_sites)
        in_factors = _partition_factors(in_features, mpo_sites)

        # Convert W[o, i] into an MPO tensor with paired physical dimensions:
        # [o_0, i_0, o_1, i_1, ..., o_{N-1}, i_{N-1}]
        tensor = weight_fp32.reshape(*out_factors, *in_factors)
        perm: list[int] = []
        for i in range(mpo_sites):
            perm.extend((i, mpo_sites + i))
        remainder = tensor.permute(*perm).contiguous()

        cores: list[torch.Tensor] = []
        left_rank = 1
        selected_ranks: list[int] = []
        retained_energies: list[float] = []
        discarded_energies: list[float] = []
        all_thresholds_met = True

        for site in range(mpo_sites - 1):
            out_dim = out_factors[site]
            in_dim = in_factors[site]
            rest_shape = remainder.shape
            rows = left_rank * out_dim * in_dim
            matricized = remainder.reshape(rows, -1)

            u, s, vh = torch.linalg.svd(matricized, full_matrices=False)
            total_energy = s.square().sum()

            # Handle numerical edge cases: zero or near-zero tensors
            eps = torch.finfo(s.dtype).eps
            if total_energy <= eps:
                rank = 1
                energy_ratio = 1.0
                discarded_energy = 0.0
                cut_threshold_met = True
            else:
                cum_energy = torch.cumsum(s.square(), dim=0)
                norm_energy = cum_energy / total_energy
                mask = norm_energy >= energy_threshold
                energy_rank = int(torch.nonzero(mask, as_tuple=False)[0].item() + 1) if mask.any() else len(s)

                # Ensure increasing energy_threshold never decreases rank
                candidate_rank = max(min_bond_dim, energy_rank)

                # If max_bond_dim prevents meeting the energy threshold, flag it
                if max_bond_dim is not None and candidate_rank > max_bond_dim:
                    rank = max_bond_dim
                    cut_threshold_met = False
                    all_thresholds_met = False
                else:
                    rank = candidate_rank
                    cut_threshold_met = True

                rank = max(1, min(len(s), rank))
                energy_ratio = float((cum_energy[rank - 1] / total_energy).item())
                discarded_energy = float(s[rank:].square().sum().item())
                if norm_energy[rank - 1].item() < energy_threshold:
                    all_thresholds_met = False
                    cut_threshold_met = False

            selected_ranks.append(rank)
            retained_energies.append(energy_ratio)
            discarded_energies.append(discarded_energy)

            u_r = u[:, :rank]
            s_r = s[:rank]
            vh_r = vh[:rank, :]

            # Left-canonical core: U_r is strictly left-orthogonal (U_r^T U_r = I)
            cores.append(
                u_r.reshape(left_rank, out_dim, in_dim, rank)
                .contiguous()
            )

            # Left-canonical remainder: absorbs the full singular spectrum (s_r * vh_r),
            # strictly preserving the Frobenius norm of the remaining tensor bipartition.
            remainder_start = 2 if site == 0 else 3
            remainder = (
                s_r.unsqueeze(1) * vh_r
            ).reshape(rank, *rest_shape[remainder_start:]).contiguous()
            left_rank = rank

        # Terminal core carries the remaining singular values and scaling
        cores.append(remainder.reshape(left_rank, out_factors[-1], in_factors[-1]).contiguous())
        bias = bias_parameter.detach().cpu() if bias_parameter is not None else None

        instance = cls(cores, in_factors, out_factors, bias)
        instance.selected_ranks = tuple(selected_ranks)
        instance.retained_energies = tuple(retained_energies)
        instance.discarded_energies = tuple(discarded_energies)
        instance.energy_threshold_met = all_thresholds_met

        # Measure relative Frobenius reconstruction error and Oseledets bound
        w_recon = instance.reconstruct_weight()
        norm_orig = torch.norm(weight_fp32)
        diff_norm_sq = float((torch.norm(weight_fp32 - w_recon) ** 2).item())
        bound_sq = sum(discarded_energies)

        if norm_orig <= eps:
            rel_error = 0.0 if torch.norm(w_recon) <= eps else 1.0
        else:
            rel_error = float((torch.sqrt(torch.tensor(diff_norm_sq)) / norm_orig).item())

        instance.relative_frobenius_error = rel_error
        instance.actual_reconstruction_error_squared = diff_norm_sq
        instance.theoretical_error_bound_squared = bound_sq

        return instance

    @classmethod
    def _from_weight_symmetric(
        cls,
        weight: torch.Tensor,
        bias_parameter: torch.Tensor | nn.Parameter | None,
        *,
        energy_threshold: float = 0.995,
        min_bond_dim: int = 1,
        max_bond_dim: int | None = None,
        mpo_sites: int = 3,
    ) -> "AdaptiveMPOLinear":
        """Reference implementation of symmetric sqrt(Sigma) splitting for benchmark comparison."""
        weight_fp32 = weight.detach().cpu().float()
        out_features, in_features = weight_fp32.shape
        out_factors = _partition_factors(out_features, mpo_sites)
        in_factors = _partition_factors(in_features, mpo_sites)

        tensor = weight_fp32.reshape(*out_factors, *in_factors)
        perm = []
        for i in range(mpo_sites):
            perm.extend((i, mpo_sites + i))
        remainder = tensor.permute(*perm).contiguous()

        cores: list[torch.Tensor] = []
        left_rank = 1
        selected_ranks: list[int] = []
        retained_energies: list[float] = []
        discarded_energies: list[float] = []
        all_thresholds_met = True

        for site in range(mpo_sites - 1):
            out_dim = out_factors[site]
            in_dim = in_factors[site]
            rest_shape = remainder.shape
            rows = left_rank * out_dim * in_dim
            matricized = remainder.reshape(rows, -1)

            u, s, vh = torch.linalg.svd(matricized, full_matrices=False)
            total_energy = s.square().sum()
            eps = torch.finfo(s.dtype).eps
            if total_energy <= eps:
                rank = 1
                energy_ratio = 1.0
                discarded_energy = 0.0
            else:
                cum_energy = torch.cumsum(s.square(), dim=0)
                norm_energy = cum_energy / total_energy
                mask = norm_energy >= energy_threshold
                energy_rank = int(torch.nonzero(mask, as_tuple=False)[0].item() + 1) if mask.any() else len(s)
                candidate_rank = max(min_bond_dim, energy_rank)
                if max_bond_dim is not None and candidate_rank > max_bond_dim:
                    rank = max_bond_dim
                    all_thresholds_met = False
                else:
                    rank = candidate_rank
                rank = max(1, min(len(s), rank))
                energy_ratio = float((cum_energy[rank - 1] / total_energy).item())
                discarded_energy = float(s[rank:].square().sum().item())
                if norm_energy[rank - 1].item() < energy_threshold:
                    all_thresholds_met = False

            selected_ranks.append(rank)
            retained_energies.append(energy_ratio)
            discarded_energies.append(discarded_energy)

            u_r = u[:, :rank]
            s_r = s[:rank]
            vh_r = vh[:rank, :]

            # Symmetric sqrt(s) core and remainder
            cores.append((u_r * torch.sqrt(s_r)).reshape(left_rank, out_dim, in_dim, rank).contiguous())
            remainder_start = 2 if site == 0 else 3
            remainder = (torch.sqrt(s_r).unsqueeze(1) * vh_r).reshape(rank, *rest_shape[remainder_start:]).contiguous()
            left_rank = rank

        cores.append(remainder.reshape(left_rank, out_factors[-1], in_factors[-1]).contiguous())
        bias = bias_parameter.detach().cpu() if bias_parameter is not None else None

        instance = cls(cores, in_factors, out_factors, bias)
        instance.selected_ranks = tuple(selected_ranks)
        instance.retained_energies = tuple(retained_energies)
        instance.discarded_energies = tuple(discarded_energies)
        instance.energy_threshold_met = all_thresholds_met

        w_recon = instance.reconstruct_weight()
        norm_orig = torch.norm(weight_fp32)
        diff_norm_sq = float((torch.norm(weight_fp32 - w_recon) ** 2).item())
        instance.relative_frobenius_error = float((torch.sqrt(torch.tensor(diff_norm_sq)) / max(norm_orig, 1e-12)).item())
        instance.actual_reconstruction_error_squared = diff_norm_sq
        instance.theoretical_error_bound_squared = sum(discarded_energies)
        return instance

    def reconstruct_weight(self) -> torch.Tensor:
        """Contract MPO cores directly into dense [out_features, in_features] weight matrix."""
        n = len(self.cores)
        res = self.cores[0].squeeze(0)  # [o_0, i_0, r_1]
        for i in range(1, n):
            c = self.cores[i]
            res = torch.tensordot(res, c, dims=([-1], [0]))

        out_indices = [2 * k for k in range(n)]
        in_indices = [2 * k + 1 for k in range(n)]
        res = res.permute(*(out_indices + in_indices)).contiguous()
        return res.reshape(_product(self.out_factors), _product(self.in_factors))


class AdaptiveMPOCompressor(MPOCompressor):
    """MPO compressor with per-cut rank selection based on left-canonical TT-SVD.

    Uses sequential unfolding matricizations and minimal rank selection satisfying the
    prescribed Frobenius energy threshold with standard left-canonical cores.
    """

    def __init__(
        self,
        rank_ratio: float = 0.5,
        layer_policy: CompressionPolicy | None = None,
        energy_threshold: float = 0.995,
        target_reduction: float = 0.30,
        mpo_sites: int = 3,
        min_bond_dim: int = 1,
        max_bond_dim: int | None = None,
    ) -> None:
        super().__init__(rank_ratio=rank_ratio, layer_policy=layer_policy or CompressionPolicy(), mpo_sites=mpo_sites)
        if not 0.0 < energy_threshold <= 1.0:
            raise ValueError("energy_threshold must be in (0, 1]")
        if not 0.0 <= target_reduction < 1.0:
            raise ValueError("target_reduction must be in [0, 1)")
        if min_bond_dim < 1:
            raise ValueError("min_bond_dim must be >= 1")
        if max_bond_dim is not None and max_bond_dim < min_bond_dim:
            raise ValueError(f"max_bond_dim ({max_bond_dim}) cannot be less than min_bond_dim ({min_bond_dim})")

        self.energy_threshold = energy_threshold
        self.target_reduction = target_reduction
        self.min_bond_dim = min_bond_dim
        self.max_bond_dim = max_bond_dim
        self.compression_reports: dict[str, AdaptiveLayerCompressionReport] = {}

    def _rank_ratio_for(self, module_path: str, layer: nn.Module | None = None) -> float:
        """Backward-compatibility helper returning a scalar ratio proxy."""
        override = self.layer_policy.layer_rank_overrides.get(module_path)
        if override is not None:
            return override
        return self.rank_ratio

    def _compress_module(self, module: nn.Module, module_path: str = "") -> None:
        for child_name, child in list(module.named_children()):
            child_path = f"{module_path}.{child_name}" if module_path else child_name

            if isinstance(child, nn.Linear):
                if self._should_compress(child_path):
                    compressed = self._compress_linear(child, child_path)
                    setattr(module, child_name, compressed)
            elif Conv1D is not None and isinstance(child, Conv1D):
                if self._should_compress(child_path):
                    compressed = self._compress_conv1d(child, child_path)
                    setattr(module, child_name, compressed)
            else:
                self._compress_module(child, child_path)

    def _compress_linear(self, child: nn.Linear, child_path: str) -> AdaptiveMPOLinear:
        override = self.layer_policy.layer_rank_overrides.get(child_path)
        max_bond = int(override) if (override is not None and override >= 1.0) else self.max_bond_dim
        threshold = override if (override is not None and 0.0 < override < 1.0) else self.energy_threshold

        orig_params = child.weight.numel() + (child.bias.numel() if child.bias is not None else 0)

        compressed = AdaptiveMPOLinear.from_linear(
            child,
            energy_threshold=threshold,
            min_bond_dim=self.min_bond_dim,
            max_bond_dim=max_bond,
            mpo_sites=self.mpo_sites,
        )

        comp_params = sum(c.numel() for c in compressed.cores) + (compressed.bias.numel() if compressed.bias is not None else 0)
        self._record_report(child_path, orig_params, comp_params, compressed)
        return compressed

    def _compress_conv1d(self, child: nn.Module, child_path: str) -> AdaptiveMPOLinear:
        override = self.layer_policy.layer_rank_overrides.get(child_path)
        max_bond = int(override) if (override is not None and override >= 1.0) else self.max_bond_dim
        threshold = override if (override is not None and 0.0 < override < 1.0) else self.energy_threshold

        orig_params = child.weight.numel() + (child.bias.numel() if getattr(child, "bias", None) is not None else 0)

        compressed = AdaptiveMPOLinear.from_conv1d(
            child,
            energy_threshold=threshold,
            min_bond_dim=self.min_bond_dim,
            max_bond_dim=max_bond,
            mpo_sites=self.mpo_sites,
        )

        comp_params = sum(c.numel() for c in compressed.cores) + (compressed.bias.numel() if compressed.bias is not None else 0)
        self._record_report(child_path, orig_params, comp_params, compressed)
        return compressed

    def _record_report(
        self,
        child_path: str,
        orig_params: int,
        comp_params: int,
        compressed: AdaptiveMPOLinear,
    ) -> None:
        ratio = comp_params / max(orig_params, 1)
        reduction_percent = (1.0 - ratio) * 100.0
        target_achieved = (1.0 - ratio) >= (self.target_reduction - 1e-6)

        self.compression_reports[child_path] = AdaptiveLayerCompressionReport(
            module_path=child_path,
            original_parameters=orig_params,
            compressed_parameters=comp_params,
            parameter_ratio=round(ratio, 6),
            reduction_percent=round(reduction_percent, 2),
            selected_ranks=compressed.selected_ranks,
            retained_energies=tuple(round(e, 6) for e in compressed.retained_energies),
            discarded_energies=tuple(round(e, 6) for e in compressed.discarded_energies),
            energy_threshold_met=compressed.energy_threshold_met,
            relative_frobenius_error=round(compressed.relative_frobenius_error, 6),
            actual_error_squared=round(compressed.actual_reconstruction_error_squared, 6),
            theoretical_error_bound_squared=round(compressed.theoretical_error_bound_squared, 6),
            target_reduction=self.target_reduction,
            target_reduction_achieved=target_achieved,
        )


def compare_mpo_compressors(
    weight: torch.Tensor,
    *,
    mpo_sites: int = 3,
    energy_threshold: float = 0.995,
    rank_ratio: float = 0.5,
    min_bond_dim: int = 1,
    max_bond_dim: int | None = None,
) -> dict[str, Any]:
    """Compare fixed-rank, symmetric adaptive, and left-canonical adaptive MPO compression."""
    weight = weight.detach().cpu().float()
    out_features, in_features = weight.shape
    orig_params = weight.numel()

    # 1. Fixed-rank MPO
    fixed_layer = MPOLinear._from_weight(weight, None, rank_ratio=rank_ratio, mpo_sites=mpo_sites)
    fixed_params = sum(c.numel() for c in fixed_layer.cores)
    fixed_ranks = tuple(c.shape[-1] for c in fixed_layer.cores[:-1])
    fixed_recon = AdaptiveMPOLinear.reconstruct_weight(fixed_layer)
    norm_orig = torch.norm(weight)
    fixed_err = float((torch.norm(weight - fixed_recon) / norm_orig).item()) if norm_orig > 0 else 0.0

    # 2. Symmetric adaptive MPO (reference)
    symmetric_layer = AdaptiveMPOLinear._from_weight_symmetric(
        weight,
        None,
        energy_threshold=energy_threshold,
        min_bond_dim=min_bond_dim,
        max_bond_dim=max_bond_dim,
        mpo_sites=mpo_sites,
    )

    # 3. Left-canonical adaptive MPO (standard)
    canonical_layer = AdaptiveMPOLinear._from_weight_adaptive(
        weight,
        None,
        energy_threshold=energy_threshold,
        min_bond_dim=min_bond_dim,
        max_bond_dim=max_bond_dim,
        mpo_sites=mpo_sites,
    )
    canonical_params = sum(c.numel() for c in canonical_layer.cores)
    canonical_ranks = canonical_layer.selected_ranks
    canonical_err = canonical_layer.relative_frobenius_error

    canonical_dict = {
        "energy_threshold": energy_threshold,
        "selected_ranks": canonical_ranks,
        "retained_energies": canonical_layer.retained_energies,
        "discarded_energies": canonical_layer.discarded_energies,
        "energy_threshold_met": canonical_layer.energy_threshold_met,
        "compressed_parameters": canonical_params,
        "parameter_ratio": round(canonical_params / orig_params, 6),
        "size_reduction_percent": round((1.0 - canonical_params / orig_params) * 100.0, 2),
        "relative_frobenius_error": round(canonical_err, 6),
        "actual_error_squared": round(canonical_layer.actual_reconstruction_error_squared, 6),
        "theoretical_error_bound_squared": round(canonical_layer.theoretical_error_bound_squared, 6),
    }

    return {
        "original_shape": (out_features, in_features),
        "original_parameters": orig_params,
        "fixed_mpo": {
            "rank_ratio": rank_ratio,
            "selected_ranks": fixed_ranks,
            "compressed_parameters": fixed_params,
            "parameter_ratio": round(fixed_params / orig_params, 6),
            "size_reduction_percent": round((1.0 - fixed_params / orig_params) * 100.0, 2),
            "relative_frobenius_error": round(fixed_err, 6),
        },
        "symmetric_adaptive_mpo": {
            "energy_threshold": energy_threshold,
            "selected_ranks": symmetric_layer.selected_ranks,
            "compressed_parameters": sum(c.numel() for c in symmetric_layer.cores),
            "relative_frobenius_error": round(symmetric_layer.relative_frobenius_error, 6),
            "actual_error_squared": round(symmetric_layer.actual_reconstruction_error_squared, 6),
            "sum_discarded_energies": round(symmetric_layer.theoretical_error_bound_squared, 6),
        },
        "canonical_adaptive_mpo": canonical_dict,
        "adaptive_mpo": canonical_dict,
    }
