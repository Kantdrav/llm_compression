from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
from torch import nn

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:  # pragma: no cover
    Conv1D = None

from .compressors import MPOLinear, _partition_factors
from .config import CompressionPolicy


class ResearchMPOLinear(MPOLinear):
    """MPO Linear layer whose TT-SVD uses an explicit bond dimension chi.

    This deliberately does not convert chi into a global rank ratio. At every
    internal TT/MPO bond, TT-SVD truncates the singular spectrum to at most chi.
    """

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        *,
        bond_dim: int,
        mpo_sites: int = 3,
    ) -> "ResearchMPOLinear":
        return cls._from_weight_exact(
            layer.weight.detach().cpu(),
            layer.bias,
            bond_dim=bond_dim,
            mpo_sites=mpo_sites,
        )

    @classmethod
    def from_conv1d(
        cls,
        layer: nn.Module,
        *,
        bond_dim: int,
        mpo_sites: int = 3,
    ) -> "ResearchMPOLinear":
        weight = layer.weight.detach().cpu().t()
        return cls._from_weight_exact(
            weight,
            getattr(layer, "bias", None),
            bond_dim=bond_dim,
            mpo_sites=mpo_sites,
        )

    @classmethod
    def _from_weight_exact(
        cls,
        weight: torch.Tensor,
        bias_parameter: torch.Tensor | nn.Parameter | None,
        *,
        bond_dim: int,
        mpo_sites: int,
    ) -> "ResearchMPOLinear":
        if mpo_sites < 2:
            raise ValueError("mpo_sites must be at least 2")
        if bond_dim < 1:
            raise ValueError("bond_dim must be >= 1")

        out_features, in_features = weight.shape
        out_factors = _partition_factors(out_features, mpo_sites)
        in_factors = _partition_factors(in_features, mpo_sites)

        # Convert W[o, i] into an MPO tensor with paired physical dimensions:
        # [o1, i1, o2, i2, ..., oN, iN].
        tensor = weight.reshape(*out_factors, *in_factors)
        perm: list[int] = []
        for i in range(mpo_sites):
            perm.extend((i, mpo_sites + i))
        remainder = tensor.permute(*perm).contiguous()

        cores: list[torch.Tensor] = []
        left_rank = 1

        # TT-SVD. chi is the actual maximum internal bond dimension.
        for site in range(mpo_sites - 1):
            out_dim = out_factors[site]
            in_dim = in_factors[site]
            rest_shape = remainder.shape
            rows = left_rank * out_dim * in_dim
            matricized = remainder.reshape(rows, -1)

            u, s, vh = torch.linalg.svd(matricized, full_matrices=False)
            rank = max(1, min(int(bond_dim), len(s)))

            u_r = u[:, :rank]
            s_r = s[:rank]
            vh_r = vh[:rank, :]

            cores.append(
                (u_r * torch.sqrt(s_r))
                .reshape(left_rank, out_dim, in_dim, rank)
                .contiguous()
            )

            remainder_start = 2 if site == 0 else 3
            remainder = (
                torch.sqrt(s_r).unsqueeze(1) * vh_r
            ).reshape(rank, *rest_shape[remainder_start:]).contiguous()
            left_rank = rank

        cores.append(
            remainder.reshape(left_rank, out_factors[-1], in_factors[-1]).contiguous()
        )
        bias = bias_parameter.detach().cpu() if bias_parameter is not None else None
        return cls(cores, in_factors, out_factors, bias)


@dataclass(slots=True)
class ResearchMPOCompressor:
    """Research-inspired MPO compressor with an explicit bond-dimension budget.

    The research path uses TT-SVD directly with chi as the maximum TT/MPO bond
    dimension. Early/sensitive modules are protected by the default layer policy.
    """

    bond_dim: int = 16
    mpo_sites: int = 3
    layer_policy: CompressionPolicy = field(default_factory=CompressionPolicy.paper_default)

    def __post_init__(self) -> None:
        if self.bond_dim < 1:
            raise ValueError("bond_dim must be >= 1")
        if self.mpo_sites < 2:
            raise ValueError("mpo_sites must be >= 2")

    def compress(self, model: nn.Module) -> nn.Module:
        self._compress_module(model)
        return model

    def _compress_module(self, module: nn.Module, module_path: str = "") -> None:
        for child_name, child in list(module.named_children()):
            child_path = f"{module_path}.{child_name}" if module_path else child_name
            if isinstance(child, nn.Linear):
                if self._should_compress(child_path):
                    setattr(
                        module,
                        child_name,
                        ResearchMPOLinear.from_linear(
                            child,
                            bond_dim=self.bond_dim,
                            mpo_sites=self.mpo_sites,
                        ),
                    )
            elif Conv1D is not None and isinstance(child, Conv1D):
                if self._should_compress(child_path):
                    setattr(
                        module,
                        child_name,
                        ResearchMPOLinear.from_conv1d(
                            child,
                            bond_dim=self.bond_dim,
                            mpo_sites=self.mpo_sites,
                        ),
                    )
            else:
                self._compress_module(child, child_path)

    def _should_compress(self, module_path: str) -> bool:
        return not any(
            re.search(pattern, module_path)
            for pattern in self.layer_policy.skip_module_patterns
        )
