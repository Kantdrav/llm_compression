from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Tuple
import re

import torch
from torch import nn

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:  # pragma: no cover - optional dependency shape varies by transformers version
    Conv1D = None

from .config import CompressionPolicy


class BaseCompressor(ABC):
    @abstractmethod
    def compress(self, model: nn.Module) -> nn.Module:
        raise NotImplementedError


class TensorNetworkCompressor(BaseCompressor):
    """Compress Linear layers using the legacy rank-ratio tensor-network path."""

    def __init__(self, rank_ratio: float = 0.5, layer_policy: CompressionPolicy | None = None):
        self.rank_ratio = rank_ratio
        self.layer_policy = layer_policy or CompressionPolicy()

    def compress(self, model: nn.Module) -> nn.Module:
        self._compress_module(model)
        return model

    def _compress_module(self, module: nn.Module, module_path: str = "") -> None:
        for child_name, child in list(module.named_children()):
            child_path = f"{module_path}.{child_name}" if module_path else child_name
            if isinstance(child, nn.Linear) and self._should_compress(child_path):
                setattr(module, child_name, TensorNetworkLinear.from_linear(child, self._rank_ratio_for(child_path)))
            else:
                self._compress_module(child, child_path)

    def _rank_ratio_for(self, module_path: str) -> float:
        return self.rank_ratio

    def _should_compress(self, module_path: str) -> bool:
        return not any(re.search(pattern, module_path) for pattern in self.layer_policy.skip_module_patterns)


class DynamicQuantizationCompressor(BaseCompressor):
    def compress(self, model: nn.Module) -> nn.Module:
        return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


class MPOCompressor(BaseCompressor):
    """Compress Linear and GPT-2 Conv1D layers with a truncated MPO."""

    def __init__(self, rank_ratio: float = 0.5, layer_policy: CompressionPolicy | None = None):
        self.rank_ratio = rank_ratio
        self.layer_policy = layer_policy or CompressionPolicy()

    def _rank_ratio_for(self, module_path: str) -> float:
        return self.rank_ratio

    def compress(self, model: nn.Module) -> nn.Module:
        self._compress_module(model)
        return model

    def _compress_module(self, module: nn.Module, module_path: str = "") -> None:
        for child_name, child in list(module.named_children()):
            child_path = f"{module_path}.{child_name}" if module_path else child_name
            if isinstance(child, nn.Linear):
                if self._should_compress(child_path):
                    setattr(module, child_name, MPOLinear.from_linear(child, self._rank_ratio_for(child_path), mpo_sites=3))
            elif Conv1D is not None and isinstance(child, Conv1D):
                if self._should_compress(child_path):
                    setattr(module, child_name, MPOLinear.from_conv1d(child, self._rank_ratio_for(child_path), mpo_sites=3))
            else:
                self._compress_module(child, child_path)

    def _should_compress(self, module_path: str) -> bool:
        return not any(re.search(pattern, module_path) for pattern in self.layer_policy.skip_module_patterns)


class MPOLinear(nn.Module):
    def __init__(
        self,
        cores: list[torch.Tensor],
        in_factors: list[int],
        out_factors: list[int],
        bias: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.cores = nn.ParameterList([nn.Parameter(c) for c in cores])
        self.in_factors = tuple(in_factors)
        self.out_factors = tuple(out_factors)
        if bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(bias)

    @classmethod
    def from_linear(cls, layer: nn.Linear, rank_ratio: float = 0.5, mpo_sites: int = 3) -> "MPOLinear":
        return cls._from_weight(layer.weight.detach().cpu(), layer.bias, rank_ratio, mpo_sites)

    @classmethod
    def from_conv1d(cls, layer: nn.Module, rank_ratio: float = 0.5, mpo_sites: int = 3) -> "MPOLinear":
        # GPT-2 Conv1D stores weights as [in_features, out_features],
        # whereas MPOLinear uses the standard [out_features, in_features] layout.
        weight = layer.weight.detach().cpu().t()
        return cls._from_weight(weight, getattr(layer, "bias", None), rank_ratio, mpo_sites)

    @classmethod
    def _from_weight(
        cls,
        weight: torch.Tensor,
        bias_parameter: torch.Tensor | nn.Parameter | None,
        rank_ratio: float,
        mpo_sites: int,
    ) -> "MPOLinear":
        if mpo_sites < 2:
            raise ValueError("mpo_sites must be at least 2")
        if not 0 < rank_ratio <= 1:
            raise ValueError("rank_ratio must be in (0, 1]")

        out_features, in_features = weight.shape
        out_factors = _partition_factors(out_features, mpo_sites)
        in_factors = _partition_factors(in_features, mpo_sites)

        tensor = weight.reshape(*out_factors, *in_factors)
        perm = []
        for i in range(mpo_sites):
            perm.extend((i, mpo_sites + i))
        tensor = tensor.permute(*perm)

        cores: list[torch.Tensor] = []
        remainder = tensor
        left_rank = 1

        for site in range(mpo_sites - 1):
            out_dim = out_factors[site]
            in_dim = in_factors[site]
            rest_shape = remainder.shape
            rows = left_rank * out_dim * in_dim
            matricized = remainder.reshape(rows, -1)
            u, s, vh = torch.linalg.svd(matricized, full_matrices=False)

            candidate_rank = max(1, int(len(s) * rank_ratio))
            rank = max(1, min(len(s), candidate_rank))
            u_r = u[:, :rank]
            s_r = s[:rank]
            vh_r = vh[:rank, :]

            cores.append((u_r * torch.sqrt(s_r)).reshape(left_rank, out_dim, in_dim, rank))

            remainder_start = 2 if site == 0 else 3
            remainder = (torch.sqrt(s_r).unsqueeze(1) * vh_r).reshape(rank, *rest_shape[remainder_start:])
            left_rank = rank

        cores.append(remainder.reshape(left_rank, out_factors[-1], in_factors[-1]))
        bias = bias_parameter.detach().cpu() if bias_parameter is not None else None
        return cls([c.contiguous() for c in cores], in_factors, out_factors, bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Contract the MPO directly without reconstructing the dense matrix.

        Core convention is [left_bond, output_site, input_site, right_bond].
        The previous implementation used axis arithmetic that could silently
        permute output/input sites during the repeated contractions. That is
        catastrophic for autoregressive LMs: the layer remains shape-correct but
        no longer represents the TT/MPO matrix produced by TT-SVD. The explicit
        einsum contraction below preserves the site order exactly.
        """
        expected_in = _product(self.in_factors)
        if inputs.shape[-1] != expected_in:
            raise ValueError(f"Expected input feature size {expected_in}, got {inputs.shape[-1]}")

        x = inputs.reshape(*inputs.shape[:-1], *self.in_factors)
        n = len(self.in_factors)

        # Integer einsum labels avoid string-label limits and make the mapping
        # from physical input/output sites to TT bonds explicit.
        input_labels = list(range(n))
        output_labels = list(range(n, 2 * n))
        bond_labels = list(range(2 * n, 2 * n + n - 1))

        current = x
        current_labels: list[object] = [Ellipsis, *input_labels]

        for site, core in enumerate(self.cores):
            in_label = input_labels[site]
            out_label = output_labels[site]
            if site == 0:
                # First core has left bond dimension 1; remove that singleton
                # dimension so it does not need to participate in the contraction.
                core_operand = core.squeeze(0)
                core_labels: list[object] = [out_label, in_label, bond_labels[0]] if n > 1 else [out_label, in_label]
                new_labels: list[object] = [Ellipsis, *output_labels[:1], *input_labels[1:], bond_labels[0]] if n > 1 else [Ellipsis, out_label]
            elif site < n - 1:
                bond_in = bond_labels[site - 1]
                bond_out = bond_labels[site]
                core_operand = core
                core_labels = [bond_in, out_label, in_label, bond_out]
                new_labels = [Ellipsis, *output_labels[: site + 1], *input_labels[site + 1 :], bond_out]
            else:
                bond_in = bond_labels[-1]
                core_operand = core.squeeze(-1)
                core_labels = [bond_in, out_label, in_label]
                new_labels = [Ellipsis, *output_labels]

            current = torch.einsum(current, current_labels, core_operand, core_labels, new_labels)
            current_labels = new_labels

        outputs = current.reshape(*inputs.shape[:-1], _product(self.out_factors))
        if self.bias is not None:
            outputs = outputs + self.bias
        return outputs


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _partition_factors(size: int, parts: int) -> list[int]:
    def prime_factors(n: int) -> list[int]:
        pf: list[int] = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                pf.append(d)
                n //= d
            d += 1 if d == 2 else 2
        if n > 1:
            pf.append(n)
        return pf

    factors = [1] * parts
    for p in sorted(prime_factors(size)):
        idx = min(range(parts), key=lambda i: factors[i])
        factors[idx] *= p
    prod = _product(factors)
    if prod != size:
        factors[-1] *= size // prod
    return factors


class TensorNetworkLinear(nn.Module):
    def __init__(self, left_core: torch.Tensor, right_core: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()
        self.left_core = nn.Parameter(left_core)
        self.right_core = nn.Parameter(right_core)
        self.bias = nn.Parameter(bias) if bias is not None else None

    @classmethod
    def from_linear(cls, layer: nn.Linear, rank_ratio: float) -> "TensorNetworkLinear":
        weight = layer.weight.detach()
        out_features, in_features = weight.shape
        rank = max(1, int(min(out_features, in_features) * rank_ratio))
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        u = u[:, :rank]
        s = s[:rank]
        vh = vh[:rank, :]
        left = u * torch.sqrt(s)
        right = torch.sqrt(s).unsqueeze(1) * vh
        return cls(left.cpu(), right.cpu(), layer.bias.detach().cpu() if layer.bias is not None else None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs.matmul(self.right_core.t()).matmul(self.left_core.t())
        if self.bias is not None:
            outputs = outputs + self.bias
        return outputs


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
