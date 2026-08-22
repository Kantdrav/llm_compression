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


@dataclass(slots=True)
class DynamicQuantizationCompressor(BaseCompressor):
    backend: str = "fbgemm"

    def compress(self, model: nn.Module) -> nn.Module:
        model = model.to("cpu").eval()
        torch.backends.quantized.engine = self.backend
        return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


@dataclass(slots=True)
class TensorNetworkCompressor(BaseCompressor):
    rank_ratio: float = 0.5
    layer_policy: CompressionPolicy = field(default_factory=CompressionPolicy)

    def compress(self, model: nn.Module) -> nn.Module:
        model = model.to("cpu").eval()
        self._compress_module(model)
        return model

    def _compress_module(self, module: nn.Module, module_path: str = "") -> None:
        for child_name, child in list(module.named_children()):
            child_path = f"{module_path}.{child_name}" if module_path else child_name
            if isinstance(child, nn.Linear):
                if self._should_compress(child_path):
                    setattr(module, child_name, TensorNetworkLinear.from_linear(child, self._rank_ratio_for(child_path)))
            elif Conv1D is not None and isinstance(child, Conv1D):
                if self._should_compress(child_path):
                    setattr(module, child_name, TensorNetworkConv1D.from_conv1d(child, self._rank_ratio_for(child_path)))
            else:
                self._compress_module(child, child_path)

    def _should_compress(self, module_path: str) -> bool:
        return not any(re.search(pattern, module_path) for pattern in self.layer_policy.skip_module_patterns)

    def _rank_ratio_for(self, module_path: str) -> float:
        return self.layer_policy.layer_rank_overrides.get(module_path, self.rank_ratio)


class LowRankCompressor(TensorNetworkCompressor):
    pass


@dataclass(slots=True)
class MPOCompressor(BaseCompressor):
    """Compress Linear layers with a truncated matrix-product operator (MPO)."""
    rank_ratio: float = 0.5
    layer_policy: CompressionPolicy = field(default_factory=CompressionPolicy)

    def compress(self, model: nn.Module) -> nn.Module:
        self._compress_module(model)
        return model

    def _compress_module(self, module: nn.Module, module_path: str = "") -> None:
        for child_name, child in list(module.named_children()):
            child_path = f"{module_path}.{child_name}" if module_path else child_name
            if isinstance(child, nn.Linear):
                if self._should_compress(child_path):
                    setattr(module, child_name, MPOLinear.from_linear(child, self._rank_ratio_for(child_path), mpo_sites=3))
            else:
                self._compress_module(child, child_path)

    def _should_compress(self, module_path: str) -> bool:
        return not any(re.search(pattern, module_path) for pattern in self.layer_policy.skip_module_patterns)

    def _rank_ratio_for(self, module_path: str) -> float:
        return self.layer_policy.layer_rank_overrides.get(module_path, self.rank_ratio)


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
        if mpo_sites < 2:
            raise ValueError("mpo_sites must be at least 2")
        if not 0 < rank_ratio <= 1:
            raise ValueError("rank_ratio must be in (0, 1]")

        weight = layer.weight.detach().cpu()
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

            # Site 0 has shape (o0, i0, o1, i1, ...); later remainders
            # already have a leading MPO bond dimension.
            remainder_start = 2 if site == 0 else 3
            remainder = (torch.sqrt(s_r).unsqueeze(1) * vh_r).reshape(rank, *rest_shape[remainder_start:])
            left_rank = rank

        cores.append(remainder.reshape(left_rank, out_factors[-1], in_factors[-1]))
        bias = layer.bias.detach().cpu() if layer.bias is not None else None
        return cls([c.contiguous() for c in cores], in_factors, out_factors, bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        expected_in = _product(self.in_factors)
        if inputs.shape[-1] != expected_in:
            raise ValueError(f"Expected input feature size {expected_in}, got {inputs.shape[-1]}")

        batch_shape = inputs.shape[:-1]
        batch_dims = len(batch_shape)
        x = inputs.reshape(*batch_shape, *self.in_factors)
        k = len(self.in_factors)

        # Intermediate layout: batch, input_0..input_{k-2}, bond, output_{k-1}.
        contracted = torch.einsum("...i,roi->...ro", x, self.cores[-1])

        for site in range(k - 2, -1, -1):
            input_axis = batch_dims + site
            bond_axis = input_axis + 1
            core = self.cores[site]
            contracted = torch.tensordot(contracted, core, dims=([input_axis, bond_axis], [2, 3]))

            # tensordot gives: batch, earlier inputs, existing outputs,
            # previous bond, current output. Move the new bond to the front
            # of the output block while keeping outputs in original order.
            prefix_len = batch_dims + site
            previous_bond_axis = contracted.ndim - 2
            current_output_axis = contracted.ndim - 1
            existing_output_start = prefix_len
            contracted = contracted.permute(
                *range(prefix_len),
                previous_bond_axis,
                current_output_axis,
                *range(existing_output_start, previous_bond_axis),
            )

        # Boundary bond has size one. Remaining dimensions are output factors
        # in the original order: o0, o1, ..., o{k-1}.
        contracted = contracted.squeeze(batch_dims)
        outputs = contracted.reshape(*batch_shape, _product(self.out_factors))
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
    def __init__(self, left_core: torch.Tensor, right_core: torch.Tensor, in_factors: Tuple[int, int], out_factors: Tuple[int, int], bias: torch.Tensor | None) -> None:
        super().__init__()
        self.left_core = nn.Parameter(left_core)
        self.right_core = nn.Parameter(right_core)
        self.in_factors = in_factors
        self.out_factors = out_factors
        self.bias = None if bias is None else nn.Parameter(bias)

    @classmethod
    def from_linear(cls, layer: nn.Linear, rank_ratio: float) -> "TensorNetworkLinear":
        weight = layer.weight.detach().cpu()
        out_features, in_features = weight.shape
        in_factors = _best_factor_pair(in_features)
        out_factors = _best_factor_pair(out_features)
        out_left, out_right = out_factors
        in_left, in_right = in_factors
        weight_tensor = weight.reshape(out_left, out_right, in_left, in_right)
        matricized = weight_tensor.permute(0, 2, 1, 3).reshape(out_left * in_left, out_right * in_right)
        u, s, vh = torch.linalg.svd(matricized, full_matrices=False)
        candidate_rank = max(1, int(len(s) * rank_ratio))
        compression_budget = max(1, ((out_features * in_features) - 1) // max(out_left * in_left + out_right * in_right, 1))
        rank = max(1, min(len(s), candidate_rank, compression_budget))
        left_core = (u[:, :rank] * torch.sqrt(s[:rank])).reshape(out_left, in_left, rank)
        right_core = (torch.sqrt(s[:rank]).unsqueeze(1) * vh[:rank, :]).reshape(rank, out_right, in_right)
        bias = layer.bias.detach().cpu() if layer.bias is not None else None
        return cls(left_core, right_core, in_factors, out_factors, bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_shape = inputs.shape[:-1]
        in_left, in_right = self.in_factors
        out_left, out_right = self.out_factors
        reshaped_inputs = inputs.reshape(*batch_shape, in_left, in_right)
        right_projected = torch.einsum("...uv,rov->...uro", reshaped_inputs, self.right_core)
        outputs = torch.einsum("...uro,bur->...bo", right_projected, self.left_core)
        outputs = outputs.reshape(*batch_shape, out_left * out_right)
        if self.bias is not None:
            outputs = outputs + self.bias
        return outputs


class TensorNetworkConv1D(nn.Module):
    def __init__(self, left_core: torch.Tensor, right_core: torch.Tensor, in_factors: Tuple[int, int], out_factors: Tuple[int, int], bias: torch.Tensor | None) -> None:
        super().__init__()
        self.left_core = nn.Parameter(left_core)
        self.right_core = nn.Parameter(right_core)
        self.in_factors = in_factors
        self.out_factors = out_factors
        self.bias = None if bias is None else nn.Parameter(bias)

    @classmethod
    def from_conv1d(cls, layer: nn.Module, rank_ratio: float) -> "TensorNetworkConv1D":
        weight = layer.weight.detach().cpu().t()
        out_features, in_features = weight.shape
        in_factors = _best_factor_pair(in_features)
        out_factors = _best_factor_pair(out_features)
        out_left, out_right = out_factors
        in_left, in_right = in_factors
        weight_tensor = weight.reshape(out_left, out_right, in_left, in_right)
        matricized = weight_tensor.permute(0, 2, 1, 3).reshape(out_left * in_left, out_right * in_right)
        u, s, vh = torch.linalg.svd(matricized, full_matrices=False)
        candidate_rank = max(1, int(len(s) * rank_ratio))
        compression_budget = max(1, ((out_features * in_features) - 1) // max(out_left * in_left + out_right * in_right, 1))
        rank = max(1, min(len(s), candidate_rank, compression_budget))
        left_core = (u[:, :rank] * torch.sqrt(s[:rank])).reshape(out_left, in_left, rank)
        right_core = (torch.sqrt(s[:rank]).unsqueeze(1) * vh[:rank, :]).reshape(rank, out_right, in_right)
        bias = layer.bias.detach().cpu() if getattr(layer, "bias", None) is not None else None
        return cls(left_core, right_core, in_factors, out_factors, bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_shape = inputs.shape[:-1]
        in_left, in_right = self.in_factors
        out_left, out_right = self.out_factors
        reshaped_inputs = inputs.reshape(*batch_shape, in_left, in_right)
        right_projected = torch.einsum("...uv,rov->...uro", reshaped_inputs, self.right_core)
        outputs = torch.einsum("...uro,bur->...bo", right_projected, self.left_core)
        outputs = outputs.reshape(*batch_shape, out_left * out_right)
        if self.bias is not None:
            outputs = outputs + self.bias
        return outputs


def _best_factor_pair(size: int) -> Tuple[int, int]:
    root = int(size ** 0.5)
    for left in range(root, 0, -1):
        if size % left == 0:
            return left, size // left
    return 1, size


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def iter_linear_layers(module: nn.Module) -> Iterable[nn.Linear]:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            yield child
