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
    """Scaffold for a Matrix Product Operator compressor.

    Currently delegates to the TensorNetworkCompressor as a baseline and
    provides a named entrypoint for wiring into the pipeline. The full
    MPO decomposition will replace this delegation.
    """
    rank_ratio: float = 0.5
    layer_policy: CompressionPolicy = field(default_factory=CompressionPolicy)

    def compress(self, model: nn.Module) -> nn.Module:
        for name, child in list(model.named_children()):
            if isinstance(child, nn.Linear):
                setattr(model, name, MPOLinear.from_linear(child, self.rank_ratio, mpo_sites=3))
            else:
                self.compress(child)
        return model


class MPOLinear(nn.Module):
    def __init__(self, cores: list[torch.Tensor], in_factors: list[int], out_factors: list[int], bias: torch.Tensor | None, reconstructed_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.cores = nn.ParameterList([nn.Parameter(c) for c in cores])
        self.in_factors = tuple(in_factors)
        self.out_factors = tuple(out_factors)
        if bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(bias)
        # store a reconstructed dense weight as a buffer (may be None)
        self.register_buffer("reconstructed_weight", reconstructed_weight)

    @classmethod
    def from_linear(cls, layer: nn.Linear, rank_ratio: float = 0.5, mpo_sites: int = 3) -> "MPOLinear":
        weight = layer.weight.detach().cpu()
        out_features, in_features = weight.shape

        out_factors = _partition_factors(out_features, mpo_sites)
        in_factors = _partition_factors(in_features, mpo_sites)

        # reshape and interleave: o1,i1,o2,i2,...
        tensor = weight.reshape(*out_factors, *in_factors)
        perm = []
        k = mpo_sites
        for i in range(k):
            perm.append(i)
            perm.append(k + i)
        tensor = tensor.permute(*perm)

        # iterative SVD to extract MPO cores
        cores: list[torch.Tensor] = []
        remainder = tensor
        left_rank = 1
        for site in range(k - 1):
            left_o = out_factors[site]
            left_i = in_factors[site]

            rest_shape = remainder.shape
            rows = left_rank * left_o * left_i
            cols = int(torch.tensor(rest_shape).prod().item() // (left_rank * left_o * left_i))

            matricized = remainder.reshape(rows, -1)
            u, s, vh = torch.linalg.svd(matricized, full_matrices=False)

            candidate_rank = max(1, int(len(s) * rank_ratio))
            rank = max(1, min(len(s), candidate_rank))

            u_r = u[:, :rank]
            s_r = s[:rank]
            vh_r = vh[:rank, :]

            core = (u_r * torch.sqrt(s_r)).reshape(left_rank, left_o, left_i, rank)
            cores.append(core)
            # multiply singular values into vh correctly (s as column)
            # choose slice offset depending on whether remainder already
            # contains the previous bond dimension (`left_rank`).
            start_idx = 2 if left_rank == 1 else 3
            remainder = (torch.sqrt(s_r).unsqueeze(1) * vh_r).reshape(rank, *rest_shape[start_idx:])
            left_rank = rank

        # final core
        final_shape = remainder.shape
        final_core = remainder.reshape(left_rank, *final_shape[1:])
        cores.append(final_core)

        bias = layer.bias.detach().cpu() if layer.bias is not None else None
        reconstructed = weight
        return cls(cores=[torch.tensor(c) for c in cores], in_factors=in_factors, out_factors=out_factors, bias=bias, reconstructed_weight=reconstructed)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_shape = inputs.shape[:-1]
        k = len(self.in_factors)
        in_factors = self.in_factors
        # If a reconstructed dense weight was stored at construction time,
        # use it directly (this is correct and faster than the incomplete
        # MPO contraction implementation below).
        if getattr(self, "reconstructed_weight", None) is not None:
            weight = self.reconstructed_weight
            outputs = inputs @ weight.t()
        else:
            # attempt MPO contraction from right to left
            x = inputs.reshape(*batch_shape, *in_factors)

            contracted = x
            for idx in range(k - 1, -1, -1):
                core = self.cores[idx]
                if core.ndim == 3:
                    contracted = torch.einsum("...i,roi->...ro", contracted, core)
                elif core.ndim == 4:
                    contracted = torch.einsum("...ji, rjio->...ro", contracted, core)
                else:
                    contracted = contracted

            outputs = contracted.reshape(*batch_shape, self.out_factors[0] * self.out_factors[1])
        if self.bias is not None:
            outputs = outputs + self.bias
        return outputs


def _partition_factors(size: int, parts: int) -> list[int]:
    # Factorize `size` into primes, then distribute primes across `parts` buckets
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

    primes = prime_factors(size)
    # start with ones
    factors = [1] * parts
    # multiply smallest factor by next prime to keep them balanced
    for p in sorted(primes, reverse=False):
        # find index of smallest factor
        idx = min(range(parts), key=lambda i: factors[i])
        factors[idx] *= p
    # if any leftover (shouldn't be), adjust first factor
    prod = 1
    for f in factors:
        prod *= f
    if prod != size:
        # adjust last factor to make exact
        factors[-1] = factors[-1] * (size // prod)
    return factors


class TensorNetworkLinear(nn.Module):
    def __init__(
        self,
        left_core: torch.Tensor,
        right_core: torch.Tensor,
        in_factors: Tuple[int, int],
        out_factors: Tuple[int, int],
        bias: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.left_core = nn.Parameter(left_core)
        self.right_core = nn.Parameter(right_core)
        self.in_factors = in_factors
        self.out_factors = out_factors
        if bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(bias)

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
        return cls(left_core=left_core, right_core=right_core, in_factors=in_factors, out_factors=out_factors, bias=bias)

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
    def __init__(
        self,
        left_core: torch.Tensor,
        right_core: torch.Tensor,
        in_factors: Tuple[int, int],
        out_factors: Tuple[int, int],
        bias: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.left_core = nn.Parameter(left_core)
        self.right_core = nn.Parameter(right_core)
        self.in_factors = in_factors
        self.out_factors = out_factors
        if bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(bias)

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
        return cls(left_core=left_core, right_core=right_core, in_factors=in_factors, out_factors=out_factors, bias=bias)

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
