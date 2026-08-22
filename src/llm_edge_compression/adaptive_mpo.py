from __future__ import annotations

import re

import torch
from torch import nn

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:  # pragma: no cover
    Conv1D = None

from .compressors import MPOCompressor, MPOLinear
from .config import CompressionPolicy


class AdaptiveMPOCompressor(MPOCompressor):
    """MPO compressor with per-layer rank selection based on spectral energy.

    The existing fixed ``rank_ratio`` path remains unchanged. Adaptive mode
    chooses a different rank ratio for each compressible layer. A target
    reduction is treated conservatively: adaptive selection never requests a
    rank below the corresponding retained-parameter floor solely to chase the
    target. This favors inference fidelity over forcing an exact size target.
    """

    def __init__(
        self,
        rank_ratio: float = 0.5,
        layer_policy: CompressionPolicy | None = None,
        energy_threshold: float = 0.995,
        target_reduction: float = 0.30,
    ) -> None:
        super().__init__(rank_ratio=rank_ratio, layer_policy=layer_policy or CompressionPolicy())
        if not 0.90 <= energy_threshold <= 0.999999:
            raise ValueError("energy_threshold must be between 0.90 and 0.999999")
        if not 0.0 <= target_reduction < 1.0:
            raise ValueError("target_reduction must be in [0, 1)")
        self.energy_threshold = energy_threshold
        self.target_reduction = target_reduction

    def _rank_ratio_for(self, module_path: str, layer: nn.Module | None = None) -> float:
        override = self.layer_policy.layer_rank_overrides.get(module_path)
        if override is not None:
            return override
        if layer is None:
            return self.rank_ratio

        weight = layer.weight.detach().float().cpu()
        if Conv1D is not None and isinstance(layer, Conv1D):
            weight = weight.t()

        # Use the layer's singular-value energy as a stable proxy for how much
        # of its spectrum can be discarded. The final MPO TT-SVD still applies
        # this ratio at every MPO site.
        with torch.no_grad():
            s = torch.linalg.svdvals(weight)
            energy = torch.cumsum(s.square(), dim=0)
            total = energy[-1].clamp_min(torch.finfo(energy.dtype).eps)
            ranks = torch.nonzero(energy / total >= self.energy_threshold, as_tuple=False)
            energy_rank = int(ranks[0].item() + 1) if ranks.numel() else len(s)

        full_rank = len(s)
        spectral_ratio = energy_rank / max(full_rank, 1)

        # Keep at least the fraction corresponding to the requested target
        # reduction. For a 30% target, the adaptive layer rank cannot be below
        # 70% of its available rank. This makes the target conservative rather
        # than forcing every layer to lose 30% of its rank.
        retained_floor = 1.0 - self.target_reduction
        return max(1.0 / max(full_rank, 1), min(1.0, max(spectral_ratio, retained_floor)))

    def _compress_module(self, module: nn.Module, module_path: str = "") -> None:
        for child_name, child in list(module.named_children()):
            child_path = f"{module_path}.{child_name}" if module_path else child_name
            if isinstance(child, nn.Linear):
                if self._should_compress(child_path):
                    ratio = self._rank_ratio_for(child_path, child)
                    setattr(module, child_name, MPOLinear.from_linear(child, ratio, mpo_sites=3))
            elif Conv1D is not None and isinstance(child, Conv1D):
                if self._should_compress(child_path):
                    ratio = self._rank_ratio_for(child_path, child)
                    setattr(module, child_name, MPOLinear.from_conv1d(child, ratio, mpo_sites=3))
            else:
                self._compress_module(child, child_path)
