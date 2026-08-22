from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
from torch import nn

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:  # pragma: no cover
    Conv1D = None

from .compressors import MPOLinear
from .config import CompressionPolicy


@dataclass(slots=True)
class ResearchMPOCompressor:
    """Research-inspired MPO compressor with an explicit bond-dimension budget.

    This follows the practical workflow described in the supplied research notes:
    use TT/MPO-SVD, protect early/sensitive modules through a layer policy, and
    expose an explicit bond dimension rather than hiding it behind a rank ratio.
    It is not a claim of byte-for-byte reproduction of any particular paper.
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
                    ratio = self._rank_ratio(child.weight.detach().cpu(), self.bond_dim)
                    setattr(module, child_name, MPOLinear.from_linear(child, ratio, mpo_sites=self.mpo_sites))
            elif Conv1D is not None and isinstance(child, Conv1D):
                if self._should_compress(child_path):
                    weight = child.weight.detach().cpu().t()
                    ratio = self._rank_ratio(weight, self.bond_dim)
                    setattr(module, child_name, MPOLinear.from_conv1d(child, ratio, mpo_sites=self.mpo_sites))
            else:
                self._compress_module(child, child_path)

    def _should_compress(self, module_path: str) -> bool:
        return not any(re.search(pattern, module_path) for pattern in self.layer_policy.skip_module_patterns)

    @staticmethod
    def _rank_ratio(weight: torch.Tensor, bond_dim: int) -> float:
        """Convert an explicit MPO bond dimension into the existing TT-SVD API.

        The existing MPOLinear expects a fraction of the available singular
        values. Computing the maximum feasible rank lets us express the paper's
        chi/bond-dimension idea without replacing the tested contraction code.
        """
        out_features, in_features = weight.shape
        # For a 3-site decomposition the first and second TT bonds cannot exceed
        # the corresponding matricization dimensions. A conservative global cap
        # is sufficient because MPOLinear also clamps the rank at every site.
        max_rank = max(1, min(out_features * in_features, max(out_features, in_features)))
        return min(1.0, max(1, bond_dim) / max_rank)
