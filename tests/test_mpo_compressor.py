from copy import deepcopy

import torch
from torch import nn

from llm_edge_compression.compressors import MPOLinear, MPOCompressor, count_parameters


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def test_mpo_compressor_reduces_parameters() -> None:
    torch.manual_seed(0)
    model = TinyModel().eval()
    baseline = count_parameters(model)

    compressed = MPOCompressor(rank_ratio=0.5).compress(model)
    compressed_parameters = count_parameters(compressed)

    sample = torch.randn(2, 16)
    output = compressed(sample)

    assert compressed_parameters < baseline
    assert output.shape == (2, 8)


def test_mpo_full_rank_preserves_output_closely() -> None:
    torch.manual_seed(0)
    model = TinyModel().eval()
    reference = deepcopy(model).eval()

    sample = torch.randn(4, 16)
    expected = reference(sample)

    compressed = MPOCompressor(rank_ratio=1.0).compress(model)
    actual = compressed(sample)

    assert any(isinstance(module, MPOLinear) for module in compressed.modules())
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)