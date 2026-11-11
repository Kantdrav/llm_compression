import torch
from torch import nn

from llm_edge_compression.compressors import MPOLinear, MixedCompressionCompressor
from llm_edge_compression.config import CompressionPolicy


class MixedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mpo_layer = nn.Linear(16, 16)
        self.quantized_layer = nn.Linear(16, 8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.quantized_layer(self.mpo_layer(inputs))


def test_mixed_compressor_applies_mpo_and_quantization() -> None:
    policy = CompressionPolicy(skip_module_patterns=(r"^quantized_layer$",))
    model = MixedCompressionCompressor(
        rank_ratio=0.5,
        layer_policy=policy,
    ).compress(MixedModel().eval())

    assert isinstance(model.mpo_layer, MPOLinear)
    assert isinstance(model.quantized_layer, torch.ao.nn.quantized.dynamic.Linear)
    assert model(torch.randn(2, 16)).shape == (2, 8)