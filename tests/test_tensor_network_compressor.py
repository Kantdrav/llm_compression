import torch
from torch import nn

from llm_edge_compression.compressors import TensorNetworkCompressor, TensorNetworkLinear, count_parameters
from llm_edge_compression.config import CompressionPolicy


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


def test_tensor_network_compressor_reduces_parameters_and_preserves_shape() -> None:
    torch.manual_seed(0)
    model = TinyModel().eval()
    baseline = count_parameters(model)

    compressed = TensorNetworkCompressor(rank_ratio=0.5).compress(model)
    compressed_parameters = count_parameters(compressed)

    sample = torch.randn(2, 16)
    output = compressed(sample)

    assert compressed_parameters < baseline
    assert output.shape == (2, 8)
    assert any(isinstance(module, TensorNetworkLinear) for module in compressed.modules())


def test_tensor_network_compressor_can_skip_sensitive_layers() -> None:
    model = TinyModel().eval()

    policy = CompressionPolicy(skip_module_patterns=(r"^net\.0$",))
    compressed = TensorNetworkCompressor(rank_ratio=0.5, layer_policy=policy).compress(model)

    assert isinstance(compressed.net[0], nn.Linear)
    assert isinstance(compressed.net[2], TensorNetworkLinear)
