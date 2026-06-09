from pathlib import Path

import torch
from torch import nn

from llm_edge_compression.compressors import TensorNetworkCompressor, TensorNetworkLinear
from llm_edge_compression.config import CompressionConfig, CompressionPolicy


class PaperStyleBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Linear(8, 8)
        self.mlp = nn.Linear(8, 8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.self_attn(inputs))


class PaperStyleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([PaperStyleBlock(), PaperStyleBlock(), PaperStyleBlock()])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            inputs = layer(inputs)
        return inputs


def test_default_policy_skips_early_layers() -> None:
    model = PaperStyleModel().eval()
    compressed = TensorNetworkCompressor(rank_ratio=0.5, layer_policy=CompressionPolicy.paper_default()).compress(model)

    assert isinstance(compressed.model.layers[0].self_attn, nn.Linear)
    assert isinstance(compressed.model.layers[1].mlp, nn.Linear)
    assert isinstance(compressed.model.layers[2].self_attn, TensorNetworkLinear)


def test_compression_config_uses_paper_default_policy(tmp_path: Path) -> None:
    config = CompressionConfig(model_id="demo", output_dir=tmp_path)

    assert config.layer_policy.skip_module_patterns