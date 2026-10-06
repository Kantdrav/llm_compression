from __future__ import annotations

import torch
from torch import nn

from llm_edge_compression.layer_pruner import (
    LayerPruningCompressor,
    angular_distance,
    get_transformer_layers,
)


class TinyDecoder(nn.Module):
    def __init__(self, layers: int = 6, hidden: int = 8):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(layers)]
        )

    def forward(self, input_ids):
        x = input_ids.float()
        for layer in self.model.layers:
            x = layer(x)
        return x


def test_angular_distance_is_zero_for_identical_vectors():
    x = torch.ones(8)
    assert angular_distance(x, x) < 1e-6


def test_deepest_pruning_preserves_final_layer():
    model = TinyDecoder(layers=6)
    compressor = LayerPruningCompressor(num_remove=2, strategy="deepest")
    compressor.fit(model, [])
    assert compressor.result is not None
    assert compressor.result.removed_layers == [3, 4]

    compressed = compressor.compress(model)
    assert len(get_transformer_layers(compressed)) == 4


def test_selected_layer_pruning_is_deterministic():
    model = TinyDecoder(layers=5)
    compressor = LayerPruningCompressor(
        selected_layers=[1, 2],
        strategy="similarity",
    )
    compressor.fit(model, [])
    compressed = compressor.compress(model)
    assert len(get_transformer_layers(compressed)) == 3
    assert compressor.result.removed_layers == [1, 2]


def test_prune_ratio():
    model = TinyDecoder(layers=10)
    compressor = LayerPruningCompressor(
        prune_ratio=0.3,
        strategy="deepest",
    )
    compressor.fit(model, [])
    assert len(compressor.result.removed_layers) == 3
