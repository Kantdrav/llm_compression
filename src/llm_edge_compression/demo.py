from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .compressors import TensorNetworkCompressor, count_parameters
from .config import ExportConfig
from .export import EdgeExporter
from .manifest import ModelManifest, write_manifest


class TinyEdgeLLM(nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_size: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed(token_ids)
        hidden_states = self.block(hidden_states)
        return self.lm_head(hidden_states)


@dataclass(slots=True)
class DemoResult:
    output_dir: Path
    manifest_path: Path
    parameter_ratio: float


def run_demo(output_dir: Path, rank_ratio: float = 0.5) -> DemoResult:
    model = TinyEdgeLLM().eval()
    original_parameters = count_parameters(model)

    compressed_model = TensorNetworkCompressor(rank_ratio=rank_ratio).compress(model)
    compressed_parameters = count_parameters(compressed_model)

    exporter = EdgeExporter(output_dir)
    export_result = exporter.export_bundle(compressed_model)

    manifest = ModelManifest(
        model_id="demo/tiny-edge-llm",
        method="tensor_inspired",
        artifact_files=export_result.artifact_files,
        metrics={
            "original_parameters": original_parameters,
            "compressed_parameters": compressed_parameters,
            "parameter_ratio": round(compressed_parameters / max(original_parameters, 1), 6),
            "sample_input_shape": [1, 8],
        },
    )
    manifest_path = write_manifest(manifest, output_dir)
    return DemoResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        parameter_ratio=compressed_parameters / max(original_parameters, 1),
    )
