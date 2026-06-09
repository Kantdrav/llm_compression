from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(slots=True)
class ExportResult:
    artifact_files: list[str]


class EdgeExporter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def export_bundle(self, model: nn.Module) -> ExportResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.output_dir / "compressed_model.pt"
        torch.save(model.state_dict(), model_path)
        return ExportResult(artifact_files=[model_path.name])
