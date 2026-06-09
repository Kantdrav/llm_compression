from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

CompressionMethod = str
ExportFormat = str


@dataclass(slots=True)
class CompressionPolicy:
    skip_module_patterns: tuple[str, ...] = ()
    layer_rank_overrides: dict[str, float] = field(default_factory=dict)

    @classmethod
    def paper_default(cls) -> "CompressionPolicy":
        return cls(
            skip_module_patterns=(
                r"^model\.layers\.[01](?:\.|$)",
                r"^transformer\.h\.[01](?:\.|$)",
                r"^blocks\.[01](?:\.|$)",
            )
        )


@dataclass(slots=True)
class CompressionConfig:
    model_id: str
    output_dir: Path
    method: CompressionMethod = "tensor_inspired"
    rank_ratio: float = 0.5
    target_device: str = "cpu"
    quantization_backend: str = "fbgemm"
    layer_policy: CompressionPolicy = field(default_factory=CompressionPolicy.paper_default)
    heal_steps: int = 0
    heal_learning_rate: float = 1e-4
    heal_weight_decay: float = 0.0
    calibration_batches: int = 8
    calibration_batch_size: int = 2
    calibration_sequence_length: int = 16


@dataclass(slots=True)
class ExportConfig:
    output_dir: Path
    export_format: ExportFormat = "bundle"
    opset_version: int = 17
