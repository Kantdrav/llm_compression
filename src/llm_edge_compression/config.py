from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
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


def compression_config_to_dict(config: CompressionConfig) -> dict[str, Any]:
    return {
        "model_id": config.model_id,
        "output_dir": config.output_dir.as_posix(),
        "method": config.method,
        "rank_ratio": config.rank_ratio,
        "target_device": config.target_device,
        "quantization_backend": config.quantization_backend,
        "layer_policy": {
            "skip_module_patterns": list(config.layer_policy.skip_module_patterns),
            "layer_rank_overrides": dict(config.layer_policy.layer_rank_overrides),
        },
        "heal_steps": config.heal_steps,
        "heal_learning_rate": config.heal_learning_rate,
        "heal_weight_decay": config.heal_weight_decay,
        "calibration_batches": config.calibration_batches,
        "calibration_batch_size": config.calibration_batch_size,
        "calibration_sequence_length": config.calibration_sequence_length,
    }


def compression_config_from_dict(data: dict[str, Any]) -> CompressionConfig:
    layer_policy_data = data.get("layer_policy") or {}
    return CompressionConfig(
        model_id=data["model_id"],
        output_dir=Path(data["output_dir"]),
        method=data.get("method", "tensor_inspired"),
        rank_ratio=float(data.get("rank_ratio", 0.5)),
        target_device=data.get("target_device", "cpu"),
        quantization_backend=data.get("quantization_backend", "fbgemm"),
        layer_policy=CompressionPolicy(
            skip_module_patterns=tuple(layer_policy_data.get("skip_module_patterns", ())),
            layer_rank_overrides=dict(layer_policy_data.get("layer_rank_overrides", {})),
        ),
        heal_steps=int(data.get("heal_steps", 0)),
        heal_learning_rate=float(data.get("heal_learning_rate", 1e-4)),
        heal_weight_decay=float(data.get("heal_weight_decay", 0.0)),
        calibration_batches=int(data.get("calibration_batches", 8)),
        calibration_batch_size=int(data.get("calibration_batch_size", 2)),
        calibration_sequence_length=int(data.get("calibration_sequence_length", 16)),
    )
