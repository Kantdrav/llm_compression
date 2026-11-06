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
                r"(^|\.)lm_head(?:\.|$)",
            )
        )


@dataclass(slots=True)
class CompressionConfig:
    model_id: str
    output_dir: Path
    method: CompressionMethod = "tensor_inspired"
    rank_ratio: float = 0.5
    adaptive_rank: bool = False
    target_reduction: float = 0.30
    adaptive_energy_threshold: float = 0.995
    bond_dim: int = 16
    mpo_sites: int = 3
    target_device: str = "cpu"
    trust_remote_code: bool = False
    quantization_backend: str = "fbgemm"
    layer_policy: CompressionPolicy = field(default_factory=CompressionPolicy.paper_default)
    layer_pruning_strategy: str = "similarity"
    layer_pruning_ratio: float = 0.0
    layer_pruning_num_layers: int = 0
    layer_pruning_seed: int = 0
    layer_pruning_selected_layers: tuple[int, ...] = ()
    heal_steps: int = 0
    heal_learning_rate: float = 1e-4
    heal_weight_decay: float = 0.0
    calibration_batches: int = 8
    calibration_batch_size: int = 2
    calibration_sequence_length: int = 16
    qlora_healing: bool = False
    qlora_output_dir: Path | None = None
    qlora_dataset: str = "allenai/c4"
    qlora_dataset_config: str = "en"
    qlora_dataset_split: str = "train"
    qlora_max_samples: int = 256
    qlora_max_seq_length: int = 512
    qlora_max_steps: int = 500
    qlora_learning_rate: float = 2e-4
    qlora_warmup_ratio: float = 0.03
    qlora_weight_decay: float = 0.0
    qlora_batch_size: int = 1
    qlora_gradient_accumulation_steps: int = 16
    qlora_r: int = 16
    qlora_alpha: int = 32
    qlora_dropout: float = 0.05
    qlora_seed: int = 0
    qlora_compute_dtype: str = "bfloat16"


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
        "adaptive_rank": config.adaptive_rank,
        "target_reduction": config.target_reduction,
        "adaptive_energy_threshold": config.adaptive_energy_threshold,
        "bond_dim": config.bond_dim,
        "mpo_sites": config.mpo_sites,
        "target_device": config.target_device,
        "trust_remote_code": config.trust_remote_code,
        "quantization_backend": config.quantization_backend,
        "layer_policy": {
            "skip_module_patterns": list(config.layer_policy.skip_module_patterns),
            "layer_rank_overrides": dict(config.layer_policy.layer_rank_overrides),
        },
        "layer_pruning_strategy": config.layer_pruning_strategy,
        "layer_pruning_ratio": config.layer_pruning_ratio,
        "layer_pruning_num_layers": config.layer_pruning_num_layers,
        "layer_pruning_seed": config.layer_pruning_seed,
        "layer_pruning_selected_layers": list(config.layer_pruning_selected_layers),
        "heal_steps": config.heal_steps,
        "heal_learning_rate": config.heal_learning_rate,
        "heal_weight_decay": config.heal_weight_decay,
        "calibration_batches": config.calibration_batches,
        "calibration_batch_size": config.calibration_batch_size,
        "calibration_sequence_length": config.calibration_sequence_length,
        "qlora_healing": config.qlora_healing,
        "qlora_output_dir": config.qlora_output_dir.as_posix() if config.qlora_output_dir else None,
        "qlora_dataset": config.qlora_dataset,
        "qlora_dataset_config": config.qlora_dataset_config,
        "qlora_dataset_split": config.qlora_dataset_split,
        "qlora_max_samples": config.qlora_max_samples,
        "qlora_max_seq_length": config.qlora_max_seq_length,
        "qlora_max_steps": config.qlora_max_steps,
        "qlora_learning_rate": config.qlora_learning_rate,
        "qlora_warmup_ratio": config.qlora_warmup_ratio,
        "qlora_weight_decay": config.qlora_weight_decay,
        "qlora_batch_size": config.qlora_batch_size,
        "qlora_gradient_accumulation_steps": config.qlora_gradient_accumulation_steps,
        "qlora_r": config.qlora_r,
        "qlora_alpha": config.qlora_alpha,
        "qlora_dropout": config.qlora_dropout,
        "qlora_seed": config.qlora_seed,
        "qlora_compute_dtype": config.qlora_compute_dtype,
    }


def compression_config_from_dict(data: dict[str, Any]) -> CompressionConfig:
    layer_policy_data = data.get("layer_policy") or {}
    return CompressionConfig(
        model_id=data["model_id"],
        output_dir=Path(data["output_dir"]),
        method=data.get("method", "tensor_inspired"),
        rank_ratio=float(data.get("rank_ratio", 0.5)),
        adaptive_rank=bool(data.get("adaptive_rank", False)),
        target_reduction=float(data.get("target_reduction", 0.30)),
        adaptive_energy_threshold=float(data.get("adaptive_energy_threshold", 0.995)),
        bond_dim=int(data.get("bond_dim", 16)),
        mpo_sites=int(data.get("mpo_sites", 3)),
        target_device=data.get("target_device", "cpu"),
        trust_remote_code=bool(data.get("trust_remote_code", False)),
        quantization_backend=data.get("quantization_backend", "fbgemm"),
        layer_policy=CompressionPolicy(
            skip_module_patterns=tuple(layer_policy_data.get("skip_module_patterns", ())),
            layer_rank_overrides=dict(layer_policy_data.get("layer_rank_overrides", {})),
        ),
        layer_pruning_strategy=data.get("layer_pruning_strategy", "similarity"),
        layer_pruning_ratio=float(data.get("layer_pruning_ratio", 0.0)),
        layer_pruning_num_layers=int(data.get("layer_pruning_num_layers", 0)),
        layer_pruning_seed=int(data.get("layer_pruning_seed", 0)),
        layer_pruning_selected_layers=tuple(int(x) for x in data.get("layer_pruning_selected_layers", ())),
        heal_steps=int(data.get("heal_steps", 0)),
        heal_learning_rate=float(data.get("heal_learning_rate", 1e-4)),
        heal_weight_decay=float(data.get("heal_weight_decay", 0.0)),
        calibration_batches=int(data.get("calibration_batches", 8)),
        calibration_batch_size=int(data.get("calibration_batch_size", 2)),
        calibration_sequence_length=int(data.get("calibration_sequence_length", 16)),
        qlora_healing=bool(data.get("qlora_healing", False)),
        qlora_output_dir=Path(data["qlora_output_dir"]) if data.get("qlora_output_dir") else None,
        qlora_dataset=data.get("qlora_dataset", "allenai/c4"),
        qlora_dataset_config=data.get("qlora_dataset_config", "en"),
        qlora_dataset_split=data.get("qlora_dataset_split", "train"),
        qlora_max_samples=int(data.get("qlora_max_samples", 256)),
        qlora_max_seq_length=int(data.get("qlora_max_seq_length", 512)),
        qlora_max_steps=int(data.get("qlora_max_steps", 500)),
        qlora_learning_rate=float(data.get("qlora_learning_rate", 2e-4)),
        qlora_warmup_ratio=float(data.get("qlora_warmup_ratio", 0.03)),
        qlora_weight_decay=float(data.get("qlora_weight_decay", 0.0)),
        qlora_batch_size=int(data.get("qlora_batch_size", 1)),
        qlora_gradient_accumulation_steps=int(data.get("qlora_gradient_accumulation_steps", 16)),
        qlora_r=int(data.get("qlora_r", 16)),
        qlora_alpha=int(data.get("qlora_alpha", 32)),
        qlora_dropout=float(data.get("qlora_dropout", 0.05)),
        qlora_seed=int(data.get("qlora_seed", 0)),
        qlora_compute_dtype=data.get("qlora_compute_dtype", "bfloat16"),
    )
