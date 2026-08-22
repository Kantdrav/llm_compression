from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from .adaptive_mpo import AdaptiveMPOCompressor
from .compressors import DynamicQuantizationCompressor, TensorNetworkCompressor, count_parameters
from .config import CompressionConfig, ExportConfig, compression_config_to_dict
from .export import EdgeExporter
from .healing import HealingConfig, heal_model
from .manifest import ModelManifest, write_manifest
from .paper_mpo import ResearchMPOCompressor


@dataclass(slots=True)
class CompressionRunResult:
    manifest: ModelManifest
    manifest_path: Path
    export_dir: Path


def _state_dict_size_bytes(model: torch.nn.Module) -> int:
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


def _format_size(size_bytes: int) -> str:
    size_mb = size_bytes / (1024 * 1024)
    size_gb = size_bytes / (1024 * 1024 * 1024)
    if size_gb >= 1:
        return f"{size_gb:.2f} GB"
    return f"{size_mb:.2f} MB"


class CompressionPipeline:
    def __init__(self, compression: CompressionConfig, export: ExportConfig | None = None) -> None:
        self.compression = compression
        self.export = export or ExportConfig(output_dir=compression.output_dir)

    def run(self) -> CompressionRunResult:
        model = AutoModelForCausalLM.from_pretrained(
            self.compression.model_id,
            low_cpu_mem_usage=True,
            trust_remote_code=self.compression.trust_remote_code,
        )
        original_parameters = count_parameters(model)
        original_model_size_bytes = _state_dict_size_bytes(model)

        compressor = self._build_compressor()
        compressed_model = compressor.compress(model)
        healing_batches = self._build_calibration_batches(model)
        if self.compression.heal_steps > 0:
            compressed_model = heal_model(
                compressed_model,
                model,
                healing_batches,
                HealingConfig(
                    steps=self.compression.heal_steps,
                    learning_rate=self.compression.heal_learning_rate,
                    weight_decay=self.compression.heal_weight_decay,
                ),
            )
        compressed_parameters = count_parameters(compressed_model)
        compressed_model_size_bytes = _state_dict_size_bytes(compressed_model)

        size_ratio = compressed_model_size_bytes / max(original_model_size_bytes, 1)
        size_reduction_percent = (1.0 - size_ratio) * 100.0

        exporter = EdgeExporter(self.export.output_dir)
        export_result = exporter.export_bundle(compressed_model)

        print("\n=== Compression Report ===")
        print(f"Method:                 {self.compression.method}")
        if self.compression.method == "paper_mpo":
            print(f"Bond dimension (chi):   {self.compression.bond_dim}")
            print(f"MPO sites:              {self.compression.mpo_sites}")
        else:
            print(f"Rank selection:         {'adaptive' if self.compression.adaptive_rank else 'fixed'}")
            if self.compression.adaptive_rank:
                print(f"Target reduction:       {self.compression.target_reduction * 100:.2f}%")
                print(f"Energy threshold:       {self.compression.adaptive_energy_threshold:.6f}")
        print(f"Model:                  {self.compression.model_id}")
        print(f"Original parameters:    {original_parameters:,}")
        print(f"Compressed parameters:  {compressed_parameters:,}")
        print(f"Parameter ratio:        {compressed_parameters / max(original_parameters, 1):.4f}")
        print(f"Original model size:    {_format_size(original_model_size_bytes)}")
        print(f"Compressed model size:  {_format_size(compressed_model_size_bytes)}")
        print(f"Size ratio:             {size_ratio:.4f}")
        print(f"Size reduction:         {size_reduction_percent:.2f}%")
        print("==========================\n")

        manifest = ModelManifest(
            model_id=self.compression.model_id,
            method=self.compression.method,
            artifact_files=export_result.artifact_files,
            compression_config=compression_config_to_dict(self.compression),
            metrics={
                "original_parameters": original_parameters,
                "compressed_parameters": compressed_parameters,
                "parameter_ratio": round(compressed_parameters / max(original_parameters, 1), 6),
                "original_model_size_bytes": original_model_size_bytes,
                "compressed_model_size_bytes": compressed_model_size_bytes,
                "size_ratio": round(size_ratio, 6),
                "size_reduction_percent": round(size_reduction_percent, 2),
                "target_device": self.compression.target_device,
                "heal_steps": self.compression.heal_steps,
            },
        )
        manifest_path = write_manifest(manifest, self.export.output_dir)
        return CompressionRunResult(manifest=manifest, manifest_path=manifest_path, export_dir=self.export.output_dir)

    def _build_compressor(self):
        if self.compression.method == "quantize":
            return DynamicQuantizationCompressor(backend=self.compression.quantization_backend)
        if self.compression.method == "mpo":
            if self.compression.adaptive_rank:
                return AdaptiveMPOCompressor(
                    rank_ratio=self.compression.rank_ratio,
                    layer_policy=self.compression.layer_policy,
                    energy_threshold=self.compression.adaptive_energy_threshold,
                    target_reduction=self.compression.target_reduction,
                )
            from .compressors import MPOCompressor
            return MPOCompressor(rank_ratio=self.compression.rank_ratio, layer_policy=self.compression.layer_policy)
        if self.compression.method == "paper_mpo":
            return ResearchMPOCompressor(
                bond_dim=self.compression.bond_dim,
                mpo_sites=self.compression.mpo_sites,
                layer_policy=self.compression.layer_policy,
            )
        return TensorNetworkCompressor(rank_ratio=self.compression.rank_ratio, layer_policy=self.compression.layer_policy)

    def _build_calibration_batches(self, model: torch.nn.Module):
        tokenizer = AutoTokenizer.from_pretrained(
            self.compression.model_id,
            use_fast=True,
            trust_remote_code=self.compression.trust_remote_code,
        )
        vocab_size = getattr(tokenizer, "vocab_size", None) or getattr(getattr(model, "config", None), "vocab_size", 32000)
        for _ in range(self.compression.calibration_batches):
            yield {
                "input_ids": torch.randint(
                    0,
                    int(vocab_size),
                    (self.compression.calibration_batch_size, self.compression.calibration_sequence_length),
                ),
            }
