from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from .compressors import DynamicQuantizationCompressor, TensorNetworkCompressor, count_parameters
from .config import CompressionConfig, ExportConfig, compression_config_to_dict
from .export import EdgeExporter
from .healing import HealingConfig, heal_model
from .manifest import ModelManifest, write_manifest


@dataclass(slots=True)
class CompressionRunResult:
	manifest: ModelManifest
	manifest_path: Path
	export_dir: Path


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

		exporter = EdgeExporter(self.export.output_dir)
		export_result = exporter.export_bundle(compressed_model)

		manifest = ModelManifest(
			model_id=self.compression.model_id,
			method=self.compression.method,
			artifact_files=export_result.artifact_files,
			compression_config=compression_config_to_dict(self.compression),
			metrics={
				"original_parameters": original_parameters,
				"compressed_parameters": compressed_parameters,
				"parameter_ratio": round(compressed_parameters / max(original_parameters, 1), 6),
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
			from .compressors import MPOCompressor
			return MPOCompressor(rank_ratio=self.compression.rank_ratio, layer_policy=self.compression.layer_policy)
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