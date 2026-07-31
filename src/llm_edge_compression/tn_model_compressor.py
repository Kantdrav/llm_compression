"""
Tensor Network Model Compressor

This module demonstrates the tensorization and compression of LLM layers
using Tensor Networks (TNs) with controllable bond dimensions. The technique
involves decomposing self-attention (SA) and multi-layer perceptron (MLP)
layers using TN decomposition, which effectively truncates correlations
present in the model.

The degree of truncation is controlled via the bond dimension of the TN,
enabling significant reduction in memory size and parameters while
maintaining model accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .compressors import TensorNetworkCompressor, count_parameters
from .config import CompressionConfig, ExportConfig, compression_config_to_dict, CompressionPolicy
from .export import EdgeExporter
from .healing import HealingConfig, heal_model
from .manifest import ModelManifest, write_manifest


@dataclass(slots=True)
class TNCompressionReport:
    """Report detailing TN compression results."""
    model_id: str
    original_parameters: int
    compressed_parameters: int
    parameter_ratio: float
    bond_dimension: float
    method: str = "tensor_network"
    output_dir: Path = None
    manifest_path: Path = None
    layers_compressed: int = 0
    layers_total: int = 0


class TensorNetworkModelCompressor:
    """
    Compresses language models using Tensor Network decomposition.
    
    This compressor applies tensorization to linear layers (used in SA and MLP blocks)
    using a specific Tensor Network structure with configurable bond dimension
    (rank_ratio), which controls the degree of correlation truncation.
    """

    def __init__(
        self,
        model_id: str,
        output_dir: Path,
        bond_dimension: float = 0.5,
        layer_policy: CompressionPolicy = None,
        heal_steps: int = 0,
        heal_learning_rate: float = 1e-4,
        calibration_batches: int = 8,
        calibration_batch_size: int = 2,
        calibration_sequence_length: int = 16,
    ):
        """
        Initialize the Tensor Network compressor.
        
        Args:
            model_id: HuggingFace model identifier
            output_dir: Directory to save compressed artifacts
            bond_dimension: TN bond dimension (0.0-1.0), controls rank truncation
                           Higher values = less compression but better accuracy
                           Lower values = more compression but potential accuracy loss
            layer_policy: CompressionPolicy for skipping certain layers
            heal_steps: Number of healing steps using knowledge distillation
            heal_learning_rate: Learning rate for healing optimization
            calibration_batches: Number of calibration batches for healing
            calibration_batch_size: Batch size for calibration
            calibration_sequence_length: Sequence length for calibration
        """
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.bond_dimension = bond_dimension
        self.layer_policy = layer_policy or CompressionPolicy.paper_default()
        self.heal_steps = heal_steps
        self.heal_learning_rate = heal_learning_rate
        self.calibration_batches = calibration_batches
        self.calibration_batch_size = calibration_batch_size
        self.calibration_sequence_length = calibration_sequence_length
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compress_and_export(self) -> TNCompressionReport:
        """
        Load, compress, and export the model.
        
        Returns:
            TNCompressionReport with compression metrics
        """
        print(f"Loading model: {self.model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        original_parameters = count_parameters(model)
        total_layers = sum(1 for _ in model.modules() if isinstance(_, nn.Linear))
        
        print(f"Original model: {original_parameters:,} parameters")
        print(f"Total linear layers: {total_layers}")
        print(f"Bond dimension (rank_ratio): {self.bond_dimension}")
        
        # Apply Tensor Network compression
        print("\nApplying Tensor Network compression...")
        compressor = TensorNetworkCompressor(
            rank_ratio=self.bond_dimension,
            layer_policy=self.layer_policy
        )
        compressed_model = compressor.compress(model)
        
        compressed_parameters = count_parameters(compressed_model)
        print(f"Compressed model: {compressed_parameters:,} parameters")
        print(f"Compression ratio: {compressed_parameters / original_parameters:.4f}")
        
        # Optional: Apply healing/fine-tuning
        if self.heal_steps > 0:
            print(f"\nApplying knowledge distillation healing ({self.heal_steps} steps)...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
            calibration_batches = self._build_calibration_batches(
                tokenizer, 
                self.calibration_batches,
                self.calibration_batch_size,
                self.calibration_sequence_length
            )
            compressed_model = heal_model(
                compressed_model,
                model,
                calibration_batches,
                HealingConfig(
                    steps=self.heal_steps,
                    learning_rate=self.heal_learning_rate,
                    weight_decay=0.0
                )
            )
            print("Healing complete.")
        
        # Export compressed model
        print("\nExporting compressed model...")
        exporter = EdgeExporter(self.output_dir)
        export_result = exporter.export_bundle(compressed_model)
        
        # Create and save manifest
        config = CompressionConfig(
            model_id=self.model_id,
            output_dir=self.output_dir,
            method="tensor_inspired",
            rank_ratio=self.bond_dimension,
            layer_policy=self.layer_policy,
            heal_steps=self.heal_steps,
        )
        
        manifest = ModelManifest(
            model_id=self.model_id,
            method="tensor_network",
            artifact_files=export_result.artifact_files,
            compression_config=compression_config_to_dict(config),
            metrics={
                "original_parameters": original_parameters,
                "compressed_parameters": compressed_parameters,
                "parameter_ratio": round(compressed_parameters / original_parameters, 6),
                "bond_dimension": self.bond_dimension,
                "total_linear_layers": total_layers,
                "heal_steps": self.heal_steps,
            }
        )
        
        manifest_path = write_manifest(manifest, self.output_dir)
        
        return TNCompressionReport(
            model_id=self.model_id,
            original_parameters=original_parameters,
            compressed_parameters=compressed_parameters,
            parameter_ratio=compressed_parameters / original_parameters,
            bond_dimension=self.bond_dimension,
            output_dir=self.output_dir,
            manifest_path=manifest_path,
            layers_total=total_layers,
        )

    def _build_calibration_batches(self, tokenizer, num_batches: int, batch_size: int, seq_length: int):
        """Generate synthetic calibration batches for healing."""
        for _ in range(num_batches):
            token_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_length))
            yield {"input_ids": token_ids}

    def validate_inference(self, prompt: str = "The quick brown fox") -> dict[str, Any]:
        """
        Validate that the compressed model can perform inference.
        
        Args:
            prompt: Input text for inference
            
        Returns:
            Dictionary with inference results
        """
        print(f"\nValidating inference with compressed model...")
        print(f"Prompt: '{prompt}'")
        
        manifest_path = self.output_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")
        
        # Load tokenizer and compressed model
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Apply compression
        compressor = TensorNetworkCompressor(
            rank_ratio=self.bond_dimension,
            layer_policy=self.layer_policy
        )
        model = compressor.compress(model)
        
        # Load compressed weights
        state_dict = torch.load(self.output_dir / "compressed_model.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        model = model.eval()
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        # Generate with compressed model
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=32,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode output
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        results = {
            "prompt": prompt,
            "generated_text": generated_text,
            "input_length": input_ids.shape[1],
            "output_length": output_ids.shape[1],
        }
        
        print(f"Generated: {generated_text[:100]}...")
        print(f"Output length: {output_ids.shape[1]} tokens")
        
        return results


def compress_model_with_tensor_networks(
    model_id: str,
    output_dir: Path,
    bond_dimension: float = 0.5,
    heal_steps: int = 0,
    validate: bool = True,
    prompt: str = "The future of AI is",
) -> TNCompressionReport:
    """
    Convenience function to compress a model with TN decomposition.
    
    Args:
        model_id: HuggingFace model identifier
        output_dir: Directory for compressed artifacts
        bond_dimension: TN bond dimension (0.0-1.0)
        heal_steps: Number of healing steps
        validate: Whether to validate inference after compression
        prompt: Prompt for validation inference
        
    Returns:
        TNCompressionReport with results
    """
    compressor = TensorNetworkModelCompressor(
        model_id=model_id,
        output_dir=output_dir,
        bond_dimension=bond_dimension,
        heal_steps=heal_steps,
    )
    
    report = compressor.compress_and_export()
    
    if validate:
        try:
            inference_results = compressor.validate_inference(prompt)
            print("\n✓ Inference validation successful!")
            print(f"  Generated {len(inference_results['generated_text'])} characters")
        except Exception as e:
            print(f"\n✗ Inference validation failed: {e}")
    
    return report


if __name__ == "__main__":
    # Example: Compress a small model with TN decomposition
    report = compress_model_with_tensor_networks(
        model_id="distilgpt2",
        output_dir=Path("artifacts/tn_distilgpt2_compressed"),
        bond_dimension=0.5,
        heal_steps=0,
        validate=True,
    )
    
    print("\n" + "="*60)
    print("Tensor Network Compression Report")
    print("="*60)
    print(f"Model: {report.model_id}")
    print(f"Original Parameters: {report.original_parameters:,}")
    print(f"Compressed Parameters: {report.compressed_parameters:,}")
    print(f"Compression Ratio: {report.parameter_ratio:.4f}")
    print(f"Bond Dimension: {report.bond_dimension}")
    print(f"Output: {report.output_dir}")
    print(f"Manifest: {report.manifest_path}")
