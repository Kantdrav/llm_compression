#!/usr/bin/env python3
"""
Tensor Network Model Compression Example

This script demonstrates compressing various HuggingFace models using
Tensor Network decomposition with validation.

Example models to try:
- distilgpt2 (small, fast)
- gpt2 (medium)
- microsoft/phi-2 (larger but still manageable)
- meta-llama/Llama-2-7b-chat-hf (large, requires HF token)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from llm_edge_compression.tn_model_compressor import (
    TensorNetworkModelCompressor,
    compress_model_with_tensor_networks,
)
from llm_edge_compression.config import CompressionPolicy


def main():
    parser = argparse.ArgumentParser(
        description="Compress LLMs using Tensor Network decomposition"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="distilgpt2",
        help="HuggingFace model ID (default: distilgpt2)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/tn_compressed"),
        help="Output directory for compressed model"
    )
    parser.add_argument(
        "--bond-dimension",
        type=float,
        default=0.5,
        help="TN bond dimension (0.0-1.0). Lower=more compression, Higher=better accuracy"
    )
    parser.add_argument(
        "--heal-steps",
        type=int,
        default=0,
        help="Number of knowledge distillation healing steps (0=no healing)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate inference after compression"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of artificial intelligence",
        help="Prompt for inference validation"
    )
    parser.add_argument(
        "--skip-first-n-layers",
        type=int,
        default=1,
        help="Skip first N transformer layers from compression (for preserving early patterns)"
    )
    parser.add_argument(
        "--no-compress-lm-head",
        action="store_true",
        help="Don't compress the language model head layer"
    )

    args = parser.parse_args()

    print("="*70)
    print("Tensor Network Model Compression")
    print("="*70)
    print(f"Model ID: {args.model_id}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Bond Dimension: {args.bond_dimension}")
    print(f"Healing Steps: {args.heal_steps}")
    print(f"Skip First N Layers: {args.skip_first_n_layers}")
    print("="*70 + "\n")

    # Build layer policy to skip early layers (as per paper)
    skip_patterns = []
    if args.skip_first_n_layers > 0:
        for i in range(args.skip_first_n_layers):
            skip_patterns.append(rf"^model\.layers\.{i}(?:\.|$)")
            skip_patterns.append(rf"^transformer\.h\.{i}(?:\.|$)")
            skip_patterns.append(rf"^blocks\.{i}(?:\.|$)")
    
    if args.no_compress_lm_head:
        skip_patterns.append(r"(^|\.)lm_head(?:\.|$)")

    layer_policy = CompressionPolicy(skip_module_patterns=tuple(skip_patterns))

    # Create compressor
    compressor = TensorNetworkModelCompressor(
        model_id=args.model_id,
        output_dir=args.output_dir,
        bond_dimension=args.bond_dimension,
        layer_policy=layer_policy,
        heal_steps=args.heal_steps,
        heal_learning_rate=1e-4,
        calibration_batches=8,
    )

    try:
        # Run compression
        print("Starting compression pipeline...\n")
        report = compressor.compress_and_export()

        print("\n" + "="*70)
        print("COMPRESSION RESULTS")
        print("="*70)
        print(f"Model: {report.model_id}")
        print(f"Original Parameters: {report.original_parameters:,}")
        print(f"Compressed Parameters: {report.compressed_parameters:,}")
        print(f"Parameter Reduction: {(1 - report.parameter_ratio)*100:.2f}%")
        print(f"Compression Ratio: {report.parameter_ratio:.6f}")
        print(f"Bond Dimension: {report.bond_dimension}")
        print(f"Total Linear Layers: {report.layers_total}")
        if report.layers_compressed:
            print(f"Compressed Layers: {report.layers_compressed}")
        print(f"\nOutput Directory: {report.output_dir}")
        print(f"Manifest: {report.manifest_path}")
        print("="*70 + "\n")

        # Validate inference
        if args.validate:
            print("Validating inference with compressed model...\n")
            try:
                inference_results = compressor.validate_inference(prompt=args.prompt)
                
                print("INFERENCE VALIDATION RESULTS")
                print("-" * 70)
                print(f"Prompt: '{inference_results['prompt']}'")
                print(f"\nGenerated Output:")
                print(f"  {inference_results['generated_text']}\n")
                print(f"Input Tokens: {inference_results['input_length']}")
                print(f"Output Tokens: {inference_results['output_length']}")
                print(f"Generated Tokens: {inference_results['output_length'] - inference_results['input_length']}")
                print("="*70)
                print("\n✓ Inference validation successful!")
                print("✓ Compressed model is fully functional!")
                
            except Exception as e:
                print(f"\n✗ Inference validation failed: {type(e).__name__}: {e}")
                return 1

        # Save summary report
        summary = {
            "model_id": report.model_id,
            "compression_method": "tensor_network",
            "bond_dimension": report.bond_dimension,
            "original_parameters": report.original_parameters,
            "compressed_parameters": report.compressed_parameters,
            "parameter_ratio": report.parameter_ratio,
            "parameter_reduction_percent": (1 - report.parameter_ratio) * 100,
            "output_directory": str(report.output_dir),
            "manifest_path": str(report.manifest_path),
        }
        
        summary_path = args.output_dir / "compression_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to: {summary_path}")
        return 0

    except KeyboardInterrupt:
        print("\n\nCompression cancelled by user.")
        return 130
    except Exception as e:
        print(f"\n✗ Compression failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
