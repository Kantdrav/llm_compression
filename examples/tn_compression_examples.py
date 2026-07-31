#!/usr/bin/env python3
"""
Advanced Tensor Network Compression Examples

Demonstrates best practices for compressing different models with TN decomposition.
"""

from pathlib import Path
import json
import sys

import torch
from llm_edge_compression.tn_model_compressor import TensorNetworkModelCompressor
from llm_edge_compression.config import CompressionPolicy


def example_1_aggressive_compression():
    """
    Example 1: Aggressive compression for edge deployment
    
    - Bond dimension: 0.4 (60% parameter reduction)
    - Compress all layers including first
    - No healing (fast)
    - Perfect for edge devices with memory constraints
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Aggressive Compression for Edge Deployment")
    print("="*70)
    
    compressor = TensorNetworkModelCompressor(
        model_id="distilgpt2",
        output_dir=Path("artifacts/tn_edge_aggressive"),
        bond_dimension=0.4,  # 60% reduction
        layer_policy=CompressionPolicy(skip_module_patterns=()),  # Compress all
        heal_steps=0,
    )
    
    report = compressor.compress_and_export()
    
    print(f"\n✓ Model: {report.model_id}")
    print(f"  Original: {report.original_parameters:,} parameters")
    print(f"  Compressed: {report.compressed_parameters:,} parameters")
    print(f"  Reduction: {(1 - report.parameter_ratio)*100:.1f}%")
    print(f"  Output: {report.output_dir}")
    
    # Validate
    results = compressor.validate_inference(prompt="Hello world")
    print(f"\n✓ Inference works: Generated {results['output_length']} tokens")
    return report


def example_2_balanced_compression():
    """
    Example 2: Balanced compression with healing
    
    - Bond dimension: 0.5 (50% reduction)
    - Skip first 2 layers (preserve early patterns)
    - Knowledge distillation healing (5 steps)
    - Good for production with reasonable accuracy
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Balanced Compression with Knowledge Distillation")
    print("="*70)
    
    # Skip first 2 layers
    policy = CompressionPolicy(
        skip_module_patterns=(
            r"^model\.layers\.[01](?:\.|$)",
            r"^transformer\.h\.[01](?:\.|$)",
        )
    )
    
    compressor = TensorNetworkModelCompressor(
        model_id="distilgpt2",
        output_dir=Path("artifacts/tn_balanced_with_healing"),
        bond_dimension=0.5,  # 50% reduction
        layer_policy=policy,
        heal_steps=3,  # Conservative healing
        heal_learning_rate=1e-4,
        calibration_batches=4,
    )
    
    report = compressor.compress_and_export()
    
    print(f"\n✓ Model: {report.model_id}")
    print(f"  Original: {report.original_parameters:,} parameters")
    print(f"  Compressed: {report.compressed_parameters:,} parameters")
    print(f"  Reduction: {(1 - report.parameter_ratio)*100:.1f}%")
    print(f"  Output: {report.output_dir}")
    
    # Validate
    results = compressor.validate_inference(prompt="The quick brown fox")
    print(f"\n✓ Inference works: Generated {results['output_length']} tokens")
    return report


def example_3_compare_bond_dimensions():
    """
    Example 3: Compare multiple bond dimensions
    
    Shows how compression ratio varies with bond dimension
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Bond Dimension Comparison")
    print("="*70)
    
    bond_dimensions = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = {}
    
    policy = CompressionPolicy(skip_module_patterns=())  # Compress all
    
    for bd in bond_dimensions:
        compressor = TensorNetworkModelCompressor(
            model_id="distilgpt2",
            output_dir=Path(f"artifacts/tn_bd_{bd}"),
            bond_dimension=bd,
            layer_policy=policy,
            heal_steps=0,
        )
        
        report = compressor.compress_and_export()
        reduction_pct = (1 - report.parameter_ratio) * 100
        results[bd] = {
            "parameters": report.compressed_parameters,
            "reduction_percent": reduction_pct,
        }
        
        print(f"\n  BD={bd}: {report.compressed_parameters:,} params ({reduction_pct:.1f}% reduction)")
        
        # Quick inference test
        try:
            compressor.validate_inference(prompt="Test")
            print(f"    ✓ Inference validated")
        except Exception as e:
            print(f"    ✗ Inference failed: {e}")
    
    # Save comparison
    comparison_path = Path("artifacts/tn_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Comparison saved to: {comparison_path}")
    return results


def example_4_different_models():
    """
    Example 4: Compress different models
    
    Shows how to apply TN compression to various models
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Compressing Different Models")
    print("="*70)
    
    models_to_test = [
        ("distilgpt2", 0.5),
        # ("gpt2", 0.5),  # Uncomment for larger model
    ]
    
    results = {}
    
    for model_id, bond_dim in models_to_test:
        print(f"\n  Compressing {model_id}...")
        
        compressor = TensorNetworkModelCompressor(
            model_id=model_id,
            output_dir=Path(f"artifacts/tn_{model_id}"),
            bond_dimension=bond_dim,
            heal_steps=0,
        )
        
        try:
            report = compressor.compress_and_export()
            results[model_id] = {
                "original_parameters": report.original_parameters,
                "compressed_parameters": report.compressed_parameters,
                "compression_ratio": report.parameter_ratio,
                "reduction_percent": (1 - report.parameter_ratio) * 100,
            }
            
            print(f"    ✓ {model_id}: {report.parameter_ratio:.4f} ratio")
            
            # Test inference
            compressor.validate_inference(prompt="AI is")
            print(f"    ✓ Inference works")
            
        except Exception as e:
            print(f"    ✗ Failed: {e}")
    
    # Save results
    results_path = Path("artifacts/tn_models_comparison.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")
    return results


def main():
    print("\n" + "="*70)
    print("Tensor Network Compression Examples")
    print("="*70)
    print("\nDemonstrating Tensor Network (TN) decomposition for LLM compression.")
    print("This implementation tensorizes self-attention and MLP layers")
    print("using a bond dimension to control correlation truncation.\n")
    
    try:
        # Run examples
        example_1_aggressive_compression()
        example_2_balanced_compression()
        example_3_compare_bond_dimensions()
        example_4_different_models()
        
        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70)
        print("\nCompressed models are ready for:")
        print("  1. Edge deployment (aggressive compression)")
        print("  2. Production use (balanced compression)")
        print("  3. Performance analysis (bond dimension studies)")
        print("\nNext steps:")
        print("  - Use 'llm-edge-compression chat' to test the compressed models")
        print("  - Review the manifests to understand compression metrics")
        print("  - Experiment with different bond dimensions for your use case")
        print()
        
    except KeyboardInterrupt:
        print("\n\nExamples cancelled by user.")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
