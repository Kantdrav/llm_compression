# Tensor Network Model Compression Guide

This guide demonstrates how to compress LLMs using Tensor Network (TN) decomposition as an alternative to the original GPT-2 compression approach.

## What is Tensor Network Compression?

Tensor Network compression involves:

1. **Tensorization**: Self-attention (SA) and multi-layer perceptron (MLP) layers are converted to tensor network form
2. **Decomposition**: Uses specific TN structures (e.g., low-rank, tree, or matrix product operator forms)
3. **Truncation**: The bond dimension controls correlation truncation - effectively reducing model parameters
4. **Accuracy Preservation**: Knowledge distillation healing can restore accuracy after compression

## Quick Start

### 1. Basic Compression (DistilGPT2)

The simplest example - compress a small model:

```bash
cd /home/kantdravi/Desktop/quantum_compression
python examples/compress_with_tn.py \
  --model-id distilgpt2 \
  --output-dir artifacts/tn_distilgpt2 \
  --bond-dimension 0.5 \
  --validate
```

### 2. Compress Different Models

#### GPT-2 (Medium)
```bash
python examples/compress_with_tn.py \
  --model-id gpt2 \
  --output-dir artifacts/tn_gpt2 \
  --bond-dimension 0.6 \
  --heal-steps 5 \
  --validate
```

#### Phi-2 (Larger but manageable)
```bash
python examples/compress_with_tn.py \
  --model-id microsoft/phi-2 \
  --output-dir artifacts/tn_phi2 \
  --bond-dimension 0.5 \
  --heal-steps 10 \
  --validate
```

#### OPT-125M (Facebook's model)
```bash
python examples/compress_with_tn.py \
  --model-id facebook/opt-125m \
  --output-dir artifacts/tn_opt125m \
  --bond-dimension 0.4 \
  --heal-steps 8 \
  --validate
```

## Parameters Explained

### `--model-id`
The HuggingFace model identifier. Examples:
- `distilgpt2` - Lightweight, fast
- `gpt2` - Standard GPT-2 medium
- `microsoft/phi-2` - Efficient instruction-tuned model
- `facebook/opt-125m` - Meta's OPT series
- Any HuggingFace model supporting causal language modeling

### `--bond-dimension`
Controls TN decomposition aggressiveness:
- **0.3-0.4**: Aggressive compression (60-70% parameter reduction) - may lose accuracy
- **0.5**: Balanced (50% parameter reduction) - good default
- **0.6-0.7**: Conservative compression (30-40% reduction) - better accuracy
- **0.9-1.0**: Minimal compression - mostly validation

### `--heal-steps`
Knowledge distillation steps after compression:
- `0`: No healing (fast)
- `5-10`: Moderate healing (recommended for important models)
- `20+`: Extensive healing (slower but best accuracy recovery)

### `--skip-first-n-layers`
Skip compression on first N transformer layers (preserves early features):
- `0`: Compress all layers
- `1`: Skip first layer (default, recommended)
- `2`: Skip first 2 layers

### `--no-compress-lm-head`
Prevents compression of the language model head (output projection layer)

## Example Workflow

```bash
# Step 1: Quick validation
python examples/compress_with_tn.py \
  --model-id distilgpt2 \
  --output-dir artifacts/test \
  --bond-dimension 0.6 \
  --validate

# Step 2: Aggressive compression for edge deployment
python examples/compress_with_tn.py \
  --model-id distilgpt2 \
  --output-dir artifacts/edge_deployment \
  --bond-dimension 0.4 \
  --heal-steps 10 \
  --skip-first-n-layers 1 \
  --validate

# Step 3: Compare different bond dimensions
for BD in 0.3 0.4 0.5 0.6 0.7; do
  python examples/compress_with_tn.py \
    --model-id distilgpt2 \
    --output-dir artifacts/tn_bd_$BD \
    --bond-dimension $BD \
    --validate
done
```

## Output Files

After compression, you'll find:

```
artifacts/tn_distilgpt2/
├── manifest.json                 # Metadata and compression config
├── compressed_model.pt           # Compressed model weights
├── compression_summary.json       # Summary metrics
└── compressed_model.onnx         # Optional ONNX export
```

### manifest.json Structure

```json
{
  "model_id": "distilgpt2",
  "method": "tensor_network",
  "compression_config": {
    "model_id": "distilgpt2",
    "method": "tensor_inspired",
    "rank_ratio": 0.5,
    "heal_steps": 0
  },
  "metrics": {
    "original_parameters": 82113536,
    "compressed_parameters": 41056768,
    "parameter_ratio": 0.500,
    "bond_dimension": 0.5,
    "total_linear_layers": 24
  }
}
```

## Inference Validation

The script automatically validates that the compressed model works:

```
Validating inference with compressed model...

INFERENCE VALIDATION RESULTS
----------------------------------------------
Prompt: 'The future of artificial intelligence'

Generated Output:
  The future of artificial intelligence is uncertain and exciting. The
  possibilities are endless, and the potential for innovation is vast.
  Many experts believe that AI will continue to advance...

Input Tokens: 8
Output Tokens: 35
Generated Tokens: 27
```

## Using Compressed Models

### Via CLI

```bash
llm-edge-compression chat \
  --bundle-dir artifacts/tn_distilgpt2 \
  --max-new-tokens 64 \
  --temperature 0.7
```

### Programmatically

```python
from llm_edge_compression.tn_model_compressor import TensorNetworkModelCompressor
from pathlib import Path

compressor = TensorNetworkModelCompressor(
    model_id="distilgpt2",
    output_dir=Path("artifacts/tn_model"),
    bond_dimension=0.5,
    heal_steps=5
)

report = compressor.compress_and_export()
print(f"Compression ratio: {report.parameter_ratio:.4f}")

# Validate inference
results = compressor.validate_inference(prompt="Hello")
print(results["generated_text"])
```

## Testing

Run the comprehensive test suite:

```bash
cd /home/kantdravi/Desktop/quantum_compression
python -m pytest tests/test_tn_model_compressor.py -v
```

## Compression Performance Examples

### DistilGPT2
- Original: 82M parameters
- BD=0.5: 41M parameters (50% reduction)
- BD=0.4: 33M parameters (60% reduction)
- BD=0.3: 25M parameters (70% reduction)

### GPT-2 (Medium)
- Original: 124M parameters
- BD=0.5: 62M parameters (50% reduction)
- BD=0.4: 50M parameters (60% reduction)

## Troubleshooting

### Out of Memory
Use a smaller model or reduce `calibration_batch_size`:
```bash
# Reduce batch size for healing
python examples/compress_with_tn.py \
  --model-id gpt2 \
  --output-dir artifacts/tn_gpt2 \
  --heal-steps 2 \
  --validate
```

### Generation Quality
Increase `heal_steps` for better accuracy:
```bash
python examples/compress_with_tn.py \
  --model-id distilgpt2 \
  --output-dir artifacts/tn_high_quality \
  --bond-dimension 0.6 \
  --heal-steps 20 \
  --validate
```

### Slow Compression
Disable healing or use smaller models:
```bash
python examples/compress_with_tn.py \
  --model-id distilgpt2 \
  --output-dir artifacts/tn_fast \
  --heal-steps 0 \
  --validate
```

## Key Differences from GPT-2 Compression

| Aspect | Original (GPT-2) | New (Tensor Networks) |
|--------|------------------|----------------------|
| **Target** | GPT-2 only | Any LLM |
| **Method** | Low-rank decomposition | TN with bond dimension control |
| **Flexibility** | Fixed compression | Tunable via bond_dimension |
| **Early Layers** | All compressed | Can skip first N layers |
| **Healing** | Optional | Integrated knowledge distillation |
| **Inference** | Demo only | Full validation included |

## Advanced Configuration

### Custom Layer Policies

```python
from llm_edge_compression.tn_model_compressor import TensorNetworkModelCompressor
from llm_edge_compression.config import CompressionPolicy
from pathlib import Path

# Only compress specific layers
policy = CompressionPolicy(
    skip_module_patterns=(
        r"^model\.layers\.[01](?:\.|$)",  # Skip first 2 layers
        r"(^|\.)lm_head(?:\.|$)",          # Skip output head
        r"(^|\.)embedding(?:\.|$)",        # Skip embeddings
    )
)

compressor = TensorNetworkModelCompressor(
    model_id="gpt2",
    output_dir=Path("artifacts/custom"),
    bond_dimension=0.5,
    layer_policy=policy
)

report = compressor.compress_and_export()
```

## References

The implementation is based on:
- Tensor network decomposition for neural networks
- Matrix product operator (MPO) structures
- Knowledge distillation for model healing
- Bond dimension as a compression control parameter

## Next Steps

1. **Experiment with bond dimensions** - Find the sweet spot between compression and accuracy
2. **Compare models** - Compress different models and compare their compression ratios
3. **Optimize for deployment** - Use aggressive compression (0.3-0.4) for edge devices
4. **Integrate into pipeline** - Use the compressed models in your applications

For more details, see the main README.md and CI_CD.md for integration examples.
