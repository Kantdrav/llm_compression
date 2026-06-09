# LLM Edge Compression

Research scaffold for compressing large language models and packaging the result for edge deployment.

This project is designed around three practical stages:

1. Load a transformer model from Hugging Face or a local checkpoint.
2. Compress it with one of the supported strategies:
   - dynamic quantization
   - low-rank factorization
   - tensor-inspired compression baseline
3. Export a self-contained edge bundle with a manifest, compressed weights, and deployment metadata.

## Why this shape

The paper you shared is about a tensor-network style compression method. Reproducing that exactly is a research project on its own, so this scaffold gives you a working starting point that can evolve toward the paper's method while already being useful for edge experiments.

## Current capabilities

- CLI for running compression jobs
- Compression config and manifest tracking
- Model export bundle generation
- A low-rank compression path that is a good baseline for tensor-network experiments

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
llm-edge-compression compress --model-id gpt2 --output-dir artifacts/gpt2-low-rank --method tensor_inspired
llm-edge-compression demo --output-dir artifacts/demo
```

The `demo` command uses a tiny local model, so it is the easiest way to verify the bundle/export flow before you point the pipeline at a full LLM checkpoint.

## Suggested next milestones

- Add a real MPO/TN decomposition for linear layers
- Add evaluation on perplexity and downstream tasks
- Add device-specific export targets such as ONNX Runtime, GGUF, or TensorRT-LLM
- Add benchmarking scripts for Raspberry Pi, Jetson, and mobile-class CPU targets
