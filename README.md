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
- Interactive querying of a compressed bundle from the CLI
- FastAPI endpoints for uploading a local model archive or downloading a Hugging Face model before compression
- Flutter web UI for switching between local upload and internet model download flows

## What the compression report means

The current pipeline does not compute task accuracy yet. It reports compression size metrics instead:

- `original_parameters`
- `compressed_parameters`
- `parameter_ratio` = `compressed_parameters / original_parameters`

Use `parameter_ratio` as the quick health check for how much smaller the model became. If you want true accuracy, add an evaluation set after compression, such as perplexity on a validation corpus or a downstream benchmark.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
llm-edge-compression compress --model-id gpt2 --output-dir artifacts/gpt2-low-rank --method tensor_inspired
llm-edge-compression demo --output-dir artifacts/demo
llm-edge-compression chat --bundle-dir artifacts/gpt2-low-rank
```

To compress a remote Hugging Face model without uploading a ZIP, call the backend `POST /compress-remote` endpoint or use the Flutter web app's "Internet model" mode.

If you deploy the Flutter web client separately, set `API_BASE_URL` at build time so the app points at your live backend.

The `demo` command uses a tiny local model, so it is the easiest way to verify the bundle/export flow before you point the pipeline at a full LLM checkpoint.

To ask queries of a compressed model, point `chat` at the bundle directory that contains `manifest.json` and `compressed_model.pt`. The CLI reloads the original Hugging Face model, rebuilds the compressed layers using the stored compression settings, loads the compressed weights, and then opens an interactive prompt.

Example session:

```bash
llm-edge-compression chat --bundle-dir artifacts/gpt2-low-rank
```

Then type a prompt such as:

```text
Explain quantum compression in one paragraph.
```

## Suggested next milestones

- Add a real MPO/TN decomposition for linear layers
- Add evaluation on perplexity and downstream tasks
- Add device-specific export targets such as ONNX Runtime, GGUF, or TensorRT-LLM
- Add benchmarking scripts for Raspberry Pi, Jetson, and mobile-class CPU targets

## Deployment

The repository includes GitHub Actions workflows for two deployment targets:

- `Deploy Backend to Render` triggers a Render deploy hook for the FastAPI server.
- `Deploy Flutter Web` builds `flutter_app` and deploys the web bundle to Vercel.

Required secrets:

- `RENDER_DEPLOY_HOOK_URL`
- `VERCEL_TOKEN`
- `API_BASE_URL`


## Similarity-guided layer pruning

The layer-pruning implementation follows the representation-similarity idea from Gromov et al., *The Unreasonable Ineffectiveness of the Deeper Layers* (ICLR 2025). It collects the final-token hidden representation entering each Transformer block on a small calibration set, computes angular distance between representations separated by the proposed pruning width, and removes the contiguous block with the smallest distance.

Supported strategies:

- `similarity`: choose the most redundant contiguous block from calibration representations.
- `deepest`: remove the deepest eligible blocks; this is a simple baseline.
- `random`: choose a reproducible random contiguous block.

The final Transformer block is protected. The selected layer indices are persisted in `manifest.json`, allowing a compressed bundle to reconstruct the same pruned architecture during inference.

Example:

```bash
llm-edge-compression compress \
  --model-id distilgpt2 \
  --output-dir artifacts/distilgpt2-pruned \
  --method layer_prune \
  --layer-pruning-ratio 0.30 \
  --layer-pruning-strategy similarity
```

For a controlled comparison, run the same ratio with `--layer-pruning-strategy deepest`. Do not interpret QA accuracy alone as sufficient validation; compare perplexity and reasoning-sensitive evaluations as well.

## Optional QLoRA healing

After structured layer pruning, the project can optionally run a QLoRA healing stage. This follows the paper's idea of repairing the representation mismatch with parameter-efficient fine-tuning while keeping the base model in 4-bit NF4 quantization.

QLoRA is intentionally an optional dependency because it requires a CUDA GPU and additional packages:

    pip install -e '.[healing]'

A typical experiment is:

    llm-edge-compression heal-qlora --model-id distilgpt2 --output-dir artifacts/distilgpt2-pruned-healing --layer-pruning-ratio 0.30 --layer-pruning-strategy similarity --dataset allenai/c4 --max-samples 256 --max-seq-length 512 --max-steps 500

The command saves a PEFT LoRA adapter rather than merging the adapter into the 4-bit base weights. This keeps the compressed base model quantized and makes the healing artifact small. For deployment, load the same pruned base architecture and attach the saved adapter with PEFT.

Useful controls include `--lora-r`, `--lora-alpha`, `--lora-dropout`, `--learning-rate`, `--gradient-accumulation-steps`, and `--compute-dtype`. `bfloat16` is preferred when the GPU supports it; otherwise the implementation falls back to `float16`.

The existing `heal_steps` path is still available for small/local experiments. QLoRA healing is separate because it has different runtime and dependency requirements and is intended for the paper-style GPU experiment.