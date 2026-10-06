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

# Complete Project Guide

## 1. Project purpose

This repository is a research and engineering scaffold for reducing the size and computational cost of decoder-only Large Language Models while preserving as much model quality as possible. It combines conventional compression baselines with structured Transformer layer pruning inspired by the ICLR 2025 work, The Unreasonable Ineffectiveness of the Deeper Layers, and optional QLoRA/PEFT healing.

The recommended research flow is:

1. Load a pretrained causal language model.
2. Establish a baseline.
3. Compress or prune the model.
4. Measure parameter and storage reduction.
5. Measure perplexity and downstream quality.
6. Optionally heal a pruned model with QLoRA.
7. Re-run evaluation.
8. Benchmark the final model on target hardware.

## 2. Repository structure

| Path | Purpose |
|---|---|
| src/llm_edge_compression/layer_pruner.py | Structured Transformer layer pruning |
| src/llm_edge_compression/qlora_healing.py | 4-bit QLoRA/PEFT healing |
| src/llm_edge_compression/healing.py | Lightweight teacher/student healing |
| src/llm_edge_compression/compressors.py | Quantization and low-rank compression |
| src/llm_edge_compression/adaptive_mpo.py | Adaptive MPO-style compression |
| src/llm_edge_compression/paper_mpo.py | Research-oriented MPO baseline |
| src/llm_edge_compression/pipeline.py | Main compression workflow |
| src/llm_edge_compression/export.py | Bundle export |
| src/llm_edge_compression/inference.py | Bundle loading and generation |
| src/llm_edge_compression/config.py | Experiment configuration |
| src/llm_edge_compression/cli.py | Command-line interface |
| src/llm_edge_compression/manifest.py | Experiment metadata |
| tests/ | Automated tests |

## 3. Installation

Requirements: Python 3.10+, PyTorch, Transformers, and the dependencies in pyproject.toml. CUDA is required for QLoRA but is not required for every compression experiment.

Clone the repository:

    git clone https://github.com/Kantdrav/llm_compression.git
    cd llm_compression

Create and activate a virtual environment on Linux/macOS:

    python3 -m venv .venv
    source .venv/bin/activate

On Windows PowerShell:

    python -m venv .venv
    .\\.venv\\Scripts\\Activate.ps1

Install the normal project:

    pip install -e .

Install QLoRA support when needed:

    pip install -e '.[healing]'

The optional healing extra installs accelerate, bitsandbytes, datasets, and peft.

## 4. Verify the installation

Check the command line interface:

    llm-edge-compression --help

Run the local demo before downloading a large model:

    llm-edge-compression demo --output-dir artifacts/demo

Run tests:

    pytest

## 5. Compression methods

The CLI currently exposes these compression methods:

- quantize: dynamic quantization baseline.
- tensor_inspired: low-rank/tensor-inspired baseline.
- mpo: MPO-style compression with fixed or adaptive rank selection.
- paper_mpo: research-oriented MPO baseline.
- layer_prune: structured Transformer block pruning.

Example tensor-inspired compression:

    llm-edge-compression compress --model-id gpt2 --output-dir artifacts/gpt2-tensor --method tensor_inspired --rank-ratio 0.5

Example quantization:

    llm-edge-compression compress --model-id gpt2 --output-dir artifacts/gpt2-quantized --method quantize

## 6. Layer pruning research path

The layer-pruning implementation is the main connection to the Gromov et al. paper. Instead of deleting arbitrary individual layers, it searches for a contiguous block of Transformer blocks that appears redundant.

For a candidate block width n, the implementation compares hidden representations separated by n layers and uses angular distance based on cosine similarity. The lowest-distance candidate is selected for the similarity strategy. The final Transformer block is protected.

Available strategies:

- similarity: representation-guided contiguous block selection.
- deepest: remove the deepest eligible blocks as a simple baseline.
- random: reproducible random contiguous block selection.

Run 30% similarity-guided pruning:

    llm-edge-compression compress --model-id distilgpt2 --output-dir artifacts/distilgpt2-pruned --method layer_prune --layer-pruning-ratio 0.30 --layer-pruning-strategy similarity

Run exact-count pruning:

    llm-edge-compression compress --model-id distilgpt2 --output-dir artifacts/distilgpt2-pruned-2 --method layer_prune --layer-pruning-num-layers 2 --layer-pruning-strategy similarity

Deepest baseline:

    llm-edge-compression compress --model-id distilgpt2 --output-dir artifacts/distilgpt2-deepest --method layer_prune --layer-pruning-ratio 0.30 --layer-pruning-strategy deepest

Random baseline:

    llm-edge-compression compress --model-id distilgpt2 --output-dir artifacts/distilgpt2-random --method layer_prune --layer-pruning-ratio 0.30 --layer-pruning-strategy random --layer-pruning-seed 42

## 7. QLoRA healing

Pruning changes the computation path and can introduce a representation mismatch. The optional QLoRA stage provides parameter-efficient healing using a 4-bit NF4 base model and LoRA adapters.

Install the optional dependencies:

    pip install -e '.[healing]'

Check CUDA:

    python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

Run a small validation experiment:

    llm-edge-compression heal-qlora --model-id distilgpt2 --output-dir artifacts/distilgpt2-healed --layer-pruning-ratio 0.30 --layer-pruning-strategy similarity --dataset allenai/c4 --dataset-config en --dataset-split train --max-samples 256 --max-seq-length 512 --max-steps 500

The command loads the original model in 4-bit mode, reconstructs the requested pruned architecture, finds suitable LoRA target modules, prepares the model for k-bit training, streams the selected dataset, and saves a PEFT adapter.

Useful QLoRA controls include:

- --lora-r
- --lora-alpha
- --lora-dropout
- --learning-rate
- --max-steps
- --max-samples
- --max-seq-length
- --batch-size
- --gradient-accumulation-steps
- --compute-dtype

The default compute dtype is bfloat16 when supported, with float16 fallback. QLoRA requires a CUDA-capable environment and is intentionally optional.

## 8. Why the adapter is separate

QLoRA normally keeps the base model quantized and trains a small LoRA adapter. This project therefore saves the adapter separately instead of automatically merging it into the 4-bit base weights. This preserves the memory-saving property of QLoRA and makes the healing artifact much smaller than a full model checkpoint.

## 9. Output artifacts

A normal compression run creates an output directory such as:

    artifacts/distilgpt2-pruned/
        compressed_model.pt
        manifest.json

manifest.json records the model ID, compression method, configuration, parameter counts, storage metrics, and layer-pruning information. For layer pruning it also records the selected layer indices and angular-distance result.

A QLoRA run creates an adapter directory containing the PEFT adapter and tokenizer/configuration artifacts.

## 10. Inspect and run inference

Inspect a compression result:

    llm-edge-compression inspect --output-dir artifacts/distilgpt2-pruned

Run interactive inference:

    llm-edge-compression chat --bundle-dir artifacts/distilgpt2-pruned

The bundle loader reconstructs the compressed architecture from the stored manifest configuration and loads the compressed weights.

## 11. Recommended evaluation

Do not select a compression method using model size alone. Compare the original model, the pruned model, and the healed model using the same evaluation data.

At minimum measure:

1. Parameter ratio = compressed parameters / original parameters.
2. Storage ratio = compressed storage / original storage.
3. Size reduction percentage.
4. Validation next-token loss.
5. Perplexity.
6. Downstream task or reasoning performance.
7. Actual inference latency and memory usage on target hardware.

For each pruning ratio, compare at least similarity, deepest, and random strategies. Then apply QLoRA to the promising configurations.

A useful experiment table is:

    baseline
    10% pruning + similarity
    10% pruning + deepest
    10% pruning + random
    20% pruning + similarity
    20% pruning + deepest
    20% pruning + random
    30% pruning + similarity
    30% pruning + deepest
    30% pruning + random
    best pruning + QLoRA

Keep the model checkpoint, tokenizer, evaluation data, sequence length, training budget, and seeds fixed when comparing methods.

## 12. Calibration data

The current lightweight compression pipeline uses synthetic calibration inputs for the structural pruning calculation. This keeps the basic pipeline easy to run, but serious research should use representative real text from the intended workload.

For example, use conversational text for a chat model, code for a coding model, or domain-specific text for a specialized model. Calibration data should be separate from the final evaluation set.

## 13. Research limitations

This repository is a practical research scaffold, not a claim of reproducing every experimental detail of the ICLR 2025 paper. The similarity-guided pruning captures the core representation-redundancy idea, while the calibration and evaluation infrastructure is intentionally lightweight.

Important limitations:

- Full standardized perplexity and downstream benchmark automation is not yet included.
- QLoRA requires CUDA and the optional healing dependencies.
- The LoRA adapter is stored separately rather than automatically merged.
- Compression quality depends on the architecture and pruning ratio.
- Parameter reduction does not automatically imply lower real-world latency; target-device benchmarks are necessary.

## 14. Troubleshooting

If QLoRA reports that CUDA is unavailable, verify torch.cuda.is_available() and use a CUDA-enabled PyTorch environment.

If peft, datasets, bitsandbytes, or accelerate are missing, install the healing extra:

    pip install -e '.[healing]'

If QLoRA runs out of GPU memory, first reduce sequence length and batch size, then increase gradient accumulation to maintain an effective batch size.

If a custom Transformer architecture is not recognized by layer pruning, add its layer container pattern to layer_pruner.py.

If similarity pruning performs poorly, compare it against deepest and random at the same pruning ratio and evaluate perplexity/task performance rather than relying on angular distance alone.

## 15. Development and roadmap

The next useful research/engineering additions are a standardized perplexity evaluator, automated experiment sweeps, real-text calibration, pruning visualizations, direct adapter loading in the inference CLI, adapter/base-model packaging, ONNX/GGUF/TensorRT export, and benchmarks on Jetson, Raspberry Pi, and mobile-class hardware.

## 16. End-to-end recommended workflow

    1. git clone the repository.
    2. Create and activate a Python virtual environment.
    3. pip install -e .
    4. Run the demo.
    5. Run a baseline compression experiment.
    6. Run similarity-guided layer pruning.
    7. Run deepest and random controls.
    8. Inspect the manifests and compression metrics.
    9. Evaluate perplexity and downstream quality.
    10. Install the healing extra on a CUDA machine.
    11. Run QLoRA healing on promising pruning levels.
    12. Re-run evaluation.
    13. Benchmark memory and latency on the target edge device.
    14. Record the experiment configuration and results for reproducibility.

The core research question is not simply how much smaller the model becomes. It is how much useful model quality can be retained for each unit of compression.
