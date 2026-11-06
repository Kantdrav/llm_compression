# Research Areas and Implementation Roadmap

**Project:** LLM Edge Compression
**Prepared:** 2026-11-06
**Purpose:** Define the research directions in this repository, explain why
they matter, and provide an actionable path from the current prototype to a
credible research result.

## 1. What the project is trying to achieve

The project aims to make language models smaller and cheaper to run on
edge-class hardware without destroying useful language behavior. The central
research question is:

> What combination of structured tensor compression, pruning, quantization,
> and lightweight recovery training gives the best quality-memory-latency
> trade-off for a specific model and device?

This is a **rate-distortion and systems** problem, not only a file-size
problem. A successful result must report both compression and quality:

- model size and parameter count;
- peak memory;
- latency and throughput;
- next-token loss or perplexity;
- downstream task quality;
- generation stability;
- recovery-training cost.

## 2. Research areas in this codebase

### Area A — MPO/TT tensor-network compression

**What it is:** Replace eligible dense linear or Transformer projection
matrices with Matrix Product Operator cores. Sequential SVD truncation keeps
the most important directions and drops less important correlations.

**Why do it:** Transformer projections contain redundancy. MPO can reduce
parameters and storage while preserving more structure than unstructured
pruning. It is the main connection to the CompactifAI paper and the tensor
network references in `research_paper_reference/`.

**Current implementation:**

- `ResearchMPOLinear` uses an explicit maximum bond dimension.
- `MPOCompressor` handles regular `Linear` and GPT-2 `Conv1D` modules.
- `ResearchMPOCompressor` protects early layers and `lm_head` by default.
- `profile_mpo_sensitivity()` measures one-layer distortion at several bond
  dimensions.

**What to investigate next:**

1. Compare `bond_dim` values such as 8, 16, 32, and 64.
2. Compare two, three, and four MPO sites.
3. Measure reconstruction error before running generation.
4. Allocate bond dimensions per layer instead of using one global value.
5. Test whether MPO contraction is faster on the intended edge device.

**Success criterion:** Find a bond-dimension policy that reduces size while
staying within a predeclared quality and latency budget.

### Area B — Layer sensitivity and adaptive compression

**What it is:** Measure how much each Transformer layer contributes to model
quality, then compress sensitive and insensitive layers differently.

**Why do it:** The current `distilgpt2` experiment demonstrates the risk of a
global setting. `bond_dim=8` reduced size by about 34%, but next-token
accuracy fell from 23.91% to 8.70% on the small evaluation set. Different
layers do not tolerate the same approximation error.

**Current implementation:**

- `profile_mpo_sensitivity()` replaces one candidate layer at a time.
- It records output MSE, normalized output MSE, parameter ratio, and size
  ratio.
- `CompressionPolicy` supports skipped modules and rank overrides.

**How to do the research:**

1. Run the profiler with a representative calibration corpus.
2. Sort layers by normalized output MSE per saved parameter.
3. Protect high-sensitivity layers.
4. Assign larger bond dimensions to medium-sensitivity layers.
5. Assign smaller dimensions only to layers with low measured distortion.
6. Compare this policy with one global bond dimension at the same final size.

**Success criterion:** The sensitivity-aware policy has lower quality loss than
the global policy for the same serialized size.

### Area C — Quantization and hybrid compression

**What it is:** Quantize layers that are not tensorized, or combine
quantization with MPO compression.

**Why do it:** MPO and quantization reduce different costs. MPO changes the
parameter structure; quantization reduces bytes per parameter. A hybrid model
may achieve a better memory trade-off than either method alone.

**Current implementation:**

- Dynamic quantization is available through `DynamicQuantizationCompressor`.
- `MixedCompressionCompressor` applies MPO to eligible layers and quantizes
  protected linear layers.
- The manifest records the compression settings and size metrics.

**How to do the research:**

1. Establish quantization-only as a baseline.
2. Measure MPO-only at matched model sizes.
3. Measure MPO plus int8 quantization.
4. Use int4 only when the target runtime supports it reliably.
5. Measure actual memory and latency, not only state-dict size.

**Success criterion:** The hybrid method improves the quality-memory Pareto
frontier over quantization-only on the selected hardware.

### Area D — Healing and parameter-efficient recovery

**What it is:** Fine-tune the compressed model to recover behavior lost during
truncation, using either teacher-student calibration or PEFT/QLoRA.

**Why do it:** SVD truncation is a local approximation. Errors accumulate
through Transformer blocks. The CompactifAI and pruning references both show
that recovery training is an important part of the compression method.

**Current implementation:**

- `heal_model()` performs teacher-student output matching.
- The pipeline now snapshots the teacher before in-place compression.
- QLoRA healing support is available through the merged `main` work.

**Important current limitation:** The default pipeline calibration batches are
random token IDs. They are useful for smoke tests but are not representative
of natural language. They should not be used as evidence of final quality.

**How to do the research:**

1. Use real text from the same domain as deployment.
2. Split calibration and evaluation data.
3. Keep the original teacher frozen.
4. Train only MPO cores, adapters, or selected sensitive modules first.
5. Compare no healing, calibration healing, and QLoRA healing.
6. Record training time, trainable parameters, and quality recovery.

**Success criterion:** Healing recovers a measurable fraction of the
compression quality loss on held-out data without overfitting calibration
examples.

### Area E — Structured layer pruning

**What it is:** Remove complete Transformer blocks using representation
similarity, deepest-layer selection, or deterministic random selection.

**Why do it:** Removing whole layers can reduce compute and memory more
directly than weight-only compression. It also tests the research hypothesis
that some deep layers are redundant or can be healed after removal.

**Current implementation:**

- `LayerPruningCompressor` supports several selection strategies.
- Layer representations can be collected from calibration batches.
- The CLI exposes pruning configuration.

**How to do the research:**

1. Compare pruning ratios such as 10%, 20%, and 30%.
2. Compare similarity-guided and deepest-layer pruning.
3. Heal each pruned model with PEFT.
4. Measure quality and end-to-end speed.
5. Combine pruning with MPO only after each method has a baseline.

**Success criterion:** A pruned and healed model meets the application quality
threshold while reducing inference cost more than weight-only compression.

### Area F — Edge-runtime systems research

**What it is:** Determine whether compressed representations are actually
better on the device where the model will run.

**Why do it:** A smaller state dictionary can still be slower if MPO
contractions are not optimized. Compression decisions must be hardware-aware.

**How to do the research:**

1. Choose one target device: CPU laptop, Raspberry Pi, Jetson, or mobile.
2. Measure cold start, warm latency, throughput, and peak memory.
3. Test several sequence lengths and batch sizes.
4. Compare original, quantized, MPO, hybrid, and pruned bundles.
5. Record software versions and thread settings.

**Success criterion:** The selected compressed model meets explicit memory and
latency limits on the target device, not only on the development machine.

### Area G — Reproducible evaluation and responsible reporting

**What it is:** Make every compression claim auditable.

**Why do it:** Small prompt sets and file-size-only comparisons can make a
bad compression configuration appear successful. Reproducible reporting turns
the prototype into defensible research.

**Required records:**

- exact model ID and revision;
- tokenizer and context length;
- calibration and evaluation data hashes;
- compression method and all hyperparameters;
- skipped and overridden module paths;
- random seeds;
- hardware and software versions;
- serialized size and peak memory;
- quality and latency results;
- known limitations and model license.

**Success criterion:** Another person can rerun a result from the command,
manifest, data description, and environment information.

## 3. Recommended research sequence

### Stage 1 — Establish baselines

Run the original model and quantization-only model on a fixed evaluation set.
Do not begin with aggressive MPO. Save baseline accuracy, loss, latency,
memory, and size.

### Stage 2 — Characterize MPO

Run a bond-dimension sweep:

```text
bond_dim = 8, 16, 32, 64
mpo_sites = 3
```

For each run, collect:

- parameter ratio;
- serialized-size ratio;
- logit MSE;
- top-1 token agreement;
- next-token accuracy;
- perplexity or loss;
- generation samples.

### Stage 3 — Use sensitivity-aware policies

Run:

```bash
python -m llm_edge_compression.cli profile \
  --model-id distilgpt2 \
  --output artifacts/distilgpt2-sensitivity.json \
  --bond-dims 8,16,32 \
  --mpo-sites 3 \
  --calibration-batches 8 \
  --calibration-batch-size 1 \
  --calibration-sequence-length 64
```

Use the results to update `CompressionPolicy` rather than guessing which
layers to skip.

### Stage 4 — Add representative healing

Replace random calibration IDs with real tokenized text. First try a short
teacher-student healing run; then compare against QLoRA healing if available.
Evaluate on held-out text, not the healing examples.

### Stage 5 — Compare hybrid and pruning methods

Only after individual baselines are understood, compare:

1. original;
2. int8 quantization;
3. MPO;
4. MPO plus quantization;
5. layer pruning;
6. layer pruning plus healing;
7. combined pruning, MPO, quantization, and healing.

### Stage 6 — Benchmark the target edge device

Select the final configuration using the measured Pareto frontier. Do not
select it from parameter count alone.

## 4. Why these areas are worth pursuing

These areas are worthwhile because they answer different failure modes:

- **MPO** answers whether structured redundancy can reduce storage.
- **Sensitivity profiling** answers where compression is safe.
- **Quantization** answers how to reduce bytes per retained parameter.
- **Healing** answers whether lost behavior can be recovered cheaply.
- **Layer pruning** answers whether whole computation blocks are redundant.
- **Runtime benchmarking** answers whether compression helps users in practice.
- **Reproducibility** answers whether the result is scientifically credible.

Together they form a complete research program rather than a single
compression trick.

## 5. Proposed thesis or paper contributions

A strong project contribution could be framed as:

> A sensitivity-aware hybrid compression pipeline for edge LLM deployment that
> allocates MPO bond dimensions per Transformer layer, combines structured
> tensor compression with quantization, and uses parameter-efficient healing to
> recover quality under a measured device memory and latency budget.

The contribution should be supported by:

1. an open implementation;
2. a reproducible sensitivity profiler;
3. controlled ablations;
4. quality-size-latency tables;
5. real-device measurements;
6. an honest limitations section.

## 6. Practical definition of success

Set thresholds before the final experiment. A reasonable initial target is:

- at least 30% lower serialized size;
- at least 25% lower peak inference memory;
- less than 1 percentage point next-token accuracy loss after healing;
- no unacceptable downstream-task regression;
- no warm-latency increase on the selected edge device;
- a reproducible bundle load from a clean environment.

These are engineering targets, not guaranteed outcomes. If a configuration
misses them, report the miss and use it to refine the next experiment.

## 7. Immediate next actions

1. Expand `evaluation.jsonl` to at least 100 representative examples.
2. Run the sensitivity profiler with real tokenized calibration text.
3. Add logit MSE, top-1 agreement, and perplexity to the evaluator.
4. Generate a bond-dimension sweep for `distilgpt2`.
5. Build a sensitivity-aware `CompressionPolicy`.
6. Compare representative-data healing against no healing.
7. Run the final candidates on the intended edge device.
8. Store each result with its manifest, command, metrics, and limitations.

## Final recommendation

Do not optimize for the largest compression percentage first. The current
experiment already shows why: `bond_dim=8` reduced model size but produced a
large quality drop. The next research milestone should be a reliable
compression-quality curve, followed by sensitivity-aware rank allocation and
representative-data healing. Once those are working, edge-device benchmarking
will show whether the method is useful beyond the storage metric.
