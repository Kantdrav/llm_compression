# Research Paper Analysis Report

**Collection:** `research_paper_reference/`  
**Analysis date:** 2026-11-06  
**Scope:** 14 PDFs supplied with this repository, plus their relevance to the
LLM edge-compression implementation.

## Executive summary

The collection presents a coherent path from the Transformer architecture to
practical model compression:

1. **Transformers and pretrained language models** establish the architecture
   and model families being compressed (`13.pdf`, `12.pdf`, `5.pdf`, `10.pdf`).
2. **Unstructured pruning and low-rank decomposition** show that substantial
   redundancy can be removed while retaining task quality (`3.pdf`, `2.pdf`).
3. **Tensor-network representations** replace large dense weight matrices with
   contracted low-order cores, providing a structured compression mechanism
   (`4.pdf`, `1.pdf`, `14.pdf`).
4. **Tensor neural networks and variational training** extend the approach to
   trainable, hybrid tensor/dense models and nontraditional workloads
   (`6.pdf`, `7.pdf`/`8.pdf`, `9.pdf`).
5. **Layer pruning with lightweight healing** demonstrates that compression
   should be evaluated jointly with selective recovery fine-tuning (`11.pdf`).

For this repository, the strongest practical conclusion is that **MPO/TT
compression should be treated as a configurable, layer-aware approximation,
not as a universal replacement for every linear layer**. The existing
`ResearchMPOCompressor` is aligned with this conclusion: it uses an explicit
bond dimension, protects sensitive modules through a layer policy, and applies
TT-SVD to eligible linear and GPT-2 `Conv1D` layers. Quality and latency must
be measured after compression; parameter count alone is insufficient.

## Collection inventory

| File | Pages | Paper | Main contribution | Relevance |
|---|---:|---|---|---|
| `1.pdf` | 42 | Roman Orús, *A Practical Introduction to Tensor Networks: Matrix Product States and Projected Entangled Pair States* (2014) | Practical MPS/PEPS concepts, contractions, and numerical methods | Provides the tensor-network foundations needed to reason about cores, bonds, and approximation error |
| `2.pdf` | 12 | Max Jaderberg, Andrea Vedaldi, Andrew Zisserman, *Speeding up Convolutional Neural Networks with Low Rank Expansions* (2014) | Low-rank spatial/channel expansions for faster CNN inference | Supports factorization as a tunable speed/accuracy trade-off |
| `3.pdf` | 9 | Song Han et al., *Learning both Weights and Connections for Efficient Neural Networks* (2015) | Train, prune, and retrain to remove unimportant connections | Motivates sparse or structured pruning and post-compression recovery |
| `4.pdf` | 9 | Alexander Novikov et al., *Tensorizing Neural Networks* (2015) | Tensor-Train factorization of dense layers | Direct precedent for replacing dense matrices with tensorized factors |
| `5.pdf` | 16 | Jacob Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2018) | Bidirectional pretraining and transfer to downstream tasks | Shows why protected layers and task-level regression tests matter |
| `6.pdf` | 14 | Raj G. Patel et al., *Quantum-Inspired Tensor Neural Networks for Partial Differential Equations* (2022) | Tensor neural networks with parameter savings and faster training for PDEs | Demonstrates tensorized models beyond standard language tasks |
| `7.pdf` | 21 | Saeed S. Jahromi and Roman Orús, *Variational Tensor Neural Networks for Deep Learning* (2024 version) | DMRG-inspired local/variational optimization for tensor neural networks | Suggests fine-tuning tensor cores instead of relying only on one-shot decomposition |
| `8.pdf` | 21 | Saeed S. Jahromi and Roman Orús, *Variational Tensor Neural Networks for Deep Learning* (duplicate of `7.pdf`) | Same paper and version as `7.pdf` | Duplicate; should not be counted as independent evidence |
| `9.pdf` | 25 | Maolin Wang et al., *Tensor Networks Meet Neural Networks: A Survey and Future Perspectives* (2025 version) | Survey of tensorial neural networks across CNNs, RNNs, GNNs, Transformers, and LLMs | Provides the broad taxonomy and identifies open research directions |
| `10.pdf` | 27 | Hugo Touvron et al., *LLaMA: Open and Efficient Foundation Language Models* (2023) | Efficient open foundation models with strong performance at smaller scale | Establishes the LLM deployment context and the value of inference-efficient models |
| `11.pdf` | 24 | Andrey Gromov et al., *The Unreasonable Ineffectiveness of the Deeper Layers* (ICLR 2025) | Layer pruning followed by small QLoRA healing | Directly supports layer sensitivity analysis and recovery fine-tuning |
| `12.pdf` | 24 | Alec Radford et al., *Language Models are Unsupervised Multitask Learners* (2019) | GPT-2 and zero-shot transfer from causal language modeling | Explains the GPT-2/causal-LM family used by the repository examples |
| `13.pdf` | 11 | Ashish Vaswani et al., *Attention Is All You Need* (2017) | Transformer architecture based on self-attention | Identifies the projection layers and attention structure targeted by compression |
| `14.pdf` | 13 | Román Orús, *Tensor Networks for Complex Quantum Systems* (2014) | Review of tensor-network structures, algorithms, and entanglement | Supplies the physics intuition behind bond dimension and representational capacity |

### Duplicate and provenance notes

- `7.pdf` and `8.pdf` have the same title, authors, abstract, arXiv version,
  and page count. They are treated as one source in the synthesis.
- PDF filenames are local numeric labels rather than stable citation keys.
  Any publication, citation, or experiment should use the paper title and
  authors above, not only the filename.
- The report summarizes the supplied documents. It does not independently
  validate every benchmark, implementation detail, license, or publication
  status stated in them.

## Paper-by-paper analysis

### 1. Transformer and language-model foundations

#### *Attention Is All You Need* (`13.pdf`)

The Transformer removes recurrence and convolution from sequence transduction
and uses self-attention plus feed-forward projections. Its key engineering
benefits are parallel training and strong quality at lower training time than
the recurrent baselines considered in the paper. For compression, the
important observation is that much of the parameter volume is concentrated in
repeated projection and feed-forward matrices. Those repeated matrices have
regular shapes and are therefore suitable candidates for low-rank or
tensor-train/MPO approximations.

#### *Language Models are Unsupervised Multitask Learners* (`12.pdf`)

GPT-2 demonstrates that a large causal Transformer trained on WebText can
transfer to multiple tasks without task-specific supervised training. This
raises the cost of a compression error: a change that appears harmless on one
prompt can affect many behaviors. Evaluation should therefore include
next-token quality and representative prompts, not only a single downstream
metric.

#### *BERT* (`5.pdf`)

BERT shows the value of bidirectional pretraining and task adaptation. Although
its objective differs from GPT-2, it reinforces a general deployment lesson:
compressed representations must preserve the information used by later task
heads. Layer policies should be configurable rather than hard-coded, and
compressed checkpoints should be tested on the intended task distribution.

#### *LLaMA* (`10.pdf`)

LLaMA emphasizes inference-aware scaling: a smaller model trained on more data
can be preferable when serving cost matters. This is directly relevant to edge
deployment. Compression should be assessed with a combined objective covering
model size, memory use, latency, throughput, and quality. A lower parameter
count is not automatically a faster model if tensor contractions or
dequantization add overhead on the target device.

### 2. Pruning and low-rank compression

#### *Learning both Weights and Connections for Efficient Neural Networks* (`3.pdf`)

The paper uses a three-stage process: learn weights, prune low-importance
connections, and retrain the remaining network. Its central lesson is that
compression and recovery are coupled. The reported reductions for AlexNet and
VGG-16 show the potential of removing redundancy, but the result depends on
the pruning pattern and the retraining stage. For this project, the analogous
workflow is: compress, evaluate on held-out text, and optionally heal with
parameter-efficient fine-tuning.

#### *Speeding up Convolutional Neural Networks with Low Rank Expansions* (`2.pdf`)

This work exploits redundancy in filters and channels to replace expensive
convolutions with lower-rank operations. It explicitly treats speedup as
tunable: more aggressive factorization can increase speed while causing a
controlled accuracy drop. The same principle applies to linear projections in
LLMs, but the measured benefit depends on backend kernels and hardware.

### 3. Tensor networks and tensorized layers

#### *Tensorizing Neural Networks* (`4.pdf`)

The paper converts dense fully connected matrices into Tensor-Train form.
Instead of storing one large matrix, the layer stores a sequence of small
cores connected by internal ranks. This can yield very large reductions in the
number of stored parameters while retaining expressive power when the chosen
factorization and ranks are appropriate.

The implementation implication is that factorization requires:

- compatible input and output factorizations;
- an explicit rank/bond budget;
- a stable reshape and permutation convention;
- a reconstruction or forward-pass test against the original layer;
- a quality evaluation after replacing the layer.

The repository's MPO path follows this pattern. Its `mpo_sites` controls the
number of cores, while `bond_dim` in the research path controls the maximum
internal TT rank directly.

#### *A Practical Introduction to Tensor Networks* (`1.pdf`)

This tutorial explains MPS and PEPS from a numerical perspective. The most
important transferable concepts are contraction order, bond dimension, and
approximation through truncation. Increasing bond dimension generally increases
representational capacity and cost; decreasing it increases compression but
can discard correlations needed by the original matrix.

For an LLM compressor, this means that a single global setting is a useful
baseline but not necessarily optimal. A future implementation should consider
per-layer bond dimensions selected using calibration error, sensitivity, or a
fixed memory budget.

#### *Tensor Networks for Complex Quantum Systems* (`14.pdf`)

This review places tensor networks in the context of entanglement and
renormalization. It explains why low bond dimensions can describe structured
states efficiently while highly entangled states require larger bonds. For
neural-network weights, this gives a useful interpretation rather than a
guarantee: tensor compression works best where the matrix has exploitable
structure in the chosen reshaping.

### 4. Variational and quantum-inspired tensor neural networks

#### *Quantum-Inspired Tensor Neural Networks for Partial Differential Equations* (`6.pdf`)

The paper uses tensor neural networks to reduce parameters and training cost
for PDE solvers while maintaining accuracy. Its relevance is methodological:
tensorization can be valuable even when the target system is not a quantum
computer, and parameter savings should be reported together with accuracy and
training/inference cost.

#### *Variational Tensor Neural Networks for Deep Learning* (`7.pdf` and `8.pdf`)

This paper combines tensor-network layers with DMRG-inspired local
optimization. It supports a stronger research direction than one-shot TT-SVD:
initialize compressed cores from a decomposition, then optimize them against
the task loss. Hybrid dense/tensor models are also useful because not every
layer benefits equally from tensorization.

The duplicate PDFs should be removed or renamed in a future cleanup if the
collection is intended for automated literature statistics.

#### *Tensor Networks Meet Neural Networks: A Survey and Future Perspectives* (`9.pdf`)

The survey organizes tensorial neural networks by data processing and model
architecture. It covers compression, multimodal fusion, multi-task learning,
quantum data, and integrations with Transformers and LLMs. Its main value for
this repository is as a map of alternatives: TT/MPO is one point in a larger
design space that includes CP, Tucker, tensor-ring, tensorized convolution,
recurrent, graph, and attention-based constructions.

## Cross-paper findings

### Finding 1: Compression is a rate-distortion problem

The papers repeatedly trade storage or compute against quality. The relevant
curve is not “maximum compression” but the Pareto frontier between:

- serialized checkpoint size;
- peak RAM and accelerator memory;
- latency and throughput on the target device;
- next-token loss/perplexity;
- downstream task quality;
- recovery or fine-tuning cost.

### Finding 2: Structure and layer sensitivity matter

Tensorization assumes that the reshaped matrix contains compressible
structure. Transformer layers also have different sensitivity. Attention
output projections, feed-forward projections, embeddings, and the language
model head should not automatically receive the same rank. The current
protected-layer policy is therefore a sound safety mechanism, but it should be
validated empirically for each model family.

### Finding 3: One-shot decomposition is a baseline, not the final method

TT-SVD/MPO gives a deterministic and reproducible initialization. The pruning
and variational papers suggest that quality can improve substantially through
retraining or local optimization. A research-grade pipeline should compare:

1. one-shot TT-SVD;
2. one-shot TT-SVD plus calibration;
3. TT-SVD plus parameter-efficient healing;
4. mixed tensor/quantized layers;
5. an uncompressed baseline.

### Finding 4: Hardware determines whether compression is useful

A lower parameter count can reduce storage while increasing runtime if the
target runtime lacks efficient kernels for the sequence of contractions.
Benchmarks must run on the intended CPU/GPU/mobile target and include warm-up,
batch size, sequence length, peak memory, and end-to-end latency.

### Finding 5: Reproducibility requires explicit configuration

The papers use different datasets, model sizes, ranks, and evaluation tasks.
Comparisons are only meaningful when the report records model identifier,
checkpoint revision, tokenizer, factorization shape, bond/rank budget, skipped
modules, calibration data, hardware, software versions, and random seeds.

## Recommendations for this repository

### Immediate

1. Keep the explicit `bond_dim` research path separate from the legacy
   `rank_ratio` path; they represent different truncation policies.
2. Add a report entry for every compression run containing parameter count,
   serialized size, peak memory, latency, throughput, and next-token quality.
3. Evaluate the original and compressed models on exactly the same tokenizer,
   prompts, sequence lengths, and held-out JSONL data.
4. Keep protected layers configurable and report the exact matched module paths.
5. Treat `7.pdf` and `8.pdf` as one source in future automated analyses.

### Next research experiments

| Experiment | Independent variable | Measurements |
|---|---|---|
| Bond-dimension sweep | `bond_dim` per layer or globally | Size, reconstruction error, perplexity, latency |
| Site-count sweep | `mpo_sites` | Core count, contraction cost, quality |
| Sensitivity-aware policy | Protected versus compressed module groups | Quality loss by module group |
| Healing study | No healing, calibration, QLoRA healing | Recovery versus trainable parameter count |
| Mixed compression | MPO plus int8 quantization | End-to-end size, speed, and quality |
| Hardware study | CPU, CUDA GPU, and mobile/edge target | Peak memory, warm latency, throughput |

## Limitations

This report is a technical synthesis of the supplied PDFs and the current
repository design. It does not reproduce the papers' experiments, verify their
reported numerical results, or establish that a given compression setting will
preserve a particular model's behavior. Several PDFs are older foundational
works, while the newest survey and layer-pruning paper may have version-specific
details. Results should therefore be cited from the original papers and
reproduced with the repository's own evaluation data before making claims about
production quality.
