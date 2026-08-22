# MPO Compression Results

## Overview

This document records the current experimental results for MPO-based compression of GPT-2 in the `llm_compression` project.

The objective is to reduce the parameter count and serialized model size while preserving useful inference behavior.

## Test Environment

- Model: `gpt2`
- Target device: CPU
- MPO sites: 3
- Rank ratio used for the main compression experiment: `0.5`
- Healing steps: `0`
- Python: `3.13.3`
- Test suite: `23 passed, 1 warning`

## Initial Implementation Issue

The initial MPO compressor only detected `torch.nn.Linear` modules. GPT-2 uses Hugging Face `Conv1D` projection modules for important transformer operations such as attention and MLP projections.

As a result, the first MPO experiment reported:

```text
Original parameters:    124,439,808
Compressed parameters:  124,439,808
Parameter ratio:        1.0000
Original model size:    474.75 MB
Compressed model size:  474.75 MB
Size ratio:             1.0000
Size reduction:         0.00%
```

This was not a genuine compressed GPT-2 representation.

## Conv1D Fix

The MPO compressor was updated to recognize both `nn.Linear` and Hugging Face `Conv1D` modules.

For GPT-2 `Conv1D`, the stored weight has the opposite orientation from a standard PyTorch linear layer. The compressor therefore transposes the weight before MPO decomposition and stores the resulting MPO cores in `MPOLinear`.

This allows the large GPT-2 projection layers to be represented using MPO cores rather than reconstructed dense weights.

## Successful MPO Compression Result

With `rank_ratio=0.5`, the resulting bundle was `artifacts/gpt2-mpo-v3`.

| Metric | Original GPT-2 | MPO (`rank_ratio=0.5`) |
|---|---:|---:|
| Parameters | 124,439,808 | 72,023,808 |
| Parameter ratio | 1.0000 | 0.5788 |
| Model size | 474.75 MB | 274.83 MB |
| Size ratio | 1.0000 | 0.5789 |
| Size reduction | 0% | **42.11%** |

The compressed model therefore uses approximately 57.9% of the original serialized state-dict size, corresponding to a **42.11% reduction**.

## Inference Quality Finding

Compression size alone is not sufficient to demonstrate a successful LLM compression method. The compressed bundle was loaded successfully, but generation at `rank_ratio=0.5` produced highly repetitive output, for example repeated occurrences of a single token such as `Women`.

This indicates substantial inference-quality degradation at the current global rank ratio.

The current conclusion is therefore:

> The MPO representation and storage compression are working, but the current rank-selection strategy does not yet preserve acceptable GPT-2 generation quality at `rank_ratio=0.5`.

This should be reported as an experimental limitation rather than hidden or treated as a successful quality-preserving compression result.

## Recommended Evaluation

The next experiments should evaluate multiple rank ratios rather than using a single global value:

```text
1.0
0.8
0.7
0.6
0.5
```

For every ratio, record:

- parameter count
- serialized model size
- parameter ratio
- size reduction percentage
- logit MSE against the original model
- maximum absolute logit error
- relative logit error
- top-1 token agreement
- perplexity
- generation quality

This will establish the compression-versus-quality trade-off.

## Full-Rank MPO Baseline

A `rank_ratio=1.0` experiment should be treated as a **full-rank MPO baseline**, not simply called the original GPT-2 model. It still passes through the MPO decomposition and reconstruction path.

The full-rank baseline is useful for determining whether the MPO implementation itself preserves the original computation before rank truncation is made more aggressive.

## Current Status

### Working

- GPT-2 MPO compression is implemented.
- GPT-2 `Conv1D` layers are handled by the MPO compressor.
- MPO cores are stored in the compressed model.
- Parameter-count metrics are reported.
- Serialized original and compressed model sizes are reported.
- `rank_ratio=0.5` currently achieves 42.11% model-size reduction.
- The test suite currently passes 23 tests.

### Needs further work

- Improve inference fidelity at lower MPO ranks.
- Perform quantitative logit-error evaluation.
- Measure perplexity against the original GPT-2 model.
- Evaluate layer-wise or adaptive MPO ranks.
- Determine whether calibration/healing can recover generation quality.
- Identify projection layers that are especially sensitive to rank truncation.

## Presentation Conclusion

The current work demonstrates a functioning MPO-based GPT-2 compression prototype with a measured **42.11% reduction in serialized model size** at `rank_ratio=0.5`. However, the same configuration currently causes significant generation degradation. Therefore, the next research objective is not simply greater compression, but finding the best compression-quality trade-off through rank selection and quantitative fidelity evaluation.
