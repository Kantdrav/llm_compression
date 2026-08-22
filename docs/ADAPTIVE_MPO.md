# Adaptive MPO Mode

The existing MPO method and fixed `--rank-ratio` behavior are preserved.

## Fixed MPO

```bash
llm-edge-compression compress \
  --model-id gpt2 \
  --output-dir artifacts/gpt2-mpo-fixed \
  --method mpo \
  --rank-ratio 0.5
```

This continues to use one fixed rank ratio for every compressible layer.

## Adaptive MPO

Adaptive mode selects a rank ratio separately for each compressible Linear or GPT-2 Conv1D layer using the cumulative squared singular-value energy of that layer.

Example:

```bash
llm-edge-compression compress \
  --model-id gpt2 \
  --output-dir artifacts/gpt2-mpo-adaptive \
  --method mpo \
  --adaptive-rank \
  --target-reduction 0.30 \
  --adaptive-energy-threshold 0.995
```

### Selection logic

For each layer:

1. Convert GPT-2 Conv1D weights to the standard output-by-input orientation.
2. Compute singular values.
3. Find the smallest rank retaining the configured spectral-energy threshold.
4. Convert that rank into an MPO rank ratio.
5. Apply a conservative retained-rank floor derived from `1 - target_reduction`.
6. Perform the existing three-site MPO decomposition with that layer-specific ratio.

The target reduction is intentionally a target/floor rather than a forced exact file-size percentage. This avoids aggressively lowering ranks just to hit a number and is intended to favor inference fidelity.

## Important

Adaptive rank selection is a compression heuristic, not a guarantee of preserved perplexity or generation quality. Always compare the resulting model against the original using logit error, top-1 agreement, perplexity, and generation tests.

A good starting point for GPT-2 is:

```text
--target-reduction 0.30
--adaptive-energy-threshold 0.995
```

If output quality remains poor, increase the energy threshold (for example `0.998`) and/or lower the target reduction.
