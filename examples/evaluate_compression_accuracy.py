#!/usr/bin/env python3
"""Compare original and compressed causal language models on JSONL text data.

Each input line must be a JSON object with a non-empty ``text`` field, for
example::

    {"text": "The capital of France is Paris."}

The reported accuracy is exact next-token accuracy. This is a useful
regression metric for compression, but it is not a downstream task score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM

from llm_edge_compression.inference import load_compressed_bundle


def _read_texts(data_path: Path, max_samples: int | None) -> Iterable[str]:
    samples = 0
    with data_path.open(encoding="utf-8") as data_file:
        for line_number, line in enumerate(data_file, start=1):
            if not line.strip():
                continue
            try:
                record: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"Line {line_number} must contain a string 'text' field")
            text = record["text"].strip()
            if not text:
                continue
            yield text
            samples += 1
            if max_samples is not None and samples >= max_samples:
                return


def _next_token_counts(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: Iterable[str],
    device: str,
    max_length: int,
) -> tuple[int, int, int]:
    correct = 0
    total = 0
    examples = 0
    model.eval()

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = inputs["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue

        model_inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**model_inputs).logits

        predictions = logits[:, :-1, :].argmax(dim=-1)
        targets = input_ids[:, 1:]
        correct += int((predictions == targets).sum().item())
        total += int(targets.numel())
        examples += 1

    return correct, total, examples


def evaluate_accuracy(
    bundle_dir: Path,
    data_path: Path,
    *,
    device: str = "cpu",
    max_length: int = 512,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Evaluate next-token accuracy before and after compression."""
    loaded = load_compressed_bundle(bundle_dir, device=device)
    tokenizer = loaded.tokenizer
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = list(_read_texts(data_path, max_samples))
    if not texts:
        raise ValueError("The evaluation data contains no usable text records")

    original_model = AutoModelForCausalLM.from_pretrained(
        loaded.manifest.model_id,
        low_cpu_mem_usage=True,
        trust_remote_code=loaded.compression.trust_remote_code,
    ).to(device)
    compressed_correct, token_count, compressed_examples = _next_token_counts(
        loaded.model, tokenizer, texts, device, max_length
    )
    original_correct, original_token_count, original_examples = _next_token_counts(
        original_model, tokenizer, texts, device, max_length
    )
    if token_count == 0 or token_count != original_token_count:
        raise ValueError("Evaluation data produced no comparable next-token targets")

    original_accuracy = original_correct / token_count
    compressed_accuracy = compressed_correct / token_count
    accuracy_drop = original_accuracy - compressed_accuracy
    relative_drop = accuracy_drop / original_accuracy if original_accuracy else 0.0
    return {
        "model_id": loaded.manifest.model_id,
        "bundle_dir": str(bundle_dir),
        "data_path": str(data_path),
        "examples": min(compressed_examples, original_examples),
        "tokens": token_count,
        "original_accuracy": original_accuracy,
        "compressed_accuracy": compressed_accuracy,
        "accuracy_drop": accuracy_drop,
        "accuracy_drop_percentage_points": accuracy_drop * 100.0,
        "relative_accuracy_drop": relative_drop,
        "relative_accuracy_drop_percent": relative_drop * 100.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True, help="Compressed model directory")
    parser.add_argument("--data", type=Path, required=True, help="JSONL file containing {'text': ...} records")
    parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum tokens evaluated per text")
    parser.add_argument("--max-samples", type=int, help="Only evaluate the first N records")
    parser.add_argument("--output", type=Path, help="Optional path for the JSON report")
    args = parser.parse_args()

    if args.max_length < 2:
        parser.error("--max-length must be at least 2")
    if not args.data.is_file():
        parser.error(f"evaluation data file not found: {args.data}")
    report = evaluate_accuracy(
        args.bundle_dir,
        args.data,
        device=args.device,
        max_length=args.max_length,
        max_samples=args.max_samples,
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Accuracy report saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())