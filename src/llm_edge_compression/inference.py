from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .compressors import DynamicQuantizationCompressor, MPOCompressor, TensorNetworkCompressor
from .config import CompressionConfig, compression_config_from_dict
from .manifest import ModelManifest, read_manifest


@dataclass(slots=True)
class LoadedCompressedModel:
    manifest: ModelManifest
    compression: CompressionConfig
    tokenizer: Any
    model: torch.nn.Module


def _build_compressor(compression: CompressionConfig):
    if compression.method == "quantize":
        return DynamicQuantizationCompressor(backend=compression.quantization_backend)
    if compression.method == "mpo":
        return MPOCompressor(rank_ratio=compression.rank_ratio, layer_policy=compression.layer_policy)
    return TensorNetworkCompressor(rank_ratio=compression.rank_ratio, layer_policy=compression.layer_policy)


def load_compressed_bundle(bundle_dir: Path, device: str = "cpu") -> LoadedCompressedModel:
    manifest = read_manifest(bundle_dir / "manifest.json")
    compression = compression_config_from_dict(manifest.compression_config or {"model_id": manifest.model_id, "output_dir": bundle_dir.as_posix()})
    tokenizer = AutoTokenizer.from_pretrained(manifest.model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(manifest.model_id, low_cpu_mem_usage=True)
    model = _build_compressor(compression).compress(model)

    state_dict = torch.load(bundle_dir / "compressed_model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return LoadedCompressedModel(manifest=manifest, compression=compression, tokenizer=tokenizer, model=model)


def generate_text(
    bundle_dir: Path,
    prompt: str,
    device: str = "cpu",
    max_new_tokens: int = 64,
    temperature: float = 0.0,
) -> str:
    loaded = load_compressed_bundle(bundle_dir, device=device)
    tokenizer = loaded.tokenizer
    model = loaded.model

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generation_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if temperature > 0:
        generation_kwargs.update({"do_sample": True, "temperature": temperature})

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def chat_loop(
    bundle_dir: Path,
    device: str = "cpu",
    max_new_tokens: int = 64,
    temperature: float = 0.0,
) -> None:
    loaded = load_compressed_bundle(bundle_dir, device=device)
    tokenizer = loaded.tokenizer
    model = loaded.model
    print(f"Loaded {loaded.manifest.model_id} from {bundle_dir}")
    print("Type a prompt and press Enter. Use /exit to quit.")

    while True:
        prompt = input("prompt> ").strip()
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            break

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        generation_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": temperature})

        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)

        print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
