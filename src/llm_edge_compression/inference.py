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


def _decode_continuation(tokenizer: Any, prompt_input_ids: torch.Tensor, output_ids: torch.Tensor) -> str:
    prompt_length = prompt_input_ids.shape[-1]
    continuation_ids = output_ids[0][prompt_length:]
    if continuation_ids.numel() == 0:
        return ""
    return tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()


def load_compressed_bundle(bundle_dir: Path, device: str = "cpu") -> LoadedCompressedModel:
    manifest = read_manifest(bundle_dir / "manifest.json")
    compression = compression_config_from_dict(manifest.compression_config or {"model_id": manifest.model_id, "output_dir": bundle_dir.as_posix()})
    tokenizer = AutoTokenizer.from_pretrained(
        manifest.model_id,
        use_fast=True,
        trust_remote_code=compression.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        manifest.model_id,
        low_cpu_mem_usage=True,
        trust_remote_code=compression.trust_remote_code,
    )
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
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
) -> str:
    loaded = load_compressed_bundle(bundle_dir, device=device)
    tokenizer = loaded.tokenizer
    model = loaded.model

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": repetition_penalty,
    }
    if temperature > 0:
        generation_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p, "top_k": top_k})
    else:
        generation_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    return _decode_continuation(tokenizer, inputs["input_ids"], output_ids)


def chat_loop(
    bundle_dir: Path,
    device: str = "cpu",
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
) -> None:
    loaded = load_compressed_bundle(bundle_dir, device=device)
    tokenizer = loaded.tokenizer
    model = loaded.model
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Loaded {loaded.manifest.model_id} from {bundle_dir}")
    print("Type a prompt and press Enter. Use /exit to quit.")

    while True:
        prompt = input("prompt> ").strip()
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            break

        chat_prompt = f"User: {prompt}\nAssistant:"
        inputs = tokenizer(chat_prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": repetition_penalty,
        }
        if temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p, "top_k": top_k})
        else:
            generation_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)

        response = _decode_continuation(tokenizer, inputs["input_ids"], output_ids)
        print(response if response else "(no new text generated)")
