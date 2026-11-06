from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from .layer_pruner import LayerPruningCompressor


@dataclass(slots=True)
class QLoRAHealingConfig:
    output_dir: Path
    dataset: str = "allenai/c4"
    dataset_config: str = "en"
    dataset_split: str = "train"
    max_samples: int = 256
    max_seq_length: int = 512
    max_steps: int = 500
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 0
    compute_dtype: str = "bfloat16"
    trust_remote_code: bool = False


@dataclass(slots=True)
class QLoRAHealingResult:
    adapter_dir: Path
    target_modules: list[str]
    trainable_parameters: int
    total_parameters: int
    max_steps: int
    dataset: str
    max_samples: int


def _resolve_compute_dtype(name: str) -> torch.dtype:
    normalized = name.lower().replace("-", "")
    if normalized in {"bfloat16", "bf16"}:
        if not torch.cuda.is_bf16_supported():
            return torch.float16
        return torch.bfloat16
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    raise ValueError("compute_dtype must be bfloat16 or float16")


def find_lora_target_modules(model: torch.nn.Module) -> list[str]:
    known_suffixes = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "c_attn",
        "c_proj",
        "c_fc",
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    }
    found: set[str] = set()
    for name, module in model.named_modules():
        if name.endswith(".lm_head") or name == "lm_head":
            continue
        suffix = name.rsplit(".", 1)[-1]
        if suffix in known_suffixes:
            found.add(suffix)

    if found:
        return sorted(found)

    linear_suffixes: set[str] = set()
    for name, module in model.named_modules():
        if name.endswith(".lm_head") or name == "lm_head":
            continue
        if isinstance(module, torch.nn.Linear):
            linear_suffixes.add(name.rsplit(".", 1)[-1])
    if not linear_suffixes:
        raise ValueError("Could not find supported linear attention/MLP modules for LoRA.")
    return sorted(linear_suffixes)


def _prepare_text_dataset(
    dataset_name: str,
    dataset_config: str,
    split: str,
    tokenizer: Any,
    max_samples: int,
    max_seq_length: int,
):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "QLoRA healing requires the optional 'healing' dependencies. "
            "Install with: pip install -e '.[healing]'"
        ) from exc

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        streaming=True,
    )
    if max_samples > 0:
        dataset = dataset.take(max_samples)

    def tokenize(example: dict[str, Any]) -> dict[str, Any]:
        text = str(example.get("text", ""))
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        encoded["labels"] = list(encoded["input_ids"])
        return encoded

    return dataset.map(tokenize)


def _build_calibration_batches(dataset: Iterable[dict[str, Any]], batch_size: int = 1, limit: int = 8):
    batches: list[dict[str, torch.Tensor]] = []
    current: list[list[int]] = []
    for example in dataset:
        ids = example.get("input_ids")
        if not ids:
            continue
        current.append(list(ids))
        if len(current) < batch_size:
            continue
        width = min(len(x) for x in current)
        if width > 0:
            batches.append({"input_ids": torch.tensor([x[:width] for x in current], dtype=torch.long)})
        current = []
        if len(batches) >= limit:
            break
    return batches


def heal_with_qlora(
    model_id: str,
    config: QLoRAHealingConfig,
    *,
    layer_pruning_num_layers: int = 0,
    layer_pruning_ratio: float = 0.0,
    layer_pruning_strategy: str = "similarity",
    layer_pruning_seed: int = 0,
    layer_pruning_selected_layers: tuple[int, ...] = (),
) -> QLoRAHealingResult:
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA healing requires a CUDA-capable GPU.")

    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise RuntimeError(
            "QLoRA healing requires PEFT. Install with: pip install -e '.[healing]'"
        ) from exc

    dtype = _resolve_compute_dtype(config.compute_dtype)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=config.trust_remote_code,
    )

    if layer_pruning_num_layers or layer_pruning_ratio or layer_pruning_selected_layers:
        compressor = LayerPruningCompressor(
            num_remove=layer_pruning_num_layers or None,
            prune_ratio=layer_pruning_ratio or None,
            strategy=layer_pruning_strategy,
            seed=layer_pruning_seed,
            selected_layers=layer_pruning_selected_layers or None,
        )
        if layer_pruning_strategy == "similarity" and not layer_pruning_selected_layers:
            calibration_source = _prepare_text_dataset(
                config.dataset,
                config.dataset_config,
                config.dataset_split,
                tokenizer,
                min(config.max_samples, 32) if config.max_samples > 0 else 32,
                config.max_seq_length,
            )
            calibration_batches = _build_calibration_batches(calibration_source, batch_size=1, limit=8)
            compressor.fit(model, calibration_batches)
        else:
            compressor.fit(model, [])
        model = compressor.compress(model)

    target_modules = find_lora_target_modules(model)
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)

    dataset = _prepare_text_dataset(
        config.dataset,
        config.dataset_config,
        config.dataset_split,
        tokenizer,
        config.max_samples,
        config.max_seq_length,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=max(1, min(10, config.max_steps)),
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16,
        seed=config.seed,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return QLoRAHealingResult(
        adapter_dir=config.output_dir,
        target_modules=target_modules,
        trainable_parameters=trainable,
        total_parameters=total,
        max_steps=config.max_steps,
        dataset=config.dataset,
        max_samples=config.max_samples,
    )
