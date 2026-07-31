from __future__ import annotations

import torch

from llm_edge_compression.inference import _decode_continuation


class DummyTokenizer:
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        return " ".join(str(int(token)) for token in token_ids.tolist())


def test_decode_continuation_returns_only_generated_tokens() -> None:
    tokenizer = DummyTokenizer()
    prompt_input_ids = torch.tensor([[10, 11, 12]])
    output_ids = torch.tensor([[10, 11, 12, 20, 21]])

    assert _decode_continuation(tokenizer, prompt_input_ids, output_ids) == "20 21"


def test_decode_continuation_returns_empty_string_when_no_new_tokens() -> None:
    tokenizer = DummyTokenizer()
    prompt_input_ids = torch.tensor([[10, 11, 12]])
    output_ids = torch.tensor([[10, 11, 12]])

    assert _decode_continuation(tokenizer, prompt_input_ids, output_ids) == ""