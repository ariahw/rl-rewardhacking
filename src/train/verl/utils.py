from __future__ import annotations

from verl import DataProto
from transformers import AutoTokenizer


def convert_responses_to_str(data: DataProto, tokenizer: AutoTokenizer):
    prompt_ids = data.batch["prompts"]
    response_ids = data.batch["responses"]
    attention_mask = data.batch["attention_mask"]

    prompt_len = prompt_ids.shape[-1]
    valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

    responses_str = []
    for i in range(len(data)):
        valid_len = valid_response_lengths[i]
        valid_response_ids = response_ids[i][:valid_len]
        response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        responses_str.append(response_str)
    return responses_str