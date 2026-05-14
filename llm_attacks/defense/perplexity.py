import torch

from llm_attacks.gcg.probe_sampling import _get_draft_model


def get_perplexity(
    text: str, model_name="openai-community/gpt2", device="cuda:0"
) -> torch.Tensor:
    model, tokenizer = _get_draft_model(model_name, device=device)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss)
