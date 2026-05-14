"""Wraps LLM interface."""

import re

from llm_attacks.configs.protocol import ModelConfig

from .base import LLM
from .gemma3_llm import Gemma3LLM
from .gpt_oss_llm import GPTOssLLM
from .llama2_llm import Llama2LLM
from .qwen3_5_llm import Qwen3_5LLM
from .qwen3_5_thinking_llm import Qwen3_5ThinkingLLM
from .qwen3_llm import Qwen3LLM
from .vicuna_llm import VicunaLLM


def create_llm(args: ModelConfig, device_map="auto") -> LLM:
    model_base_name = args.model.rsplit("/", 1)[-1].lower()
    model_id = args.model
    model_kwargs = args.model_kwargs.copy()
    model_kwargs.update(device_map=device_map)
    if re.match(r"gemma-3", model_base_name):
        return Gemma3LLM(model_id, model_kwargs=model_kwargs)
    elif re.match(r"(meta-)?llama-[23]", model_base_name):
        llm = Llama2LLM(model_id, model_kwargs=model_kwargs)
        llm.system_prompt = ""
        return llm
    elif re.match(r"vicuna-[0-9]+b-v1\.", model_base_name):
        llm = VicunaLLM(model_id, model_kwargs=model_kwargs)
        return llm
    elif re.match(r"qwen3\.5-", model_base_name):
        if args.enable_thinking:
            llm = Qwen3_5ThinkingLLM(model_id, model_kwargs=model_kwargs)
        else:
            llm = Qwen3_5LLM(model_id, model_kwargs=model_kwargs)
        return llm
    elif re.match(r"qwen3-", model_base_name):
        llm = Qwen3LLM(model_id, model_kwargs=model_kwargs)
        return llm
    elif re.match(r"gpt-oss", model_base_name):
        llm = GPTOssLLM(model_id, model_kwargs=model_kwargs)
        return llm
    else:
        raise ValueError(f"Unknown model {model_base_name}")
