"""Base class `LLMUserInterface`."""

import logging
from typing import Iterable, Literal, Optional, TypeVar, overload, override

import torch
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BatchEncoding,
    BatchFeature,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)

from llm_attacks.utils.prompt import *

try:
    import flash_attn

    has_flash_attn = True
except ImportError:
    has_flash_attn = False

T = TypeVar("T")

_logger = logging.getLogger(__name__)


class LLM:
    """Stateless LLM interface."""

    DEFAULT_SYSTEM_PROMPT: str | None = None
    MASK_TOKEN = -100  # mask out labels

    def __init__(
        self,
        model_id="google/gemma-3-4b-it",
        model_kwargs: Optional[dict] = None,
        processor_kwargs: Optional[dict] = None,
    ):
        all_model_kwargs = dict(device_map="auto", dtype="auto")
        all_model_kwargs.update(model_kwargs or {})
        _logger.info(f"Loading model {model_id}...")
        if has_flash_attn:
            all_model_kwargs.update(attn_implementation="flash_attention_2")
        else:
            _logger.warning(
                "Flash attention not found, try 'pip install flash-attn' or "
                "check flash-attention compatibility"
            )
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_id, **all_model_kwargs)
            .eval()
            .requires_grad_(False)
        )
        _logger.info(f"Model loaded as class {type(self.model).__name__}")
        _logger.info(f"Model dtype: {self.model.dtype}, device: {self.model.device}")
        # For a multi-modal model, this is ProcessorMixin, which is a multi-modal
        # processor. For a text model, this is PreTrainedTokenizerBase, which is directly
        # the tokenizer
        all_processor_kwargs = dict()
        all_processor_kwargs.update(processor_kwargs or {})
        self.processor: ProcessorMixin | PreTrainedTokenizerBase = (
            AutoProcessor.from_pretrained(model_id, **all_processor_kwargs)
        )
        _logger.info(f"Processor class is {type(self.processor).__name__}")
        self.system_prompt = self.DEFAULT_SYSTEM_PROMPT

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if isinstance(self.processor, PreTrainedTokenizerBase):
            return self.processor
        else:
            return self.processor.tokenizer

    def is_multimodal(self):
        return not isinstance(self.processor, PreTrainedTokenizerBase)

    @overload
    def apply_chat_template(
        self,
        prompts: BatchedPromptLike,
        targets: list[str] | None = ...,
        add_generation_prompt=...,
        continue_final_message=...,
        tokenize: Literal[True] = True,
        **kwargs,
    ) -> dict: ...

    @overload
    def apply_chat_template(
        self,
        prompts: BatchedPromptLike,
        targets: list[str] | None = ...,
        add_generation_prompt=...,
        continue_final_message=...,
        tokenize: Literal[False] = False,
        **kwargs,
    ) -> list[str]: ...

    def apply_chat_template(
        self,
        prompts: BatchedPromptLike,
        targets: list[str] | None = None,
        add_generation_prompt=False,
        continue_final_message=False,
        tokenize=True,
        **kwargs,
    ) -> list[str] | dict:
        """Apply chat template to a batch of prompt.

        Args:
            prompts (BatchedPromptLike):
                User prompt or preceding conversation.
            targets (Optional[list[str]]):
                Target labels. Default: None.
            add_generation_prompt (bool):
                Whethere to append 'Assistant:'. Default: False.
            continue_final_message (bool):
                Whether to continue final message. Default: False.
            tokenize (bool):
                Return tokenized output or not. Default: True.
            kwargs:
                All kwargs passed to `processor.apply_chat_template`.

        Returns:
            rv (list[str] | dict):
                If tokenize=True, then rv is inputs to the model.
                Otherwise, rv is string before tokenization.
        """
        raise NotImplementedError

    @overload
    def generate(
        self, prompt: PromptLike, tokenize_kwargs: dict | None = ..., **kwargs
    ) -> str:
        """Generate response from LLM.

        Args:
            prompt (PromptLike):
                Single or a batch of prompts.
                Each prompt is a single message or a list of messages.
            kwargs:
                All generate kwargs.

        Returns:
            message (str):
                Fully decoded output.
        """
        ...

    @overload
    def generate(
        self, prompt: BatchedPromptLike, tokenize_kwargs: dict | None = ..., **kwargs
    ) -> list[str]:
        """Generate a batch of response from LLM.

        Args:
            prompt (list[PromptLike]):
                Batch of prompts.
                Each prompt is a single message or a list of messages.
            kwargs:
                All generate kwargs.

        Returns:
            message (list[str]):
                Fully decoded output.
        """
        ...

    def generate(
        self,
        prompt: PromptLike | BatchedPromptLike,
        tokenize_kwargs: dict | None = None,
        **kwargs,
    ) -> str | list[str]:
        """Generate response from LLM.

        Args:
            prompt (PromptLike | list[PromptLike]):
                Single or a batch of prompts.
                Each prompt is a single message or a list of messages.
            tokenize_kwargs (Optional[dict]):
                Kwargs of tokenization.
            kwargs:
                All generate kwargs.

        Returns:
            message (str | list[str]):
                Fully decoded output.
        """
        raise NotImplementedError

    def get_loss(self, inputs: dict, **kwargs) -> torch.FloatTensor:
        """Get model loss.

        Args:
            inputs (dict):
                Model inputs.
            **kwargs:
                All arguments to pass to model forward.

        Returns:
            loss (Tensor):
                Model loss.
        """
        raise NotImplementedError
