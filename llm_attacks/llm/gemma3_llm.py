## the interface we now use
import copy
import re
import warnings
from typing import TypedDict

import torch
from torch import Tensor, nn
from transformers import Gemma3ForConditionalGeneration, Gemma3Processor
from transformers.models.gemma3.modeling_gemma3 import Gemma3CausalLMOutputWithPast

from .base import *
from .base import _logger


class Gemma3ModelInput(TypedDict):
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    labels: torch.LongTensor | None


class Gemma3LLM(LLM):
    """Gemma3 interface. Typical multi-modal model.

    Attributes:
        model (Gemma3ForConditionalGeneration):
            LLM.
        processor (Gemma3Processor):
            Input processor.
        generate_kwargs (dict):
            `model.generate` configs.
        dtype (torch.dtype):
            Dtype for inputs.
    """

    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
    MASK_TOKEN = -100  # mask out labels

    @override
    def apply_chat_template(
        self,
        prompts: BatchedPromptLike,
        targets: list[str] | None = None,
        add_generation_prompt=False,
        continue_final_message=False,
        tokenize=True,
        **kwargs,
    ) -> list[str] | BatchFeature | BatchEncoding:
        # 0. Check parameters
        if targets is not None:
            # These parameters are dummy in this case
            add_generation_prompt = True
            continue_final_message = False

        # 1. Apply default chat template
        is_mm = self.is_multimodal()
        conversations: list[Conversation] = [
            make_conversation(
                p,
                system_prompt=self.system_prompt,
                copy_=True,
                pure_text=not is_mm,
                pure_multimodal=is_mm,
            )
            for p in prompts
        ]

        # 2. tokenize inputs
        if tokenize:
            tokenize_kwargs = dict(
                tokenize=tokenize,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            tokenize_kwargs = dict(tokenize=tokenize)
        tokenize_kwargs.update(kwargs)
        prompt_tokens: list[str] | Gemma3ModelInput = (
            self.processor.apply_chat_template(
                conversations,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                # padding=True,
                **tokenize_kwargs,
            )
        )
        if targets is None:
            return prompt_tokens

        # 3. Append assistant response and tokenize everything, that's the model input.
        # This 'double tokenization' is used by `trl.SFTTrainer`, so we assume it not
        # very time-consuming.
        for target, conversation in zip(targets, conversations):
            if is_mm:
                content = [{"type": "text", "text": target}]
            else:
                content = target
            conversation.append({"role": "assistant", "content": content})
        inputs: list[str] | Gemma3ModelInput = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=False,
            # this argument strips the last '<end_of_turn>'
            continue_final_message=True,
            # padding=True,
            **tokenize_kwargs,
        )
        if not tokenize:
            return inputs

        # 4. Inject labels into inputs
        # Note: the following is performed on CPU, since there's not a lot tokens
        input_ids: torch.LongTensor = inputs["input_ids"]
        attention_mask: torch.LongTensor = inputs["attention_mask"]
        labels = input_ids.clone().detach()
        for i in range(input_ids.shape[0]):
            prompt_len = prompt_tokens["attention_mask"][i].sum().item()
            total_len = attention_mask[i].sum().item()
            if not torch.all(
                prompt_tokens["input_ids"][i][-prompt_len:]
                == input_ids[i][-total_len : -total_len + prompt_len]
            ):
                warnings.warn(
                    "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                    "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                    "token handling. Verify that the tokenizer is processing text consistently."
                )
            # Mask out the paddings
            labels[i][attention_mask[i] == 0] = self.MASK_TOKEN
            # Mask out input prompts
            labels[i, -total_len : -total_len + prompt_len] = self.MASK_TOKEN
        inputs["labels"] = labels
        inputs = inputs.to(self.device)
        if isinstance(inputs, BatchFeature):
            inputs = inputs.to(self.dtype)
        return inputs

    @override
    def generate(
        self,
        prompt: PromptLike | BatchedPromptLike,
        tokenize_kwargs: dict | None = None,
        **kwargs,
    ) -> str | list[str]:
        if is_prompt_batched(prompt):
            is_batched = True
            prompts = prompt
        else:
            is_batched = False
            prompts = [prompt]

        # Apply chat template and tokenize
        inputs: Gemma3ModelInput = self.apply_chat_template(
            prompts=prompts,
            add_generation_prompt=True,
            tokenize=True,
            **(tokenize_kwargs or {}),
        ).to(self.device)
        if isinstance(inputs, BatchFeature):
            inputs = inputs.to(self.dtype)
        input_len = inputs["input_ids"].shape[-1]

        # Generate
        with torch.inference_mode():
            generation: torch.LongTensor = self.model.generate(**inputs, **kwargs)
            generation = generation[:, input_len:]

        decoded: list[str] = self.processor.batch_decode(
            generation, skip_special_tokens=True
        )
        return decoded if is_batched else decoded[0]

    @override
    def get_loss(
        self, inputs: BatchFeature, reduction="mean", **kwargs
    ) -> torch.FloatTensor:
        """Get model loss.

        Args:
            inputs (BatchFeature):
                Model inputs dict.
            reduction (str):
                Reduction. Choices: 'mean', 'sum', 'none'. Default: 'mean'.
            **kwargs:
                All arguments to pass to model forward.

        Returns:
            loss (Tensor):
                Model loss.
        """
        output: Gemma3CausalLMOutputWithPast = self.model(
            **inputs,
            return_dict=True,
            **kwargs,
        )
        if reduction == "mean":
            # Note: we need to MINIMIZE loss for the attack
            return output.loss

        # ---------------------------------------------------------------------
        # This is a copy from Gemma3ForConditionalGeneration.forward to support
        # reduction="none"
        loss = None
        labels: Tensor | None = inputs.get("labels")
        attention_mask: Tensor | None = inputs.get("attention_mask")
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = output.logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(
                    logits.device
                )
                shift_logits = shift_logits[
                    shift_attention_mask.to(logits.device) != 0
                ].contiguous()
                shift_labels = shift_labels[
                    shift_attention_mask.to(shift_labels.device) != 0
                ].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()

            vocab_size = shift_logits.shape[-1]
            flat_logits = shift_logits.view(-1, vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)

            # New: Reduction
            indices = shift_attention_mask.sum(1).cumsum(0).cpu().tolist()
            loss_values: list[Tensor] = []
            # Note: `ignore_index`` for CE loss is activate here, so size_average differs
            # from `len(flat_logits)`
            for i in range(len(output.logits)):
                if i == 0:
                    loss_value = loss_fct(
                        flat_logits[: indices[i]], flat_labels[: indices[i]]
                    )
                else:
                    loss_value = loss_fct(
                        flat_logits[indices[i - 1] : indices[i]],
                        flat_labels[indices[i - 1] : indices[i]],
                    )
                loss_values.append(loss_value)
            loss = torch.stack(loss_values)
            if reduction == "mean":
                loss = loss.mean()
            elif reduction == "sum":
                loss = loss.sum()
            elif reduction == "none":
                pass
            else:
                raise ValueError(
                    f"Invalid reduction mode: {reduction}. Expected one of 'none', 'mean', 'sum'."
                )
        # ---------------------------------------------------------------------
        # Note: we need to MINIMIZE loss for the attack
        return loss
