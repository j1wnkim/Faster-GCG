from .gemma3_llm import *


class VicunaLLM(Gemma3LLM):
    """Llama2 interface. Typical text model.

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

    DEFAULT_SYSTEM_PROMPT = (
        "A chat between a curious user and an artificial intelligence assistant. The "
        "assistant gives helpful, detailed, and polite answers to the user's questions."
    )

    def __init__(
        self,
        model_id="lmsys/vicuna-7b-v1.5",
        model_kwargs=None,
        processor_kwargs=None,
    ):
        super().__init__(
            model_id=model_id,
            model_kwargs=model_kwargs,
            processor_kwargs=processor_kwargs,
        )

    def _apply_chat_template(
        self,
        conversations: BatchedPromptLike,
        add_generation_prompt=False,
        continue_final_message=False,
        tokenize=True,
        **kwargs,
    ) -> list[str] | BatchEncoding:
        rv: list[str] = []
        for conversation in conversations:
            item = ""
            for message in conversation:
                if message["role"] == "system":
                    item += message["content"]
                elif message["role"] == "user":
                    item += " USER: " + message["content"]
                elif message["role"] in ("model", "assistant"):
                    item += " ASSISTANT: " + message["content"]
            if add_generation_prompt:
                item += " ASSISTANT:"
            rv.append(item)
        if tokenize:
            return self.tokenizer(rv, **kwargs)
        else:
            return rv

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
                return_tensors="pt",
            )
        else:
            tokenize_kwargs = dict(tokenize=tokenize)
        tokenize_kwargs.update(kwargs)
        prompt_tokens: list[str] | Gemma3ModelInput = self._apply_chat_template(
            conversations,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            # padding=True,
            **tokenize_kwargs,
        )
        if targets is None:
            return prompt_tokens

        # 3. Append assistant response and tokenize everything, that's the model input.
        # This 'double tokenization' is used by `trl.SFTTrainer`, so we assume it not
        # very time-consuming.
        for target, conversation in zip(targets, conversations):
            conversation.append({"role": "assistant", "content": target})
        inputs: list[str] | Gemma3ModelInput = self._apply_chat_template(
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
