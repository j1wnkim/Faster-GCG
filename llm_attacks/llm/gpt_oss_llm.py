from .vicuna_llm import *


class GPTOssLLM(VicunaLLM):
    """GPT-OSS interface. Typical thinking/non-thinking text model.

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

    DEFAULT_SYSTEM_PROMPT = None

    def __init__(
        self,
        model_id="openai/gpt-oss-20b",
        model_kwargs=None,
        processor_kwargs=None,
    ):
        super().__init__(
            model_id=model_id,
            model_kwargs=model_kwargs,
            processor_kwargs=processor_kwargs,
        )

    @override
    def _apply_chat_template(
        self,
        conversations,
        add_generation_prompt=False,
        continue_final_message=False,
        tokenize=True,
        **kwargs,
    ):
        # GPT-oss without thinking
        # https://huggingface.co/openai/gpt-oss-20b/discussions/86
        enable_thinking = kwargs.get("enable_thinking", False)
        if enable_thinking or not add_generation_prompt:
            return self.tokenizer.apply_chat_template(
                conversations=conversations,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                tokenize=tokenize,
                **kwargs,
            )
        rv: list[str] = self.tokenizer.apply_chat_template(
            conversations=conversations,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
            **kwargs,
        )
        for i, s in enumerate(rv):
            assert s.endswith(r"<|start|>assistant")
            rv[i] = s + r"<|channel|>analysis<|message|><|end|><|start|>assistant"
        if tokenize:
            return self.tokenizer(rv, **kwargs)
        else:
            return rv
