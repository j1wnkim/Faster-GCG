from .gemma3_llm import *


class Qwen3LLM(Gemma3LLM):
    """Qwen3 interface. Typical thinking/non-thinking text model.

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
        model_id="Qwen/Qwen3-8B",
        model_kwargs=None,
        processor_kwargs=None,
    ):
        super().__init__(
            model_id=model_id,
            model_kwargs=model_kwargs,
            processor_kwargs=processor_kwargs,
        )

    @override
    def apply_chat_template(
        self,
        prompts,
        targets=None,
        add_generation_prompt=False,
        continue_final_message=False,
        tokenize=True,
        **kwargs,
    ):
        kwargs.setdefault("enable_thinking", False)
        return super().apply_chat_template(
            prompts=prompts,
            targets=targets,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=tokenize,
            **kwargs,
        )
