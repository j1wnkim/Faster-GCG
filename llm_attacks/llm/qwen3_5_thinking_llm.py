from .qwen3_5_llm import *


class Qwen3_5ThinkingLLM(Qwen3_5LLM):
    """Qwen3.5 interface with thinking as default."""

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
        kwargs.setdefault("enable_thinking", True)
        return super().apply_chat_template(
            prompts=prompts,
            targets=targets,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=tokenize,
            **kwargs,
        )
