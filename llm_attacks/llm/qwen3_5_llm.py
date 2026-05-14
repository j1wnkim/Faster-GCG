from .qwen3_llm import *


class Qwen3_5LLM(Qwen3LLM):
    def __init__(
        self, model_id="Qwen/Qwen3.5-4B", model_kwargs=None, processor_kwargs=None
    ):
        super().__init__(
            model_id=model_id,
            model_kwargs=model_kwargs,
            processor_kwargs=processor_kwargs,
        )
        # Explicitly set pad token to EOS token (eliminates the log)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id

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
        # Fix for Qwen3.5
        inputs.pop("mm_token_type_ids")

        # Generate
        with torch.inference_mode():
            generation: torch.LongTensor = self.model.generate(**inputs, **kwargs)
            generation = generation[:, input_len:]

        decoded: list[str] = self.processor.batch_decode(
            generation, skip_special_tokens=True
        )
        return decoded if is_batched else decoded[0]
