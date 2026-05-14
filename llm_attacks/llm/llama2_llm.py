from .gemma3_llm import *


class Llama2LLM(Gemma3LLM):
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

    DEFAULT_SYSTEM_PROMPT = None
    MASK_TOKEN = -100  # mask out labels

    def __init__(
        self,
        model_id="meta-llama/Llama-2-7b-chat-hf",
        model_kwargs=None,
        processor_kwargs=None,
    ):
        super().__init__(
            model_id=model_id,
            model_kwargs=model_kwargs,
            processor_kwargs=processor_kwargs,
        )
