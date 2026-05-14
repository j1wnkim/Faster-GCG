from dataclasses import dataclass, field
from typing import Any, Optional

from transformers.hf_argparser import HfArg


@dataclass
class ModelConfig:
    model: str = HfArg(default="google/gemma-3-4b-it", help="Model name")
    enable_thinking: bool = HfArg(default=False, help="Enable thinking mode of LLM")
    resume: int = HfArg(
        default=-1,
        help="Resume specific epoch. 0 means resuming from the end of first epoch (epoch "
        "0). -1 to disable.",
    )
    load: str = HfArg(default="", help="Load a file")
    generate_kwargs: dict[str, Any] = HfArg(default_factory=dict, help="Generate args")
    tokenize_kwargs: dict[str, Any] = HfArg(default_factory=dict, help="tokenize args")
    # kwargs are for extension usage only
    model_kwargs: dict[str, Any] = field(default_factory=dict, init=False)
