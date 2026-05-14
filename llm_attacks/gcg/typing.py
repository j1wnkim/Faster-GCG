import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import transformers


@dataclass
class ProbeSamplingConfig:
    draft_model: transformers.PreTrainedModel
    draft_tokenizer: transformers.PreTrainedTokenizer
    r: int = 8
    sampling_factor: int = 16


@dataclass
class GCGConfig:
    num_steps: int = 250
    optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x x"
    template: str = "{prompt}{optim_str}"
    search_width: int = 512
    """the number of candidate sequences to return"""
    batch_size: int = None
    topk: int = 256
    """the topk to be used when sampling from the gradient"""
    n_replace: int = 1
    """the number of token positions to update per sequence"""
    buffer_size: int = 0
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = False
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    seed: int = None
    verbosity: str = "INFO"
    probe_sampling_config: Optional[ProbeSamplingConfig] = None

    # Faster-GCG configs
    distance_penalty: float = 0.0
    """Distance penalty weight. 0 to disable."""
    distance_penalty_option: int = 1
    avoid_loop: bool = False
    """Set to avoid loop."""
    temperature: float = math.inf
    """Sampling temperature.

    - inf for random sampling as the original GCG.
    - 0 for greedy sampling.
    - Any other positive number to set the temperature.
    """
    temperature_option: int = 1

    # I-GCG configs
    igcg_harmful_guidance: str | None = None


@dataclass
class GCGResult:
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]
