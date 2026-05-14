from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from transformers.hf_argparser import HfArg


@dataclass
class AdversaryConfig:
    attack: dict[str, Any] = HfArg(default_factory=dict, help="Attack args")
    attack_init: str = HfArg(
        default="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        help="Attack initialization string",
    )
    gcg_search_width: int = HfArg(default=512, help="GCG search width")
    gcg_num_steps: int = HfArg(default=500, help="GCG num steps")
    gcg_topk: int = HfArg(default=256, help="GCG topk")
    gcg_template: str = HfArg(default=r"{prompt} {optim_str}", help="GCG template")
    # Faster-GCG configs
    faster_gcg_distance_penalty: float = HfArg(
        default=0.0, help="Faster-GCG distance penalty weight"
    )
    faster_gcg_avoid_loop: bool = HfArg(default=False, help="Faster-GCG avoid loop")
    faster_gcg_temperature: float = HfArg(
        default=float("inf"), help="Faster-GCG sampling temperature"
    )
    igcg_harmful_guidance: Optional[str] = HfArg(
        default=None,
        help="I-GCG harmful guidance",
    )

    def __post_init__(self):
        self.attack.update(
            optim_str_init=self.attack_init,
            search_width=self.gcg_search_width,
            num_steps=self.gcg_num_steps,
            topk=self.gcg_topk,
            template=self.gcg_template,
            distance_penalty=self.faster_gcg_distance_penalty,
            avoid_loop=self.faster_gcg_avoid_loop,
            temperature=self.faster_gcg_temperature,
            igcg_harmful_guidance=self.igcg_harmful_guidance,
        )
