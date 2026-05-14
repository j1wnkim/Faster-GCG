import ast
import os.path as osp
from dataclasses import dataclass, field
from typing import Any, Optional

from transformers.hf_argparser import HfArg

from llm_attacks.configs.protocol import AdversaryConfig, ModelConfig


# --- Specific types ---
def _keep_recovery_type(s: str):
    if s in ("default", "all", "lr"):
        return s
    else:
        return list(map(int, s.split(",")))


def _parse_kwargs(values: list[str]):
    kw = {}
    for value in values:
        key, value = value.split("=", 1)
        try:
            kw[key] = ast.literal_eval(value)
        except ValueError:
            # fallback to string (avoid need to escape on command line)
            kw[key] = str(value)
    return kw


# ------


@dataclass
class CLIConfig(AdversaryConfig, ModelConfig):
    # Dataset & augmentation
    data_dir: str = HfArg(default="./dataset", help="Dataset root directory")
    dataset: str = HfArg(default="CIFAR10", help="Dataset name")
    dataset_split: Optional[str] = HfArg(default=None, help="Dataset split name")
    subset: int = HfArg(
        default=0,
        help="Randomly select a subset for quicker verification. 0 to disable.",
    )

    # Model
    supported_models: dict[str, dict] = HfArg(
        default_factory=dict,
        help="List all supported models",
    )
    evaluation_agent: dict = HfArg(
        default_factory=dict,
        help="Evaluation agent configurations",
    )
    """Evaluation agent configurations"""
    evaluation_test: list[dict] = HfArg(
        default_factory=list, help="Evaluation protocol for testing"
    )
    """Evaluation protocol for testing"""
    # kwargs are for extension usage only
    model_kwargs_: list[str] = HfArg(
        default_factory=list,
        aliases=["--model_kwargs", "--model-kwargs"],
        help="Additional keyword arguments for model in the form of [KEY=VALUE ...]",
    )
    # generate_kwargs_: list[str] = HfArg(
    #     default_factory=list,
    #     aliases=["--generate_kwargs", "--generate-kwargs"],
    #     help="Additional keyword arguments for model generation in the form of "
    #     "[KEY=VALUE ...]",
    # )
    # tokenize_kwargs_: list[str] = HfArg(
    #     default_factory=list,
    #     aliases=["--tokenize_kwargs", "--tokenize-kwargs"],
    #     help="Additional keyword arguments for model tokenization in the form of "
    #     "[KEY=VALUE ...]",
    # )

    # Device
    dtype: str = HfArg(default="bfloat16", help="Data type (reserved)")
    device: str = HfArg(
        default="auto",
        aliases=["--gpu"],
        help="Device selection, support 0/cuda:0/cpu...",
    )
    dist_bn: str = HfArg(
        default="none",
        help="How to distribute BN",
        metadata=dict(choices=["none", "reduce", "broadcast"]),
    )
    amp: bool = HfArg(default=False, help="Set to True to enable AMP (reserved)")
    torchcompile: Optional[str] = HfArg(
        default=None,
        help="Enable compilation w/ specified backend",
        metadata=dict(nargs="?", const="inductor"),
    )
    torchcompile_mode: Optional[str] = HfArg(default=None, help="`torch.compile` mode")

    # Adversary
    judge_threshold: float = HfArg(
        default=0.8,
        help="Threshold used to decide if response is harmful/attack succeeded",
    )
    attack_kwargs_: list[str] = HfArg(
        default_factory=list,
        aliases=["--attack_kwargs", "--attack-kwargs"],
        help="Additional keyword arguments for attack in the form of [KEY=VALUE ...]",
    )

    # Misc
    ckpt_hist: int = HfArg(default=3, help="Number of checkpoints to keep")
    ckpt_interval: int = HfArg(default=10, help="Checkpoint interval (by epoch)")
    debug: bool = HfArg(default=False, help="Use debug mode and output debug log")
    eval: bool = HfArg(default=False, help="Evaluation mode")
    eval_best: bool = HfArg(default=False, help="Evaluate best checkpoint")
    eval_last: bool = HfArg(default=False, help="Evaluate last checkpoint")
    eval_count: int = HfArg(default=1, help="Evaluation count of GCG")
    expr_path: Optional[str] = HfArg(
        default=None, help="Directly assign experiment path"
    )
    expr_name: Optional[str] = HfArg(
        default=None, aliases=["--expr-suffix", "--suffix"], help="Experiment suffix"
    )
    # The raw argument
    keep_recovery_: str = HfArg(
        default="default",
        aliases=["--keep_recovery", "--keep-recovery"],
        help=(
            "Keep recovery mode: default: keep last 2 recoveries; "
            "all: keep all recoveries; "
            "lr: besides default, keep recoveries before lr change; "
            "comma separated list of int: besides default, keep if epoch match one of list values."
        ),
    )
    # The parsed argument
    keep_recovery: str | list[int] = field(default=None, init=False)
    log: list[str] = HfArg(
        default_factory=lambda: ["stream"],
        help="Log outputs",
        metadata=dict(choices=["file", "csv", "stream", "wandb"]),
    )
    opt_level: int = HfArg(
        default=1,
        aliases="-O",
        help="Program optimization level.",
    )
    output_dir: str = HfArg(default="./output", help="Output root directory")
    overwrite: bool = HfArg(default=False, help="Overwrite existing training directory")
    seed: Optional[int] = HfArg(default=None, help="Set random seed")
    val_interval: int = HfArg(default=1, help="Validation interval (by epoch)")
    verify: bool = HfArg(default=False, help="Verify configurations and exit")
    wandb_project: Optional[str] = HfArg(default=None, help="wandb project name")
    wandb_name: Optional[str] = HfArg(default=None, help="wandb name")
    wandb_tags: Optional[list[str]] = HfArg(default=None, help="wandb tags")
    wandb_resume_id: Optional[str] = HfArg(
        default=None, help="If resuming a run, the id of the run in wandb"
    )
    workers: int = HfArg(default=0, aliases="-j", help="Number of model workers")
    save_raw_data: bool = HfArg(
        default=False, help="Save raw data (e.g., query count) to file"
    )
    comment: Optional[str] = HfArg(default=None, aliases="-m", help="Comment to add")

    # Automatically set
    distributed: bool = field(default=False, init=False)
    """True if in distributed environment. Automatically set."""
    local_rank: int = field(default=0, init=False)
    """Local rank of the process. Automatically set."""
    rank: int = field(default=0, init=False)
    """Global rank of the process. Automatically set."""
    world_size: int = field(default=1, init=False)
    """World size of the environment. Automatically set."""

    def _expand_model_config(self, cfg: dict):
        name = cfg.pop("name", "")
        if name in self.supported_models:
            default_cfg = self.supported_models[name]
            for k, v in default_cfg.items():
                cfg.setdefault(k, v)
        elif name:
            raise KeyError(f"Model {name} not found in supported_models")
        return cfg

    def _expand_evaluator_config_single(self, cfg: dict):
        model_name = cfg.pop("model_name", "")
        if model_name in self.supported_models:
            model_config = cfg.get("model_config", {})
            default_cfg = self.supported_models[model_name]
            for k, v in default_cfg.items():
                model_config.setdefault(k, v)
            cfg["model_config"] = model_config
        elif model_name:
            raise KeyError(f"Model {model_name} not found in supported_models")
        return cfg

    def _expand_evaluator_config(self, cfg: dict | list[dict]):
        if isinstance(cfg, dict):
            self._expand_evaluator_config_single(cfg)
        else:
            for c in cfg:
                self._expand_evaluator_config_single(c)
        return cfg

    def __post_init__(self):
        self.data_dir = osp.expanduser(self.data_dir)
        self.output_dir = osp.expanduser(self.output_dir)
        self.keep_recovery = _keep_recovery_type(self.keep_recovery_)
        self.model_kwargs = _parse_kwargs(self.model_kwargs_)
        AdversaryConfig.__post_init__(self)
        self.attack.update(_parse_kwargs(self.attack_kwargs_))

        self._expand_model_config(self.evaluation_agent)
        self._expand_evaluator_config(self.evaluation_test)
