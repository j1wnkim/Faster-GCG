import asyncio
import copy
import csv
import json
import logging
import multiprocessing as mp
import os
import os.path as osp
import pickle
import re
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from typing import Optional, OrderedDict, TypedDict

import torch
import torch.distributed as dist
import transformers
from pydantic import AliasChoices, Field, ValidationError
from pydantic import BaseModel as PydanticModel
from tqdm import tqdm

from llm_attacks import configs, utils
from llm_attacks.agent import (
    AsyncAutomatedLLMAgent,
    AsyncBaseAgent,
    create_automated_llm_agent,
    create_llm_agent,
)
from llm_attacks.configs import parse_args
from llm_attacks.datasets import load_dataset
from llm_attacks.evaluator import BaseEvaluator, EvaluatorEnsemble, get_evaluator
from llm_attacks.gcg import GCG, GCGConfig, create_gcg, is_jailbroken
from llm_attacks.gcg import render_template as gcg_render_template
from llm_attacks.llm import *
from llm_attacks.utils import meter

from .cli_config import CLIConfig
from .expr_path import get_expr_path

try:
    import swanlab as wandb

    has_wandb = True
except ImportError:
    try:
        import wandb

        has_wandb = True
    except ImportError:
        has_wandb = False


class UnreachableCodeError(RuntimeError):
    pass


class AttackState(PydanticModel):
    """Attack state to dump to csv.

    Supported dataset formats:
    - AdvBench
    - JailbreakBench/JBB-Behaviors
    """

    index: int = Field(validation_alias=AliasChoices("index", "Index", "i"))
    """Data index."""
    prompt: str = Field(validation_alias=AliasChoices("prompt", "Goal"))
    """Input prompt."""
    target: str = Field(validation_alias=AliasChoices("target", "Target"))
    """Target string."""
    iter: Optional[int] = None
    """Current attack iteration."""
    adv_suffix: Optional[str] = None
    """Adversarial suffix string."""
    loss: Optional[float] = None
    """Suffix loss."""
    harmful_score: Optional[float] = None
    """1 - jailbroken. 0 - not jailbroken."""
    attack_prompt: Optional[str] = None
    """Rendered attack prompt."""
    model_response: Optional[str] = None
    """Model response to attack prompt."""
    harmful_reason: str | None = None
    """Reason for harmfulness score from judge."""

    # Do not log the following to csv
    additional_info: Optional[dict] = None
    """Additional information or raw data that comes to json only."""


@dataclass
class WorkerConfig:
    """Attack state to dump to csv."""

    args: CLIConfig
    log_level: str
    log_path: str | None
    num_workers: int
    in_queue: mp.Queue  # mp.Queue[dict | None]
    out_queue: mp.Queue  # mp.Queue[dict]


class Worker:
    TARGET_PREFIX = "Sure, here"

    def __init__(self, worker_id: int, cfg: WorkerConfig):
        self.worker_id = worker_id
        self.cfg = cfg
        self.args = cfg.args
        self.in_queue = cfg.in_queue
        self.out_queue = cfg.out_queue

    def setup_logging(self):
        self.logger = utils.setup_default_logging(
            self.cfg.log_level, log_path=self.cfg.log_path
        )
        # less verbosity on openAI logs
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    def setup_random_seed(self):
        """Setup random seed.

        Diffseed for processes applied. `timm` works the same.
        """
        args = self.args
        if args.seed is not None:
            utils.seed_rngs(args.seed + self.worker_id)

    def create_model(self):
        """Create agents.

        Set attribute `target_model`.
        """
        args = self.args
        # FIXME seems like logger is corrupted
        self.logger.info(
            f"Creating model for worker {self.worker_id}/{self.cfg.num_workers}..."
        )
        self.llm = create_llm(args, device_map=f"cuda:{self.worker_id}")

    def create_adversary(self):
        """Create attack and evaluation agents."""
        self.gcg = create_gcg(self.llm, **self.args.attack)

    def run_one_case(self, llm: LLM, gcg: GCG, state: AttackState, use_tqdm=True):
        args = self.args
        message = state.prompt
        target = state.target
        template = gcg_render_template(gcg.config.template, dict(prompt=message))
        result = gcg.run(template, target, use_tqdm=use_tqdm)
        adv_message = gcg_render_template(template, dict(optim_str=result.best_string))
        model_response = llm.generate(
            prompt=adv_message,
            tokenize_kwargs=args.tokenize_kwargs,
            **args.generate_kwargs,
        )
        # Set state
        state.adv_suffix = result.best_string
        state.loss = result.best_loss
        state.model_response = model_response.strip()
        state.harmful_score = float(is_jailbroken(state.model_response))
        state.additional_info = dict(losses=result.losses, strings=result.strings)
        return state

    def train_worker(self):
        llm = self.llm
        gcg = self.gcg
        worker_id = self.worker_id
        while True:
            item: dict | None = self.in_queue.get()
            if item is None:
                break
            state = AttackState.model_validate(item)
            task_id = state.index
            target_replacement = gcg.config.igcg_harmful_guidance
            if target_replacement is not None:
                if state.target.startswith(self.TARGET_PREFIX):
                    state.target = (
                        target_replacement + state.target[len(self.TARGET_PREFIX) :]
                    )
                else:
                    warnings.warn(
                        f"Case {task_id} target does not start with "
                        f"'{self.TARGET_PREFIX}', no replacement is made"
                    )
            if any([_ in self.args.model for _ in ("Qwen3.5", "Llama-3")]):
                state.target += ":"
            else:
                state.target += ":\n"
            try:
                self.logger.info(f"[Worker {worker_id}] starting case {task_id}")
                state = self.run_one_case(llm, gcg, state, use_tqdm=(worker_id == 0))
                self.out_queue.put(state.model_dump())
            except Exception as e:
                traceback.print_exc()
                self.logger.error(f"[Worker {worker_id}] error on case {task_id}: {e}")
        self.logger.info(f"[Worker {worker_id}] exit")

    def run(self):
        self.setup_logging()
        self.setup_random_seed()
        utils.cuda_optimization()
        self.create_model()
        self.create_adversary()
        self.train_worker()
