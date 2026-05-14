from .worker import *


class Runner:
    """Manage running state.

    Attributes represent current running state.

    Attributes:
        args (CLIConfig):
            Training arguments.
        logger (Logger):
            Default logging.
        train_logger (Logger):
            Training logging. Suitable for bulk logging.
        device (torch.device):
            Device to use.
    """

    def __init__(self, *, argv=None):
        self.args = parse_args(CLIConfig, args=argv)
        """CLI arguments."""
        self.logger = logging.root
        """Default logger."""
        self.train_logger = logging.root
        """Train logger."""
        self.train_results: list[AttackState] = []
        """Train results dict."""
        self.setup_coupled_args()

    def setup_coupled_args(self):
        """CLI arguments setup."""
        args = self.args
        # Here are some coupled arguments set for convenience
        if args.verify:
            args.debug = True
        if args.debug:
            args.opt_level = 0
            torch.autograd.set_detect_anomaly(True)
        if args.eval_best or args.eval_last:
            args.eval = True
        self.num_workers = args.workers or torch.cuda.device_count()
        assert self.num_workers > 0, "Number of workers should be positive"
        args.evaluation_agent.setdefault("model", args.model)

    def setup_distributed_and_device(self):
        """Setup distributed environment and device.

        We use vllm's TP (tensor parallel) for inference, so device is unused. But we keep
        the code.

        Set attributes `device`, `use_dist_bn`.
        """
        args = self.args
        if args.device == "auto" and not utils.distributed.is_distributed_env():
            args.device = "cuda:0"  # hard code for now
            device = torch.device(args.device)
            args.distributed = False
            args.world_size = 1
            args.rank = 0  # global rank
            args.local_rank = 0
        else:
            # This case should not be reached
            raise UnreachableCodeError
            # This case always adapts to timm device initialization
            if args.device == "auto":
                args.device = "cuda" if torch.cuda.is_available() else "cpu"
            if args.device.isdigit():
                args.device = f"cuda:{args.device}"
            device = utils.init_distributed_device(args)
            # timm also sets: device, world_size, rank, local_rank, distributed
        self.device = device
        utils.cuda_optimization(3)
        self.use_dist_bn = args.dist_bn in ("broadcast", "reduce")

    @property
    def is_dist(self):
        """True if in distributed environment."""
        return self.args.distributed

    @property
    def is_primary_proc(self) -> bool:
        """True if is global primary process."""
        return utils.distributed.is_global_primary(self.args)

    @property
    def is_local_primary_proc(self) -> bool:
        """True if is local primary process."""
        return utils.distributed.is_local_primary(self.args)

    @property
    def rank(self):
        """Global rank."""
        return self.args.rank

    @property
    def world_size(self):
        """World size."""
        return self.args.world_size

    @property
    def is_train(self):
        """True if in training mode.

        Train denotes attack prompt generation, and eval denotes evaluating existing prompts.
        """
        return not self.args.eval

    @property
    def is_train_from_init(self):
        """True if is training from initial, not resuming checkpoint.

        Resume argument is not used.
        """
        return not self.args.eval and self.args.resume < 0

    @property
    def is_eval(self):
        """True if in evaluation mode."""
        return self.args.eval

    @property
    def create_new_path(self):
        """True if a new path will be created."""
        # We create new experiment path for eval since our task focuses on eval
        return self.is_train_from_init or (self.is_eval and self.args.load)

    def setup_auto_lr(self):
        """Setup automatic lr based on distributed environment.

        lr can only be scaled after setting up distributed.
        We scale here to let the lr be recorded in `args.yaml`.
        """
        # Note that the order matters
        # configs.auto_bs(self.args, self.args.world_size)
        # configs.auto_lr(self.args)
        # configs.auto_ema_decay(self.args)
        pass  # reserved

    def setup_experiment_path(self):
        """Setup experiment path.

        Set attribute `expr_path`.

        Raises:
            RuntimeError: when experiment path exists for train, and no --overwrite
            is specified.
        """
        args = self.args
        if self.is_primary_proc:
            # Only primary process is allowed to probe and create experiment path
            expr_path = get_expr_path(args, self.create_new_path)
            if self.create_new_path:
                if not args.overwrite and osp.exists(expr_path):
                    raise RuntimeError(
                        "Experiment path exists, please specify '--overwrite' "
                        "to force overwrite."
                    )
                if not args.verify:
                    # Don't actually make dir for verification
                    os.makedirs(expr_path, exist_ok=True)
        if self.is_dist:
            # Broadcast experiment path to all processes
            obj_list = [expr_path if self.is_primary_proc else None]
            dist.broadcast_object_list(obj_list)
            expr_path = obj_list[0]
        self.expr_path = expr_path

    def setup_random_seed(self):
        """Setup random seed.

        Diffseed for processes applied. `timm` works the same.
        """
        args = self.args
        if args.seed is not None:
            utils.seed_rngs(args.seed + args.rank)

    def _get_log_base_name(self) -> str:
        if self.is_train:
            log_filename = "train"
        else:
            eval_name = "eval"
            if self.is_primary_proc:
                # If we don't `create_new_path` for eval, then eval logs may collide.
                # Avoid that by appending a sequence number.
                if osp.exists(osp.join(self.expr_path, eval_name + ".log")):
                    eval_idx = 1
                    eval_name = f"eval_{eval_idx}"
                    while osp.exists(osp.join(self.expr_path, eval_name + ".log")):
                        eval_idx += 1
                        eval_name = f"eval_{eval_idx}"
            log_filename = eval_name
            if self.is_dist:
                # Broadcast filename to all processes
                obj_list = [log_filename if self.is_primary_proc else None]
                dist.broadcast_object_list(obj_list)
                log_filename = obj_list[0]
        return log_filename

    def setup_logging(self):
        """Initialize a decent logging.

        Set attributes `logger`, `train_logger`.
        """
        args = self.args
        # Suppress transformers warning
        if not self.args.debug:
            transformers.logging.set_verbosity_error()
        if self.is_primary_proc and "file" in args.log and not self.create_new_path:
            assert osp.isdir(self.expr_path), (
                f"Experiment path '{self.expr_path}' does not exist. "
                "Logging to file will fail."
            )
        original_log_dst = args.log
        if args.verify:
            args.log = ["stream"]
        self.log_level = logging.DEBUG if self.args.debug else logging.INFO
        log_filename = self._get_log_base_name()
        if "file" in args.log:
            log_file = osp.join(self.expr_path, log_filename + ".log")
        else:
            log_file = None
        self.log_file = log_file
        utils.setup_default_logging(self.log_level, log_path=log_file)
        self.logger = logging.getLogger(__name__)  # avoid the name 'root'
        self.train_logger = utils.setup_default_logging(
            self.log_level,
            log_path=log_file,
            name="train",
            no_stream="stream" not in self.args.log,
        )
        if self.is_primary_proc:
            self.train_logger.info(args)

        # Setup logging (primary process)
        if self.is_primary_proc:
            self.logger.info(f"Experiment path is '{self.expr_path}'.")
            self.logger.debug("Logging to: {}".format(", ".join(original_log_dst)))
            if "file" in original_log_dst:
                self.logger.info(f"Log name: {log_filename}")
        csv_filename = f"{log_filename}.csv"
        self.csv_path = (
            osp.join(self.expr_path, csv_filename) if "csv" in args.log else None
        )
        if self.is_primary_proc and "wandb" in args.log and self.is_train:
            # Do not expect wandb log
            raise UnreachableCodeError
            if has_wandb:
                assert not args.wandb_resume_id or args.resume, (
                    "Resume id provided without resuming checkpoint"
                )
                wandb_name = None
                if getattr(args, "wandb_name", None):
                    wandb_name = args.wandb_name
                elif args.wandb_tags:
                    wandb_name = "-".join(args.wandb_tags)
                wandb.init(
                    project=args.wandb_project,
                    config=asdict(args),
                    tags=args.wandb_tags,
                    name=wandb_name,
                    resume="must" if args.wandb_resume_id else None,
                    id=args.wandb_resume_id if args.wandb_resume_id else None,
                )
            else:
                self.logger.warning(
                    "You've requested to log metrics to wandb but package not found. "
                    "Metrics not being logged to wandb, try `pip install wandb`"
                )

        mode_prefix = "Training" if self.is_train else "Testing"
        if self.is_dist:
            self.logger.info(
                f"{mode_prefix} in distributed mode with multiple processes, 1 device "
                f"per process. Process {args.rank}, total {args.world_size}, device "
                f"{args.device}."
            )
        else:
            pass

        # Train logger
        # Each worker holds a logger, there is no problem, but we will avoid non-primary
        # logging.
        if self.is_primary_proc:
            dump_file = "args.yaml" if self.is_train else "eval_args.yaml"
            dump_file = osp.join(self.expr_path, dump_file)
            if osp.isfile(dump_file):
                # args.yaml exists, move args.yaml -> args.yaml.0 args.yaml.1 ...
                suff_idx = 0
                while suff_idx < 1024 and osp.exists(f"{dump_file}.{suff_idx}"):
                    suff_idx += 1
                os.rename(dump_file, f"{dump_file}.{suff_idx}")
            if not args.verify:
                configs.dump_args(dump_file, args)
        # less verbosity on openAI logs
        for pkg in ("openai", "httpx", "httpcore", "filelock"):
            logging.getLogger(pkg).setLevel(logging.WARNING)

    def load_dataset(self):
        """Create dataset."""
        args = self.args
        dataset = load_dataset(
            args.dataset, args.data_dir, split=args.dataset_split
        )  ## dataset for the name, data_dir for the root if local
        if self.args.subset > 0:
            # Subset for quickrun
            dataset = dataset.select(range(self.args.subset))
        self.testset = dataset

    def create_model(self):
        """Create agents.

        Set attribute `target_model`.
        """
        pass  # reserved

    def create_dataloader(self):
        """Create dataloader from dataset."""
        pass  # reserved

    def distributed_cleanup(self):
        if self.is_dist:
            # vllm uses multiprocessing
            dist.barrier()
            dist.destroy_process_group()
            self.logger.info(f"Process {self.rank} exit.")

    def parse_checkpoint_path(self):
        """Get checkpoint path based on loading conditions.

        Set attribute `ckpt_path`.
        """
        args = self.args
        self.ckpt_path: str | None = None
        if args.load:
            self.ckpt_path = args.load
        elif args.resume >= 0 or args.eval:
            pkl_path = osp.join(self.expr_path, "train_raw_data.pkl")
            if args.eval_count > 1 and osp.exists(pkl_path):
                self.ckpt_path = pkl_path
            else:
                # Eval count=1 or raw data doesn't exist
                self.ckpt_path = osp.join(self.expr_path, "train.csv")

    def load_checkpoint(self, ckpt_path=None, verbose=False):
        """Load checkpoint for evaluation.

        Args:
            ckpt_path (str, optional):
                checkpoint path override.
        """
        ckpt_path = ckpt_path or self.ckpt_path
        # reserved

    def setup_distributed_model(self):
        """Setup DDP in multi-gpu context.

        Set attributes `model_n`, `model_ema_n`.
        """
        pass  # reserved

    def _create_automated_agent(self, cfg: dict):
        cfg = copy.copy(cfg)
        num_workers = cfg.pop("num_workers", 4)

        def factory():
            return create_automated_llm_agent(
                seed=self.args.seed + 1 if self.args.seed is not None else None,
                **cfg,
            )

        agent = factory()
        if isinstance(agent, AsyncAutomatedLLMAgent):
            # Avoid accidental use of concurrency-unsafe model
            agent = None
        semaphore = asyncio.Semaphore(num_workers)
        return factory, agent, semaphore

    def _create_evaluator(
        self, cfg: dict | list[dict]
    ) -> tuple[BaseEvaluator, asyncio.Semaphore]:
        """Setup evaluation dispatcher."""
        if isinstance(cfg, dict):
            num_workers = cfg.get("model_config", {}).pop("num_workers", 4)
            semaphore = asyncio.Semaphore(num_workers)
            return get_evaluator(**cfg), semaphore
        else:
            num_workers = 1 << 20
            for base_cfg in cfg:
                num_workers = min(
                    num_workers, base_cfg.get("model_config", {}).pop("num_workers", 4)
                )
            semaphore = asyncio.Semaphore(num_workers)
            return EvaluatorEnsemble(cfg), semaphore

    def create_adversary(self):
        """Create attack and evaluation agents."""
        args = self.args
        seed = args.seed if args.seed is not None else None
        cfg = copy.copy(args.evaluation_agent)
        num_workers: int = cfg.pop("num_workers", 4)
        assert num_workers > 0, "Number of evaluation agent workers should be positive"
        self.eval_agent_factory = lambda: create_llm_agent(seed=seed, **cfg)
        self.num_eval_agent_workers = num_workers
        self.eval_agent_semaphore = asyncio.Semaphore(num_workers)
        # Create GCG config for reference
        gcg_cfg_d = copy.copy(self.args.attack)
        gcg_cfg_d.pop("type_")
        self.gcg_config = GCGConfig(**gcg_cfg_d)

        self.evaluator_test, self.evaluator_test_semaphore = self._create_evaluator(
            args.evaluation_test
        )

    @staticmethod
    def _gather_evaluation_for_print(cfg: dict | list[dict]):
        """Print evaluation config."""
        is_multiple = True
        if isinstance(cfg, dict):
            is_multiple = False
            cfg = [cfg]
        lines: list[str] = []
        for c in cfg:
            name: str = c.get("name", "")
            model: str | None = c.get("model_config", {}).get("model")
            if model:
                name = f"{name} (Model: {model})"
            if is_multiple:
                lines.append(f"{len(lines) + 1}. {name}")
            else:
                lines.append(name)
        return lines

    def make_training_briefing(self):
        """A briefing part before training.

        Here train stands for generating adversarial prompt.
        """
        if self.is_primary_proc:
            logger = utils.IndentLogger(log_fn=self.logger.info, indent_unit=" " * 4)
            args = self.args
            logger.log("Attack generation briefing:")
            with logger.indent():
                # Dataset
                if args.subset > 0:
                    dataset_len_log = f"Subset: {len(self.testset)}"
                else:
                    dataset_len_log = f"Length: {len(self.testset)}"
                logger.log(f"Dataset: {args.dataset} ({dataset_len_log})")
                with logger.indent():
                    if args.dataset_split:
                        logger.log(f"Split: {args.dataset_split}")

                name = args.model.rsplit("/", 1)[-1]
                logger.log(f"Model: {name}")
                logger.log(f"Adversary: {args.attack.get('type_', '')}")
                with logger.indent():
                    content = json.dumps(args.attack, indent=len(logger.indent_unit))
                    for i, l in enumerate(content.splitlines()):
                        logger.log(f"Config: {l}" if i == 0 else l)

    def make_testing_briefing(self):
        """A briefing part before testing.

        Here train stands for generating adversarial prompt.
        """
        args = self.args
        if args.eval_count > 1:
            assert self.ckpt_path.endswith(".pkl"), (
                "Raw data is required for evaluation count > 1"
            )
        else:
            assert self.ckpt_path.endswith((".pkl", ".csv")), (
                "Checkpoint should be pickle or csv"
            )
        if self.is_primary_proc:
            logger = utils.IndentLogger(log_fn=self.logger.info, indent_unit=" " * 4)
            logger.log("Attack evaluation briefing:")
            with logger.indent():
                logger.log(f"Evaluation file: {self.ckpt_path}")
                name = args.model.rsplit("/", 1)[-1]
                logger.log(f"Target model: {name}")
                logger.log(f"Eval count: {args.eval_count}")
                logger.log("Judge:")
                with logger.indent():
                    for line in self._gather_evaluation_for_print(args.evaluation_test):
                        logger.log(line)

    @staticmethod
    def _get_meters_to_reduce(meters: list[meter.Meter]):
        return [
            m
            for m in meters
            if isinstance(m, meter.AverageMeter) and "time" not in m.name
        ]

    @staticmethod
    def spawn_worker(worker_id: int, cfg: WorkerConfig):
        Worker(worker_id=worker_id, cfg=cfg).run()

    # -------- Train functions --------
    def train_loop(self):
        """Here train stands for generating adversarial prompt."""
        ctx = mp.get_context("spawn")
        self.in_queue = ctx.Queue()  # mp.Queue[dict | None]
        self.out_queue = ctx.Queue()  # mp.Queue[dict]
        dataset = self.testset

        # launch workers
        workers: list[mp.Process] = []
        worker_cfg = WorkerConfig(
            args=self.args,
            log_level=self.log_level,
            log_path=self.log_file,
            num_workers=self.num_workers,
            in_queue=self.in_queue,
            out_queue=self.out_queue,
        )
        for wid in range(self.num_workers):
            p = ctx.Process(
                target=self.spawn_worker,
                args=(wid, worker_cfg),
            )
            p.start()
            workers.append(p)
        # enqueue all tasks
        for i, data in enumerate(dataset):
            data = data.copy()
            data.setdefault("i", i)
            # state = AttackState.model_validate(data)
            # Pydantic model is much larger than dict
            self.in_queue.put(data)
        # sentinels
        for _ in range(self.num_workers):
            self.in_queue.put(None)
        # wait until all input tasks are done
        for _ in range(len(dataset)):
            state = AttackState.model_validate(self.out_queue.get())
            log_contents = [
                f"============ Case {state.index} ============",
                f"Behavior: {state.prompt}",
                f"Best loss: {state.loss}, best string: {state.adv_suffix}",
                f"LLM response: {state.model_response}",
            ]
            self.train_logger.info("\n".join(log_contents))
            self.train_results.append(state)
        # all tasks done
        for w in workers:
            w.join()

    def finalize_train(self):
        """Things to do even if the train is aborted."""
        self.train_results.sort(key=lambda x: (x.index, x.iter or 0))
        # Note that `dict` is sorted after py3.7, so don't bother with OrderedDict
        results = [
            r.model_dump(exclude={"additional_info"}) for r in self.train_results
        ]
        # Write to csv
        if self.is_primary_proc and self.csv_path is not None and len(results) > 0:
            with open(self.csv_path, "w") as fp:
                raw_data_writer = csv.DictWriter(fp, results[0].keys())
                raw_data_writer.writeheader()
                for item in results:
                    raw_data_writer.writerow(item)
        # Save raw data
        if self.is_primary_proc and self.expr_path is not None and len(results) > 0:
            raw_data_obj: list[dict] = []
            for result in self.train_results:
                if result.additional_info is not None:
                    item = result.additional_info.copy()
                    item["index"] = result.index
                    item["prompt"] = result.prompt
                    item["target"] = result.target
                    raw_data_obj.append(item)
            if len(raw_data_obj) > 0:
                filename = ("eval" if self.is_eval else "train") + "_raw_data.pkl"
                with open(osp.join(self.expr_path, filename), "wb") as fp:
                    pickle.dump(raw_data_obj, fp)

        success_count = 0
        sum_score = 0.0
        sum_loss = 0.0
        for result in self.train_results:
            harmful_score = result.harmful_score or 0.0
            sum_score += harmful_score
            success_count += int(self.args.judge_threshold <= harmful_score)
            sum_loss += result.loss or 0.0
        if len(results) > 0:
            success_count /= len(results)
            sum_score /= len(results)
            sum_loss /= len(results)
        self.train_logger.info(f"Average loss: {sum_loss:.3f}")
        self.train_logger.info(
            f"Average score: {sum_score:.1%} (effective count: {len(results)})"
        )
        self.train_logger.info(f"ASR criterion: {self.args.judge_threshold:.2f}")
        self.train_logger.info(
            f"Attack success rate: {success_count:.1%} (effective count: {len(results)})"
        )

    # -------- Test functions --------
    async def async_action_evaluate_harmfulness(self, state: AttackState):
        """Evaluate harmfulness."""
        assert state.attack_prompt, "Attack prompt should be set"
        assert state.model_response, "Model response should be set"
        evaluator, semaphore = self.evaluator_test, self.evaluator_test_semaphore

        # Only single turn
        query = state.attack_prompt
        response = state.model_response

        # use the dispatcher to evaluate harmfulness, which will call the evaluation agent(s) internally
        try:
            async with semaphore:
                output = await evaluator.evaluate(
                    forbidden_prompt=query, response=response
                )
            state.harmful_score = output.score
            state.harmful_reason = output.reason
            # state.eval_response_raw = output.raw
        except RuntimeError as exc:
            self.train_logger.error(f"[Index {state.index}] Evaluation failed: {exc}")
            state.harmful_score = 0.0
            state.harmful_reason = "EVALUATION FAILED"
            # state.eval_response_raw = ""

    async def async_test_single_behavior(self, state: AttackState):
        """Test: sort with loss. Select first success with lowest loss."""
        agent: AsyncBaseAgent = self.eval_agent_factory()
        gcg_template = self.gcg_config.template
        generate_kwargs = copy.copy(self.args.generate_kwargs)
        if "max_new_tokens" in generate_kwargs:
            generate_kwargs["max_tokens"] = generate_kwargs.pop("max_new_tokens")
        if not generate_kwargs.pop("do_sample", True):
            # generate_kwargs["top_k"] = 1
            generate_kwargs["top_p"] = 1
            generate_kwargs["temperature"] = 0
        if state.additional_info is None:
            # From csv, only one evaluation
            if state.attack_prompt:
                adv_message = state.attack_prompt
            else:
                adv_message = gcg_render_template(
                    gcg_template, dict(prompt=state.prompt, optim_str=state.adv_suffix)
                )
            async with self.eval_agent_semaphore:
                model_response = await agent.ask(adv_message, **generate_kwargs)
            # Set state
            state.attack_prompt = adv_message
            state.model_response = model_response.strip()
            if not state.model_response:
                # In some rare cases like gemini, the response may be empty w/o refusal
                state.model_response = "THE MODEL RESPONDED NOTHING"
            await self.async_action_evaluate_harmfulness(state)
            return state
        else:
            # From json. Sort losses and evaluate one-by-one
            losses: list[float] = state.additional_info["losses"]
            strings: list[str] = state.additional_info["strings"]
            num_iters = self.gcg_config.num_steps
            i_start = max(num_iters - self.args.eval_count, 0)
            if self.args.eval_count > 1:
                candidates = [
                    (losses[i], strings[i], i) for i in range(i_start, num_iters)
                ]
                candidates.sort()
            else:
                candidates = [(losses[i], strings[i], i) for i in range(num_iters)]
                candidates.sort()
                candidates = candidates[:1]
            best_state: Optional[AttackState] = None
            for loss, adv_suffix, iter in candidates:
                adv_message = gcg_render_template(
                    gcg_template, dict(prompt=state.prompt, optim_str=adv_suffix)
                )
                async with self.eval_agent_semaphore:
                    model_response = await agent.new_chat().ask(
                        adv_message, **generate_kwargs
                    )
                state.attack_prompt = adv_message
                state.model_response = model_response.strip()
                if not state.model_response:
                    # In some rare cases like gemma-3, the response may be empty
                    # without refusal
                    state.model_response = "THE MODEL RESPONDED NOTHING"
                await self.async_action_evaluate_harmfulness(state)
                if best_state is None or state.harmful_score > best_state.harmful_score:
                    # Set state
                    state.iter = iter
                    state.adv_suffix = adv_suffix
                    best_state = state.model_copy()
                    if best_state.harmful_score >= self.args.judge_threshold:
                        break
            state = best_state
            # Record lowest loss regardless of eval_count
            state.loss = min(losses[:num_iters])
            return state

    async def async_test_loop(self):
        """Evaluate a set of adversarial prompt."""
        assert self.ckpt_path is not None, "No file to evaluate"
        async with asyncio.TaskGroup() as tg:
            states: list[AttackState] = []
            if self.ckpt_path.endswith(".pkl"):
                # Raw data
                with open(self.ckpt_path, "rb") as fp:
                    raw_data_obj: list[dict] = pickle.load(fp)
                    num_iters = self.gcg_config.num_steps
                    if len(raw_data_obj[0]["losses"]) < num_iters:
                        actual_iters = len(raw_data_obj[0]["losses"])
                        raise RuntimeError(
                            f"Evaluation config requires {num_iters} iters, but the "
                            f"data was trained with {actual_iters} iters"
                        )
                    for item in raw_data_obj:
                        states.append(
                            AttackState(
                                index=item["index"],
                                prompt=item["prompt"],
                                target=item["target"],
                                additional_info=item,
                            )
                        )
            elif self.ckpt_path.endswith(".csv"):
                # Csv data table
                with open(self.ckpt_path, "r") as fp:
                    for row in csv.DictReader(fp):
                        for k, v in row.items():
                            if v == "":
                                # Convert empty string to None
                                row[k] = None
                        states.append(AttackState.model_validate(row))
            else:
                raise ValueError("Checkpoint path should be .pkl or .csv")

            tasks = [
                tg.create_task(self.async_test_single_behavior(state))
                for state in states
            ]
            it = asyncio.as_completed(tasks)  # iterator
            if self.is_primary_proc and "stream" not in self.args.log:
                it = tqdm(it, desc="Evaluating", total=len(tasks))
            finished_cases = 0
            sum_score = 0.0
            sum_loss = 0.0
            for t in it:
                # Not in order
                state = await t
                log_contents = [
                    f"============ Case {state.index} ============",
                    f"Iter: {state.iter}, loss: {state.loss:.4f}, score: {state.harmful_score:.3f}",
                    f"Prompt: {state.attack_prompt}",
                    f"LLM response: {state.model_response}",
                ]
                self.train_logger.info("\n".join(log_contents))
                self.train_results.append(state)
                if isinstance(it, tqdm):
                    finished_cases += 1
                    sum_score += state.harmful_score or 0.0
                    sum_loss += state.loss or 0.0
                    it.set_postfix(
                        {
                            "loss": f"{sum_loss / finished_cases:.3f}",
                            "ASR": f"{sum_score / finished_cases:.1%}",
                        }
                    )

    def _run(self):
        """Main function inside (probable) distributed context."""
        self.setup_auto_lr()
        self.setup_experiment_path()
        self.setup_random_seed()
        self.setup_logging()
        self.load_dataset()
        self.create_model()
        # self.create_dataset_transform()
        self.create_dataloader()

        self.parse_checkpoint_path()
        """Build optimizer and scheduler, and load model checkpoint.

        If we are resuming training, we must (1) build optimizer, (2) resume everything,
        (3) build EMA. If we are just testing, we skip building optimizer, and just init
        the model. This may be a bit confusing, but we follow the pipeline in timm.
        """
        if self.is_train:
            # self.resume_checkpoint()
            pass
        else:
            self.load_checkpoint(verbose=True)

        self.setup_distributed_model()
        self.create_adversary()
        if self.is_train:
            self.make_training_briefing()
            if not self.args.verify:
                try:
                    self.train_loop()
                finally:
                    self.finalize_train()
        else:
            self.make_testing_briefing()
            if not self.args.verify:
                try:
                    asyncio.run(self.async_test_loop())
                finally:
                    # This can be shared
                    self.finalize_train()
        if self.args.verify and self.is_primary_proc:
            self.logger.info("Verification completed!")

    def run(self):
        """Main function."""
        self.setup_distributed_and_device()
        try:
            self._run()
        finally:
            self.distributed_cleanup()
