import queue
import threading

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .gcg import *
from .gcg import _logger

_draft_model_cache: dict[str, tuple[PreTrainedModel, PreTrainedTokenizer]] = {}


def _get_draft_model(
    name: str, device="auto"
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    if name in _draft_model_cache:
        # Cache hit
        return _draft_model_cache[name]
    draft_model = (
        AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, device_map=device
        )
        .eval()
        .requires_grad_(False)
    )
    draft_tokenizer = AutoTokenizer.from_pretrained(name)
    _draft_model_cache[name] = (draft_model, draft_tokenizer)
    return draft_model, draft_tokenizer


class GCGProbeSampling(GCG):
    """GCG with probe sampling."""

    def __init__(
        self,
        llm: LLM,
        config: GCGConfig | dict | None = None,
    ):
        self.llm = llm
        if config is None:
            self.config = GCGConfig()
        elif isinstance(config, dict):
            self.config = GCGConfig(**config)
        else:
            self.config = config
        _logger.setLevel(getattr(logging, self.config.verbosity))

        tokenizer = self.llm.tokenizer
        self.embedding_layer = self.llm.model.get_input_embeddings()
        if config.allow_non_ascii:
            self.not_allowed_ids = get_nonascii_toks(
                tokenizer, device=self.llm.device, is_ascii=lambda s: s.isprintable()
            )
        else:
            self.not_allowed_ids = get_nonascii_toks(tokenizer, device=self.llm.device)
        self.prefix_cache: Cache | None = None
        self.stop_flag = False

        # Probe sampling
        self.draft_model = None
        self.draft_tokenizer = None
        self.draft_embedding_layer = None
        if self.config.probe_sampling_config:
            self.draft_model = self.config.probe_sampling_config.draft_model
            self.draft_tokenizer = self.config.probe_sampling_config.draft_tokenizer
            self.draft_embedding_layer = self.draft_model.get_input_embeddings()
            if self.draft_tokenizer.pad_token is None:
                configure_pad_token(self.draft_tokenizer)

        if self.llm.dtype in (torch.float32, torch.float64):
            _logger.warning(
                f"Model is in {self.llm.dtype}. Use a lower precision data "
                "type, if possible, for much faster optimization."
            )

        if self.llm.device == torch.device("cpu"):
            _logger.warning(
                "Model is on the CPU. Use a hardware accelerator for faster optimization."
            )

        if not tokenizer.chat_template:
            _logger.warning(
                "Tokenizer does not have a chat template. Assuming base model and setting chat template to empty."
            )
            tokenizer.chat_template = (
                "{% for message in messages %}{{ message['content'] }}{% endfor %}"
            )

    @property
    def model(self):
        return self.llm.model

    @property
    def processor(self):
        return self.llm.processor

    @property
    def tokenizer(self):
        return self.llm.tokenizer

    def sample_ids_from_grad(
        self,
        ids: Tensor,
        grad: Tensor,
        search_width: int,
        topk: int = 256,
        n_replace: int = 1,
        not_allowed_ids: Tensor = False,
        distances=None,
    ):
        """Returns `search_width` combinations of token ids based on the token gradient.

        Args:
            ids : Tensor, shape = (n_optim_ids)
                the sequence of token ids that are being optimized
            grad : Tensor, shape = (n_optim_ids, vocab_size)
                the gradient of the GCG loss computed with respect to the one-hot token embeddings
            search_width : int
                the number of candidate sequences to return
            topk : int
                the topk to be used when sampling from the gradient
            n_replace : int
                the number of token positions to update per sequence
            not_allowed_ids : Tensor, shape = (n_ids)
                the token ids that should not be used in optimization

        Returns:
            sampled_ids : Tensor, shape = (search_width, n_optim_ids)
                sampled token ids
        """
        n_optim_tokens = len(ids)
        original_ids = ids.repeat(search_width, 1)

        if not_allowed_ids is not None:
            grad[:, not_allowed_ids.to(grad.device)] = float("inf")

        topk_grad, topk_ids = (-grad).topk(topk, dim=1)
        # print("Grad:", torch.std_mean(grad[0, topk_ids[0]]))
        if self.config.distance_penalty_option == 2 and distances is not None:
            topk_dist = torch.gather(distances, 1, topk_ids)
            topk_ids = torch.gather(
                topk_ids,
                1,
                torch.argsort(
                    topk_grad + self.config.distance_penalty * topk_dist, dim=1
                ),
            )
            # print("Distance:", torch.std_mean(distances[0, topk_ids[0]]))

        # Faster-GCG - 2: Temperature sampling
        temperature = self.config.temperature
        if not math.isinf(temperature) and self.config.num_steps > 1:
            # Annealing
            temperature = self.config.temperature / math.log2(self.step + 2)
        if math.isinf(temperature):
            # Original GCG: random sampling
            # Random position(s) for each sample in batch
            sampled_ids_pos = (
                torch.rand((search_width, n_optim_tokens), device=grad.device)
                .topk(n_replace, dim=1)
                .indices
            )
            # Random replacement in topk for each selected position
            sampled_ids_val = torch.gather(
                topk_ids[sampled_ids_pos],
                2,
                torch.randint(
                    0, topk, (search_width, n_replace, 1), device=grad.device
                ),
            ).squeeze(2)
        elif temperature < 0.01:
            # 0 temperature: greedy sampling
            assert n_replace == 1
            sampled_ids_pos = torch.arange(n_optim_tokens, device=grad.device)
            # Ceil div
            sampled_ids_pos = sampled_ids_pos.repeat(
                ceil_div(search_width, n_optim_tokens)
            )
            sampled_ids_pos = sampled_ids_pos[:search_width].unsqueeze(1)
            replacement_ind = torch.arange(
                ceil_div(search_width, n_optim_tokens), device=grad.device
            )
            replacement_ind = replacement_ind.repeat_interleave(n_optim_tokens)
            replacement_ind = replacement_ind[:search_width].reshape(-1, 1, 1)
            assert sampled_ids_pos.numel() == search_width
            assert replacement_ind.numel() == search_width
            sampled_ids_val = torch.gather(
                topk_ids[sampled_ids_pos],
                2,
                replacement_ind,
            ).squeeze(2)
        else:
            # Faster-GCG: temperature sampling
            # Random position(s) for each sample in batch
            sampled_ids_pos = (
                torch.rand((search_width, n_optim_tokens), device=grad.device)
                .topk(n_replace, dim=1)
                .indices
            )
            if self.config.temperature_option == 1:
                # Option 1: (k-r)**(1/T)
                probs = torch.arange(topk, 0, -1).to(torch.float32) / topk
                probs = torch.pow(probs, 1 / temperature).to(grad.device)
            else:
                # Option 2: exp(-r/T)
                probs = torch.arange(topk).to(torch.float32)
                probs = (-probs / temperature).softmax(dim=-1).to(grad.device)
            replacement = False
            if not replacement:
                # Sample without replacement
                sampled_ids_pos = sampled_ids_pos.cpu()
                probs = probs.cpu()
                topk_ids = topk_ids.cpu()
                sampled_ids_val = torch.zeros_like(sampled_ids_pos)
                for pos in sampled_ids_pos.unique().tolist():
                    mask = sampled_ids_pos == pos
                    num = mask.sum().item()
                    if num == 0:
                        continue
                    # 对该 pos：无放回采样 num 个 rank
                    sampled_rank = torch.multinomial(
                        probs,
                        num_samples=num,
                        replacement=False,
                    )
                    # 映射到 token id
                    sampled_token = topk_ids[pos, sampled_rank]
                    # 回填（顺序无所谓）
                    sampled_ids_val[mask] = sampled_token
                sampled_ids_pos = sampled_ids_pos.to(original_ids.device)
                sampled_ids_val = sampled_ids_val.to(original_ids.device)
            else:
                distribution = Categorical(probs=probs)
                sampled_ids_val = torch.gather(
                    topk_ids[sampled_ids_pos],
                    2,
                    distribution.sample((search_width, n_replace, 1)),
                ).squeeze(2)

        new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)

        return new_ids

    def filter_ids(self, ids: Tensor, search_width: int):
        """Filters out sequences of token ids that change after re-tokenization.

        Args:
            ids : Tensor, shape = (search_width, n_optim_ids)
                token ids
            tokenizer : ~transformers.PreTrainedTokenizer
                the model's tokenizer

        Returns:
            filtered_ids : Tensor, shape = (new_search_width, n_optim_ids)
                all token ids that are the same after re-tokenization
        """
        tokenizer = self.tokenizer
        # FIXME in some cases, the problem is caused by varying before_ids.
        # It needs to be re-encoded.
        # before_ids = self.before_ids[..., -1:]
        # after_ids = self.after_ids[..., :1]
        # all_ids = torch.cat(
        #     [
        #         before_ids.expand(ids.shape[0], -1),
        #         ids,
        #         after_ids.expand(ids.shape[0], -1),
        #     ],
        #     dim=1,
        # )
        all_ids = ids
        ids_decoded = tokenizer.batch_decode(all_ids)
        filtered_ids = []

        # tuple[int, ...] is hashable
        ids_list: list[tuple[int, ...]] = [tuple(seq) for seq in ids.cpu().tolist()]
        cached_ids: list[Tensor] = []
        cached_losses: list[float] = []
        local_samples_cache: set[tuple[int, ...]] = set()

        for i in range(len(ids_decoded)):
            # Re-tokenize the decoded token ids
            ids_encoded = tokenizer(
                ids_decoded[i], return_tensors="pt", add_special_tokens=False
            ).to(ids.device)["input_ids"][0]
            # Re-tokenization check
            if not torch.equal(all_ids[i], ids_encoded):
                continue

            if self.config.avoid_loop:
                # Faster-GCG - 3: Check for evaluated token sequences
                if ids_list[i] in self.evaluated_seqs:
                    # cached_ids.append(ids[i])
                    # cached_losses.append(self.evaluated_seqs[ids_list[i]])
                    continue
                if ids_list[i] in local_samples_cache:
                    # A re-sample in current batch
                    continue

                # All tests passed, log in local cache
                local_samples_cache.add(ids_list[i])

            filtered_ids.append(ids[i])
            if len(filtered_ids) >= search_width:
                break

        if not filtered_ids:
            # breakpoint()
            # This occurs in some cases, e.g. using the Llama-3 tokenizer with a bad initialization
            raise RuntimeError(
                "No token sequences are the same after decoding and re-encoding. "
                "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
            )

        cached_ids_ = torch.stack(cached_ids) if len(cached_ids) else None
        # print(f"Cache hit {len(cached_losses)}/{len(self.evaluated_seqs)}")

        return torch.stack(filtered_ids), cached_ids_, cached_losses

    def run(
        self,
        prompt: str,
        target: str,
        use_tqdm=True,
    ) -> GCGResult:
        config = self.config
        tokenizer = self.tokenizer
        model = self.model
        self.evaluated_seqs: dict[tuple[int, ...], float] = {}

        # if config.seed is not None:
        #     transformers.set_seed(config.seed)

        if type(tokenizer).__name__.startswith("Gemma"):
            # A small fix for gemma chat template
            target = target.rstrip()

        # Append the GCG string at the end of the prompt if location not specified
        if r"{optim_str}" not in prompt:
            prompt = render_template(self.config.template, dict(prompt=prompt))

        prompt_before, prompt_after = prompt.split(r"{optim_str}")
        # Strip very important here!!! Some tokenizers (e.g. the llama one) are sensitive
        # to leading and trailing whitespaces
        inputs_to_cache: BatchEncoding = self.llm.apply_chat_template(
            [prompt_before.strip()], tokenize=True, continue_final_message=True
        )
        before_ids: torch.LongTensor = inputs_to_cache["input_ids"]
        input_ids: torch.LongTensor = self.llm.apply_chat_template(
            [prompt_before + self.config.optim_str_init],
            tokenize=True,
            continue_final_message=True,
        )["input_ids"]
        l = before_ids.shape[1]
        if not torch.all(before_ids == input_ids[..., :l]):
            warnings.warn("Tokenizer not prefix-preserving in before_ids")
        optim_ids: torch.LongTensor = tokenizer(
            self.config.optim_str_init, return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        if not torch.equal(optim_ids, input_ids[..., l:]):
            warnings.warn(
                "Optim ids not the same with individual and with in-context encoding"
            )
            optim_str_reinit = tokenizer.batch_decode(input_ids[..., l:])[0]
            self.config.optim_str_init = optim_str_reinit
            warnings.warn(f"Re-initialize optim str to '{optim_str_reinit}'")
        prompt_ids: torch.LongTensor = self.llm.apply_chat_template(
            [prompt_before + self.config.optim_str_init + prompt_after],
            tokenize=True,
            add_generation_prompt=True,
        )["input_ids"]
        if not torch.equal(input_ids, prompt_ids[..., : input_ids.shape[1]]):
            warnings.warn("Prompt_after modifies input encoding")
        inputs_to_cache = inputs_to_cache.to(model.device)
        before_ids = inputs_to_cache["input_ids"]
        after_ids: torch.LongTensor = prompt_ids[..., input_ids.shape[1] :].to(
            model.device
        )
        target_ids: torch.LongTensor = tokenizer(
            [target], add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(model.device)

        target = " " + target if config.add_space_before_target else target

        # Embed everything that doesn't get optimized
        embedding_layer = self.embedding_layer
        before_embeds, after_embeds, target_embeds = [
            embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)
        ]

        # Compute the KV Cache for tokens that appear before the optimized tokens
        if config.use_prefix_cache:
            with torch.inference_mode():
                output = self.model(**inputs_to_cache, use_cache=True)
                self.prefix_cache: Cache = output.past_key_values

        self.target_ids = target_ids
        self.before_ids = before_ids
        self.after_ids = after_ids
        # (bs, n_tokens, n_dim) Tensors
        self.before_embeds: Tensor = before_embeds
        self.after_embeds: Tensor = after_embeds
        self.target_embeds: Tensor = target_embeds

        # Initialize components for probe sampling, if enabled.
        self.draft_prefix_cache: Cache | None = None
        if config.probe_sampling_config:
            assert (
                self.draft_model and self.draft_tokenizer and self.draft_embedding_layer
            ), "Draft model wasn't properly set up."

            # Tokenize everything that doesn't get optimized for the draft model
            draft_before_ids = self.draft_tokenizer(
                [prompt_before], padding=False, return_tensors="pt"
            )["input_ids"].to(model.device, torch.int64)
            draft_after_ids = self.draft_tokenizer(
                [prompt_after], add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device, torch.int64)
            self.draft_target_ids = self.draft_tokenizer(
                [target], add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device, torch.int64)

            (
                self.draft_before_embeds,
                self.draft_after_embeds,
                self.draft_target_embeds,
            ) = [
                self.draft_embedding_layer(ids)
                for ids in (
                    draft_before_ids,
                    draft_after_ids,
                    self.draft_target_ids,
                )
            ]

            if config.use_prefix_cache:
                with torch.no_grad():
                    output = self.draft_model(
                        inputs_embeds=self.draft_before_embeds, use_cache=True
                    )
                    self.draft_prefix_cache: Cache = output.past_key_values

        # Initialize the attack buffer
        buffer = self.init_buffer()
        optim_ids = buffer.get_best_ids()

        losses = []
        optim_strings = []
        it = range(config.num_steps)
        if use_tqdm:
            it = tqdm(it)
        for step in it:
            self.step = step
            # Compute the token gradient
            optim_ids_onehot_grad, distances = self.compute_token_gradient(optim_ids)

            cached_ids = None
            cached_losses: list[float] = []

            with torch.no_grad():
                for retry_i in range(5):
                    # Sample candidate token sequences based on the token gradient
                    sampled_ids = self.sample_ids_from_grad(
                        optim_ids.squeeze(0),
                        optim_ids_onehot_grad.squeeze(0),
                        max(64, config.search_width * 2),
                        config.topk,
                        config.n_replace,
                        not_allowed_ids=self.not_allowed_ids,
                        distances=distances,
                    )

                    if config.filter_ids:
                        try:
                            sampled_ids, cached_ids, cached_losses = self.filter_ids(
                                sampled_ids, config.search_width
                            )
                            break
                        except RuntimeError:
                            continue
                    else:
                        break
                else:
                    _logger.warning(
                        "Applicable string not found with 5 re-tries. Fall back to"
                        " previous string."
                    )
                    sampled_ids = buffer.get_best_ids()
                    assert sampled_ids.ndim == 2, "BUG in code"

                new_search_width = sampled_ids.shape[0]

                # Compute loss on all candidate sequences
                batch_size = config.batch_size or new_search_width
                if self.prefix_cache:
                    input_embeds = torch.cat(
                        [
                            embedding_layer(sampled_ids),
                            after_embeds.expand(new_search_width, -1, -1),
                            target_embeds.expand(new_search_width, -1, -1),
                        ],
                        dim=1,
                    )
                else:
                    input_embeds = torch.cat(
                        [
                            before_embeds.expand(new_search_width, -1, -1),
                            embedding_layer(sampled_ids),
                            after_embeds.expand(new_search_width, -1, -1),
                            target_embeds.expand(new_search_width, -1, -1),
                        ],
                        dim=1,
                    )

                if self.config.probe_sampling_config is not None:
                    current_loss, optim_ids = find_executable_batch_size(
                        self._compute_candidates_loss_probe_sampling, batch_size
                    )(
                        input_embeds,
                        sampled_ids,
                    )
                else:
                    raise NotImplementedError

                # Update the buffer based on the loss
                if use_tqdm:
                    it.set_postfix({"Loss": f"{current_loss:.4f}"})
                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            optim_str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)

            buffer.log_buffer(tokenizer)

            if self.stop_flag:
                _logger.info("Early stopping due to finding a perfect match.")
                break

        min_loss_index = losses.index(min(losses))

        result = GCGResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            losses=losses,
            strings=optim_strings,
        )

        # Check tokenize
        optim_ids = tokenizer(
            result.best_string, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(model.device)
        all_token_ids = torch.cat([before_ids, optim_ids, after_ids, target_ids], dim=1)
        # Here begins round-trip
        inputs: BatchEncoding = self.llm.apply_chat_template(
            [render_template(prompt, dict(optim_str=result.best_string))],
            targets=[target],
        ).to(model.device)
        if not torch.equal(inputs["input_ids"], all_token_ids):
            _logger.error(
                "Tokenization check failed. Tokenizer is not behaving consistently."
            )
        # Check loss
        loss = self.llm.get_loss(inputs).item()
        if not math.isclose(loss, result.best_loss, rel_tol=0.05, abs_tol=0.01):
            _logger.error(
                "Loss check failed. Loss is not as expected "
                f"(actual {loss:.2f} vs expected {result.best_loss:.2f})."
            )

        return result

    def _compute_candidates_loss_probe_sampling(
        self,
        search_batch_size: int,
        input_embeds: Tensor,
        sampled_ids: Tensor,
    ) -> Tuple[float, Tensor]:
        """Computes the GCG loss using probe sampling (https://arxiv.org/abs/2403.01251).

        Args:
            search_batch_size : int
                the number of candidate sequences to evaluate in a given batch
            input_embeds : Tensor, shape = (search_width, seq_len, embd_dim)
                the embeddings of the `search_width` candidate sequences to evaluate
            sampled_ids: Tensor, all candidate token id sequences. Used for further sampling.

        Returns:
            A tuple of (min_loss: float, corresponding_sequence: Tensor)

        """
        probe_sampling_config = self.config.probe_sampling_config
        assert probe_sampling_config, "Probe sampling config wasn't set up properly."

        B = input_embeds.shape[0]
        probe_size = B // probe_sampling_config.sampling_factor
        probe_idxs = torch.randperm(B)[:probe_size].to(input_embeds.device)
        probe_embeds = input_embeds[probe_idxs]

        def _compute_probe_losses(
            result_queue: queue.Queue, search_batch_size: int, probe_embeds: Tensor
        ) -> None:
            probe_losses = self._compute_candidates_loss_original(
                search_batch_size, probe_embeds
            )
            result_queue.put(("probe", probe_losses))

        def _compute_draft_losses(
            result_queue: queue.Queue,
            search_batch_size: int,
            draft_sampled_ids: Tensor,
        ) -> None:
            assert self.draft_model and self.draft_embedding_layer, (
                "Draft model and embedding layer weren't initialized properly."
            )

            draft_losses = []
            draft_prefix_cache_batch = None
            for i in range(0, B, search_batch_size):
                with torch.no_grad():
                    batch_size = min(search_batch_size, B - i)
                    draft_sampled_ids_batch = draft_sampled_ids[i : i + batch_size]

                    if self.draft_prefix_cache:
                        draft_embeds = torch.cat(
                            [
                                self.draft_embedding_layer(draft_sampled_ids_batch),
                                self.draft_after_embeds.repeat(batch_size, 1, 1),
                                self.draft_target_embeds.repeat(batch_size, 1, 1),
                            ],
                            dim=1,
                        )
                        # --- CRITICAL: correct positions ---
                        draft_prefix_cache_batch = copy_prefix_cache(
                            self.draft_prefix_cache, self.draft_model.config, batch_size
                        )
                        prefix_len = draft_prefix_cache_batch.get_seq_length()
                        input_len = draft_embeds.shape[1]
                        cache_position = torch.arange(
                            prefix_len, prefix_len + input_len, device=self.model.device
                        )
                        attention_mask = torch.ones(
                            (draft_embeds.shape[0], prefix_len + input_len),
                            device=self.model.device,
                        )
                        draft_output = self.draft_model(
                            inputs_embeds=draft_embeds,
                            attention_mask=attention_mask,
                            past_key_values=draft_prefix_cache_batch,
                            cache_position=cache_position,
                            # use_cache=False is useless. Cache still mutates.
                        )
                    else:
                        draft_embeds = torch.cat(
                            [
                                self.draft_before_embeds.repeat(batch_size, 1, 1),
                                self.draft_embedding_layer(draft_sampled_ids_batch),
                                self.draft_after_embeds.repeat(batch_size, 1, 1),
                                self.draft_target_embeds.repeat(batch_size, 1, 1),
                            ],
                            dim=1,
                        )
                        draft_output = self.draft_model(inputs_embeds=draft_embeds)

                    draft_logits = draft_output.logits
                    tmp = draft_embeds.shape[1] - self.draft_target_ids.shape[1]
                    shift_logits = draft_logits[..., tmp - 1 : -1, :].contiguous()
                    shift_labels = self.draft_target_ids.repeat(batch_size, 1)

                    if self.config.use_mellowmax:
                        label_logits = torch.gather(
                            shift_logits, -1, shift_labels.unsqueeze(-1)
                        ).squeeze(-1)
                        loss = mellowmax(
                            -label_logits, alpha=self.config.mellowmax_alpha, dim=-1
                        )
                    else:
                        loss = (
                            torch.nn.functional.cross_entropy(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1),
                                reduction="none",
                            )
                            .view(batch_size, -1)
                            .mean(dim=-1)
                        )

                    draft_losses.append(loss)

            draft_losses = torch.cat(draft_losses)
            result_queue.put(("draft", draft_losses))

        def _convert_to_draft_tokens(token_ids: Tensor) -> Tensor:
            decoded_text_list = self.tokenizer.batch_decode(token_ids)
            assert self.draft_tokenizer, "Draft tokenizer wasn't properly initialized."
            return self.draft_tokenizer(
                decoded_text_list,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )["input_ids"].to(self.draft_model.device, torch.int64)

        result_queue = queue.Queue()
        draft_sampled_ids = _convert_to_draft_tokens(sampled_ids)

        # Step 1. Compute loss of all candidates using the draft model
        draft_thread = threading.Thread(
            target=_compute_draft_losses,
            args=(result_queue, search_batch_size, draft_sampled_ids),
        )

        # Step 2. In parallel to 1., compute loss of the probe set on target model
        probe_thread = threading.Thread(
            target=_compute_probe_losses,
            args=(result_queue, search_batch_size, probe_embeds),
        )

        draft_thread.start()
        probe_thread.start()

        draft_thread.join()
        probe_thread.join()

        results = {}
        while not result_queue.empty():
            key, value = result_queue.get()
            results[key] = value

        probe_losses = results["probe"]
        draft_losses = results["draft"]

        # Step 3. Calculate agreement score using Spearman correlation
        draft_probe_losses = draft_losses[probe_idxs]
        rank_correlation = spearmanr(
            probe_losses.cpu().type(torch.float32).numpy(),
            draft_probe_losses.cpu().type(torch.float32).numpy(),
        ).correlation
        if not np.isfinite(rank_correlation):
            # Handle NaN case. It often occurs when probe_size<=4.
            warnings.warn(
                f"rank_correlation was {rank_correlation} from {len(probe_losses)} "
                "samples. Set to 0.0."
            )
            rank_correlation = 0.0
        # normalized from [-1, 1] to [0, 1]
        alpha = (1 + rank_correlation) / 2

        # Step 4. Calculate the filtered set and evaluate using the target model.
        R = probe_sampling_config.r
        filtered_size = int((1 - alpha) * B / R)
        filtered_size = max(1, min(filtered_size, B))

        _, top_indices = torch.topk(draft_losses, k=filtered_size, largest=False)

        filtered_embeds = input_embeds[top_indices]
        filtered_losses = self._compute_candidates_loss_original(
            search_batch_size, filtered_embeds
        )

        # Step 5. Return best loss between probe set and filtered set
        best_probe_loss = probe_losses.min().item()
        best_filtered_loss = filtered_losses.min().item()

        probe_ids = sampled_ids[probe_idxs]
        filtered_ids = sampled_ids[top_indices]
        return (
            (best_probe_loss, probe_ids[probe_losses.argmin()].unsqueeze(0))
            if best_probe_loss < best_filtered_loss
            else (
                best_filtered_loss,
                filtered_ids[filtered_losses.argmin()].unsqueeze(0),
            )
        )
