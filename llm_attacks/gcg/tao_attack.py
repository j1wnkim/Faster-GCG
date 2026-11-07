from typing import override

from .gcg import *
from .gcg import _logger

REFUSAL_PREFIXES = [
    "I cannot",
    "I'm sorry",
    "As an AI",
]


class TAOAttack(GCG):
    """Implements TAO-Attack.

    Ref: https://arxiv.org/abs/2603.03081
    """

    @override
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
        if distances is not None:
            # Unnormalized grad in TAO attack
            topk_dist = torch.gather(distances, 1, topk_ids)

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
            sampled_ids_pos = torch.arange(
                n_optim_tokens**n_replace, device=grad.device
            )
            # Ceil div
            sampled_ids_pos = sampled_ids_pos.repeat(
                ceil_div(search_width, len(sampled_ids_pos))
            )
            sampled_ids_pos = sampled_ids_pos[:search_width].unsqueeze(1)
            # Option 2: exp(-r/T)
            probs = (-topk_dist / temperature).softmax(dim=-1).to(grad.device)
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
                        probs[pos],
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

    @override
    def compute_token_gradient(
        self,
        optim_ids: Tensor,
    ) -> Tensor:
        """Computes the gradient of the GCG loss w.r.t the one-hot token matrix.

        Args:
            optim_ids : Tensor, shape = (1, n_optim_ids)
                the sequence of token ids that are being optimized
        """
        model = self.model
        embedding_layer = self.embedding_layer

        # Create the one-hot encoding matrix of our optimized token ids
        # (num_optim_tokens, vocab_size)
        optim_ids_onehot = torch.nn.functional.one_hot(
            optim_ids, num_classes=embedding_layer.num_embeddings
        )
        optim_ids_onehot = optim_ids_onehot.to(model.device, model.dtype)
        # optim_ids_onehot.requires_grad_()

        # (1, num_optim_tokens, vocab_size) @ (vocab_size, embed_dim) -> (1, num_optim_tokens, embed_dim)
        optim_embeds = optim_ids_onehot @ embedding_layer.weight
        optim_embeds.requires_grad_()

        # TAO attack -- stage 1: refusal-aware loss
        refusal_str = REFUSAL_PREFIXES[self.current_refusal_index]
        refusal_ids: torch.LongTensor = self.tokenizer(
            [refusal_str], add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(model.device)
        refusal_embeds = embedding_layer(refusal_ids)

        if self.prefix_cache:
            input_embeds = torch.cat(
                [optim_embeds, self.after_embeds, self.target_embeds, refusal_embeds],
                dim=1,
            )
            # --- CRITICAL: correct positions ---
            prefix_cache = copy_prefix_cache(self.prefix_cache, self.model.config)
            prefix_len = prefix_cache.get_seq_length()
            input_len = input_embeds.shape[1]
            cache_position = torch.arange(
                prefix_len, prefix_len + input_len, device=model.device
            )
            position_ids = cache_position.unsqueeze(0).expand(input_embeds.shape[0], -1)
            attention_mask = torch.ones(
                (input_embeds.shape[0], prefix_len + input_len), device=model.device
            )
            output = model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                past_key_values=prefix_cache,
                cache_position=cache_position,
                position_ids=position_ids,
                # use_cache=False is useless. Cache still mutates.
            )
        else:
            input_embeds = torch.cat(
                [
                    self.before_embeds,
                    optim_embeds,
                    self.after_embeds,
                    self.target_embeds,
                    refusal_embeds,
                ],
                dim=1,
            )
            output = model(inputs_embeds=input_embeds)

        logits: Tensor = output.logits.float()

        # Shift logits so token n-1 predicts token n
        # part 1: loss on target
        shift = input_embeds.shape[1] - self.target_ids.shape[1] - refusal_ids.shape[1]
        shift_begin = shift - 1
        shift_end = shift_begin + self.target_ids.shape[1]
        shift_logits = logits[
            ..., shift_begin:shift_end, :
        ].contiguous()  # (1, num_target_ids, vocab_size)
        shift_labels = self.target_ids

        if self.config.use_mellowmax:
            label_logits = torch.gather(
                shift_logits, -1, shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
        else:
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
        loss_target = loss

        # TAO attack -- stage 1: refusal-aware loss
        # part 2: loss on refusal
        shift = input_embeds.shape[1] - refusal_ids.shape[1]
        shift_logits = logits[..., shift - 1 : -1, :].contiguous()
        shift_labels = refusal_ids

        if self.config.use_mellowmax:
            label_logits = torch.gather(
                shift_logits, -1, shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
        else:
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
        loss_refusal = loss

        tao_alpha = 0.2
        loss = loss_target - tao_alpha * loss_refusal

        optim_embeds_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_embeds])[
            0
        ]
        num_optim_tokens = optim_ids.shape[1]

        tao_w = 5
        tao_mu = 1.5e-3
        current_loss_item = loss.detach().item()
        if len(self.tao_past_losses) >= tao_w:
            past_loss_mean = sum(self.tao_past_losses) / len(self.tao_past_losses)
            self.tao_past_losses.append(current_loss_item)
            self.tao_past_losses = self.tao_past_losses[-tao_w:]
            current_loss_mean = sum(self.tao_past_losses) / len(self.tao_past_losses)
            if abs(current_loss_mean - past_loss_mean) < tao_mu:
                self.current_refusal_index = (self.current_refusal_index + 1) % len(
                    REFUSAL_PREFIXES
                )
        else:
            self.tao_past_losses.append(current_loss_item)

        with torch.no_grad():
            # TAO attack -- DPTO
            # (num_optim_tokens, embed_dim)
            optim_ids_embedding = embedding_layer.weight[optim_ids.squeeze(0)]
            # (num_optim_tokens, vocab_size, embed_dim)
            # optim_embeds_grad: (1, num_optim_tokens, embed_dim)
            chunks: list[Tensor] = []
            norm_chunks: list[Tensor] = []
            for j in range(num_optim_tokens):
                # (vocab_size, embed_dim)
                sub_distance = embedding_layer.weight - optim_ids_embedding[j, :]
                chunk = (optim_embeds_grad[:, j : j + 1, None, :] * sub_distance).sum(
                    dim=-1
                )
                chunks.append(chunk)
                norm_chunks.append(sub_distance.norm(dim=-1))
            # (1, num_optim_tokens, vocab_size)
            unnormalized_onehot_grad = torch.cat(chunks, dim=1)
            # (1, num_optim_tokens, vocab_size)
            distances = torch.stack(norm_chunks, dim=0).unsqueeze(0)
            optim_ids_onehot_grad = unnormalized_onehot_grad / (
                optim_embeds_grad.norm(dim=-1, keepdim=True) * distances
            ).clamp_min(1e-6)

        return optim_ids_onehot_grad, unnormalized_onehot_grad.squeeze(0)

    @override
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

        # TAO attack -- stage 1: refusal-aware loss
        self.current_refusal_index = 0
        self.tao_past_losses: list[float] = []

        self.target_ids = target_ids
        self.before_ids = before_ids
        self.after_ids = after_ids
        # (bs, n_tokens, n_dim) Tensors
        self.before_embeds: Tensor = before_embeds
        self.after_embeds: Tensor = after_embeds
        self.target_embeds: Tensor = target_embeds

        # Initialize components for probe sampling, if enabled.
        if config.probe_sampling_config:
            raise NotImplementedError

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

                if self.config.probe_sampling_config is None:
                    loss = find_executable_batch_size(
                        self._compute_candidates_loss_original, batch_size
                    )(input_embeds)
                    all_losses_list: list[float] = loss.cpu().tolist()

                    # Faster-GCG - 3: Cache evaluated token sequences
                    if self.config.avoid_loop:
                        sampled_ids_list: list[tuple[int, ...]] = [
                            tuple(ids) for ids in sampled_ids.cpu().tolist()
                        ]
                        for ids, l in zip(sampled_ids_list, all_losses_list):
                            self.evaluated_seqs[ids] = l
                        # Selected cached items are also candidates
                        if cached_ids is not None:
                            sampled_ids = torch.cat([sampled_ids, cached_ids], dim=0)
                        all_losses_list += cached_losses

                    loss_with_idx = [(l, i) for (i, l) in enumerate(all_losses_list)]
                    loss_with_idx.sort()
                    min_loss, min_loss_idx = loss_with_idx[0]
                    current_loss = min_loss
                    optim_ids = sampled_ids[min_loss_idx].unsqueeze(0)
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
