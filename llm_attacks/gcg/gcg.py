import logging
import warnings
from typing import List, Optional, Tuple, Union

import torch
import transformers
from scipy.stats import spearmanr
from torch import Tensor
from torch.distributions import Categorical
from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedTokenizerBase, ProcessorMixin

from llm_attacks.llm import LLM
from llm_attacks.utils import ceil_div
from llm_attacks.utils.prompt import *

from .cache_utils import Cache, copy_prefix_cache
from .typing import *
from .utils import *

_logger = logging.getLogger(__name__)


class AttackBuffer:
    def __init__(self, size: int):
        self.buffer = []  # elements are (loss: float, optim_ids: Tensor)
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if isinstance(loss, Tensor):
            loss = loss.item()
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        else:
            self.buffer[-1] = (loss, optim_ids)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]

    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]

    def log_buffer(self, tokenizer: ProcessorMixin):
        message = "buffer:"
        for loss, ids in self.buffer:
            optim_str: str = tokenizer.batch_decode(ids)[0]
            optim_str = optim_str.replace("\\", "\\\\")
            optim_str = optim_str.replace("\n", "\\n")
            message += f"\nloss: {loss}" + f" | string: {optim_str}"
        _logger.info(message)


class GCG:
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
        if self.config.probe_sampling_config:
            raise NotImplementedError(
                "GCG with probe sampling uses the class GCGProbeSampling"
            )

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
        # if not math.isinf(temperature) and self.config.num_steps > 1:
        #     # Annealing
        #     temperature = self.config.temperature / math.log2(self.step + 2)
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

    def init_buffer(self) -> AttackBuffer:
        config = self.config
        model = self.model
        tokenizer = self.tokenizer

        _logger.info(f"Initializing attack buffer of size {config.buffer_size}...")

        # Create the attack buffer and initialize the buffer ids
        buffer = AttackBuffer(config.buffer_size)

        if isinstance(config.optim_str_init, str):
            init_optim_ids = tokenizer(
                config.optim_str_init, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device)
            if config.buffer_size > 1:
                init_buffer_ids = (
                    tokenizer(
                        INIT_CHARS, add_special_tokens=False, return_tensors="pt"
                    )["input_ids"]
                    .squeeze()
                    .to(model.device)
                )
                init_indices = torch.randint(
                    0,
                    init_buffer_ids.shape[0],
                    (config.buffer_size - 1, init_optim_ids.shape[1]),
                )
                init_buffer_ids = torch.cat(
                    [init_optim_ids, init_buffer_ids[init_indices]], dim=0
                )
            else:
                init_buffer_ids = init_optim_ids

        else:  # assume list
            if len(config.optim_str_init) != config.buffer_size:
                _logger.warning(
                    f"Using {len(config.optim_str_init)} initializations but buffer size is set to {config.buffer_size}"
                )
            try:
                init_buffer_ids = tokenizer(
                    config.optim_str_init, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(model.device)
            except ValueError:
                _logger.error(
                    "Unable to create buffer. Ensure that all initializations tokenize to the same length."
                )

        true_buffer_size = max(1, config.buffer_size)

        # Compute the loss on the initial buffer entries
        if self.prefix_cache:
            init_buffer_embeds = torch.cat(
                [
                    self.embedding_layer(init_buffer_ids),
                    self.after_embeds.expand(true_buffer_size, -1, -1),
                    self.target_embeds.expand(true_buffer_size, -1, -1),
                ],
                dim=1,
            )
        else:
            init_buffer_embeds = torch.cat(
                [
                    self.before_embeds.expand(true_buffer_size, -1, -1),
                    self.embedding_layer(init_buffer_ids),
                    self.after_embeds.expand(true_buffer_size, -1, -1),
                    self.target_embeds.expand(true_buffer_size, -1, -1),
                ],
                dim=1,
            )

        init_buffer_losses = find_executable_batch_size(
            self._compute_candidates_loss_original, true_buffer_size
        )(init_buffer_embeds)

        # Populate the buffer
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])

        buffer.log_buffer(tokenizer)

        _logger.info("Initialized attack buffer.")

        return buffer

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
        optim_ids_onehot.requires_grad_()

        # (1, num_optim_tokens, vocab_size) @ (vocab_size, embed_dim) -> (1, num_optim_tokens, embed_dim)
        optim_embeds = optim_ids_onehot @ embedding_layer.weight

        if self.prefix_cache:
            input_embeds = torch.cat(
                [optim_embeds, self.after_embeds, self.target_embeds], dim=1
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
                ],
                dim=1,
            )
            output = model(inputs_embeds=input_embeds)

        logits: Tensor = output.logits.float()

        # Shift logits so token n-1 predicts token n
        shift = input_embeds.shape[1] - self.target_ids.shape[1]
        shift_logits = logits[
            ..., shift - 1 : -1, :
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

        optim_ids_onehot_grad = torch.autograd.grad(
            outputs=[loss], inputs=[optim_ids_onehot]
        )[0]
        num_optim_tokens = optim_ids.shape[1]
        # optim_ids_onehot_grad[:, range(num_optim_tokens), optim_ids.squeeze(0)] = float(
        #     "inf"
        # )

        distances = None
        if self.config.distance_penalty > 0:
            # Faster-GCG - 1: Add distance penalty
            with torch.no_grad():
                # (X-Y)^2 = X^2 + Y^2 - 2XY
                if not hasattr(self, "all_tokens_squared_norm"):
                    # (vocab_size,). Compute and cache it
                    self.all_tokens_squared_norm = (embedding_layer.weight**2).sum(1)
                # (num_optim_tokens,)
                optim_ids_squared_norm = self.all_tokens_squared_norm[
                    optim_ids.squeeze(0)
                ]
                # (num_optim_tokens, embed_dim)
                optim_ids_embedding = embedding_layer.weight[optim_ids.squeeze(0)]
                # (num_optim_tokens, vocab_size)
                distances = (
                    optim_ids_squared_norm[:, None]
                    + self.all_tokens_squared_norm[None, :]
                    - 2 * optim_ids_embedding @ embedding_layer.weight.T
                )
                distances = distances.sqrt()
                # This is meaningless. But you need to tune distance_penalty after
                # deleting this line.
                distances = distances / distances.norm(dim=-1, keepdim=True).clamp_min(
                    1e-6
                )
                # This is meaningless. But you need to tune distance_penalty after
                # deleting this line.
                optim_ids_onehot_grad = (
                    optim_ids_onehot_grad
                    / optim_ids_onehot_grad.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                )
                # Process diagonal. Give no-change the largest penalty
                # print("Grad:", torch.std_mean(optim_ids_onehot_grad))
                # print("Dist:", torch.std_mean(distances))
                if self.config.distance_penalty_option == 1:
                    # Old distance penalty
                    distances[range(num_optim_tokens), optim_ids.squeeze(0)] = (
                        distances.max()
                    )
                    optim_ids_onehot_grad = (
                        optim_ids_onehot_grad + self.config.distance_penalty * distances
                    )

        return optim_ids_onehot_grad, distances

    def _compute_candidates_loss_original(
        self,
        search_batch_size: int,
        input_embeds: Tensor,
    ) -> Tensor:
        """Computes the GCG loss on all candidate token id sequences.

        Args:
            search_batch_size : int
                the number of candidate sequences to evaluate in a given batch
            input_embeds : Tensor, shape = (search_width, seq_len, embed_dim)
                the embeddings of the `search_width` candidate sequences to evaluate
        """
        all_loss = []

        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i : i + search_batch_size]
                current_batch_size = input_embeds_batch.shape[0]

                if self.prefix_cache:
                    # --- CRITICAL: correct positions ---
                    prefix_cache_batch = copy_prefix_cache(
                        self.prefix_cache, self.model.config, current_batch_size
                    )
                    prefix_len = prefix_cache_batch.get_seq_length()
                    input_len = input_embeds_batch.shape[1]
                    cache_position = torch.arange(
                        prefix_len, prefix_len + input_len, device=self.model.device
                    )
                    attention_mask = torch.ones(
                        (input_embeds_batch.shape[0], prefix_len + input_len),
                        device=self.model.device,
                    )
                    outputs = self.model(
                        inputs_embeds=input_embeds_batch,
                        attention_mask=attention_mask,
                        past_key_values=prefix_cache_batch,
                        cache_position=cache_position,
                        # use_cache=False is useless. Cache still mutates.
                    )
                else:
                    outputs = self.model(inputs_embeds=input_embeds_batch)

                logits = outputs.logits.float()
                del outputs  # reduce VRAM usage

                tmp = input_embeds.shape[1] - self.target_ids.shape[1]
                shift_logits = logits[..., tmp - 1 : -1, :].contiguous()
                shift_labels = self.target_ids.repeat(current_batch_size, 1)

                if self.config.use_mellowmax:
                    label_logits = torch.gather(
                        shift_logits, -1, shift_labels.unsqueeze(-1)
                    ).squeeze(-1)
                    loss = mellowmax(
                        -label_logits, alpha=self.config.mellowmax_alpha, dim=-1
                    )
                else:
                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        reduction="none",
                    )

                loss = loss.view(current_batch_size, -1).mean(dim=-1)
                all_loss.append(loss)

                if self.config.early_stop:
                    if torch.any(
                        torch.all(
                            torch.argmax(shift_logits, dim=-1) == shift_labels, dim=-1
                        )
                    ).item():
                        self.stop_flag = True

        return torch.cat(all_loss, dim=0)


# A wrapper around the GCG `run` method that provides a simple API
def run(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: Union[str, List[dict]],
    target: str,
    config: Optional[GCGConfig] = None,
) -> GCGResult:
    """Generates a single optimized string using GCG.

    Args:
        model: The model to use for optimization.
        tokenizer: The model's tokenizer.
        messages: The conversation to use for optimization.
        target: The target generation.
        config: The GCG configuration to use.

    Returns:
        A GCGResult object that contains losses and the optimized strings.
    """
    raise NotImplementedError
    # TODO adapt to new API
    gcg = GCG(model, tokenizer, config or GCGConfig())
    result = gcg.run(messages, target)
    return result
