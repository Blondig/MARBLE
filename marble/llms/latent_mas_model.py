"""
Latent (white-box) model wrapper for the LatentMAS communication baseline.

This is vendored, nearly verbatim, from the official LatentMAS implementation
(``models.py`` in https://github.com/Gen-Verse/LatentMAS, arXiv:2511.20639), so
that MARBLE's latent baseline reuses the paper's exact latent-reasoning and
latent-space-realignment mechanics rather than re-deriving them.

Trimmed relative to the upstream file:
* Only the Hugging Face (``transformers``) path is kept -- the vLLM backend and
  the second-HF-model / hidden-state-export helpers used by ``run_batch_vllm``
  are dropped (we use the ``run_batch`` / HF path).
* The ``args`` coupling for the realignment flag is replaced by an explicit
  ``latent_space_realign`` constructor argument.
* Unused imports (matplotlib, csv) are removed.

Core methods reused unchanged:
* :meth:`ModelWrapper.generate_latent_batch` -- Coconut-style latent reasoning
  that feeds the (realigned) last hidden state back as the next input embedding
  and returns the accumulated KV cache (the agent's latent working memory).
* :meth:`ModelWrapper.generate_text_batch` -- decode text conditioned on an
  (optional) ``past_key_values`` latent cache (used by the final "judger" agent).

torch/transformers are imported at module load, so this module is only imported
lazily by the engine when ``coordinate_mode: latent`` is configured.
"""

from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.cache_utils import Cache
except ImportError:  # pragma: no cover - older transformers
    Cache = None


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def _past_length(past_key_values: Optional[Tuple]) -> int:
    if not past_key_values:
        return 0
    k = past_key_values[0][0]
    return k.shape[-2]


def _slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
    if tokens_to_keep <= 0:
        return tensor[..., 0:0, :].contiguous()
    keep = min(tokens_to_keep, tensor.shape[-2])
    start = tensor.shape[-2] - keep
    return tensor[..., start:, :].contiguous()


def truncate_past(past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
    """Keep only the most recent ``tokens_to_keep`` positions of a KV cache.

    Mirrors ``LatentMASMethod._truncate_past`` -- used to bound the latent
    working memory that is threaded down the agent chain.
    """
    if past_kv is None or tokens_to_keep <= 0:
        return None
    if Cache is not None and isinstance(past_kv, Cache):
        legacy = past_kv.to_legacy_cache()
        trimmed_legacy = tuple(
            tuple(_slice_tensor(t, tokens_to_keep) for t in layer) for layer in legacy
        )
        return past_kv.__class__.from_legacy_cache(trimmed_legacy)
    trimmed_layers = []
    for layer in past_kv:
        if isinstance(layer, tuple):
            trimmed_layers.append(
                tuple(_slice_tensor(t, tokens_to_keep) for t in layer)
            )
        elif torch.is_tensor(layer):
            trimmed_layers.append(_slice_tensor(layer, tokens_to_keep))
        else:
            trimmed_layers.append(layer)
    return tuple(trimmed_layers)


class ModelWrapper:
    """Shared local causal LM exposing latent reasoning and KV-conditioned decode."""

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        latent_space_realign: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.latent_space_realign = bool(latent_space_realign)
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        # for ablation parity with upstream
        self.pre_aligned = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _ensure_pad_token(self.tokenizer)
        with torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=(
                    torch.bfloat16 if torch.cuda.is_available() else torch.float32
                ),
            )
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(device)
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        if self.latent_space_realign:
            self._ensure_latent_realign_matrix(self.model, self.device)

    def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            try:
                # Qwen3 et al. default to a <think> reasoning block, which eats
                # the decode budget and breaks downstream JSON parsing. The
                # latent reasoning already happens in KV space, so disable it.
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    enable_thinking=False,
                )
            except TypeError:
                # Template does not accept enable_thinking (non-Qwen3 models).
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=add_generation_prompt
                )
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_chat_input(
        self, messages: List[Dict], add_generation_prompt: bool = True
    ) -> Tuple[str, torch.Tensor, torch.Tensor, List[str]]:
        prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        active_ids = input_ids[0][attention_mask[0].bool()].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(active_ids)
        return prompt_text, input_ids, attention_mask, tokens

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        prompts: List[str] = []
        for messages in batch_messages:
            prompts.append(
                self.render_chat(messages, add_generation_prompt=add_generation_prompt)
            )
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        tokens_batch: List[List[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            active_ids = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))
        return prompts, input_ids, attention_mask, tokens_batch

    # ---- latent-space realignment (training-free) ---------------------- #
    def _build_latent_realign_matrix(
        self, model, device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_embeds = (
            model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        )
        output_embeds = (
            model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        )
        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)
        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError(
                "Cannot build latent realignment matrix: embedding weights not accessible."
            )
        input_weight = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
        output_weight = output_embeds.weight.detach().to(device=device, dtype=torch.float32)
        gram = torch.matmul(output_weight.T, output_weight)
        reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram = gram + reg
        rhs = torch.matmul(output_weight.T, input_weight)
        realign_matrix = torch.linalg.solve(gram, rhs)
        target_norm = input_weight.norm(dim=1).mean().detach()

        if not self.latent_space_realign:
            # keep an identity map; we still renormalize to the target norm.
            realign_matrix = torch.eye(
                realign_matrix.shape[0],
                device=realign_matrix.device,
                dtype=realign_matrix.dtype,
            )
        return realign_matrix, target_norm

    def _ensure_latent_realign_matrix(
        self, model, device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = id(model)
        info = self._latent_realign_matrices.get(key)
        target_device = torch.device(device)

        if info is None:
            matrix, target_norm = self._build_latent_realign_matrix(model, target_device)
        else:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)

        target_norm = (
            target_norm.to(device=target_device, dtype=matrix.dtype)
            if isinstance(target_norm, torch.Tensor)
            else torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)
        )
        self._latent_realign_matrices[key] = (matrix, target_norm)
        return matrix, target_norm

    def _apply_latent_realignment(
        self, hidden: torch.Tensor, model: torch.nn.Module
    ) -> torch.Tensor:
        matrix, target_norm = self._ensure_latent_realign_matrix(model, hidden.device)
        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix)

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pre_aligned = aligned.detach().clone()
        self.pre_aligned = pre_aligned
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)

    # ---- generation --------------------------------------------------- #
    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[List[str], Optional[Tuple]]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        cache_position = None
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            cache_position = torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        sequences = outputs.sequences
        generations: List[str] = []
        for idx, length in enumerate(prompt_lengths):
            length = int(length)
            generated_ids = sequences[idx, length:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generations.append(text)
        return generations, outputs.past_key_values

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)

    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values

        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, D]

        for step in range(latent_steps):
            latent_vec = self._apply_latent_realignment(last_hidden, self.model)
            latent_embed = latent_vec.unsqueeze(1)

            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=self.device,
            )
            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

        return past

    # ------------------------------------------------------------------ #
    # Latent-MAS additions (graph-latent building blocks).
    # Principle: CONTENT travels as KV; CONTROL decisions are a tiny decoded
    # true/false. These are NOT in upstream LatentMAS (fixed pipeline); they are
    # what graph-latent (dynamic rounds + agent-agent dialogue) needs.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_legacy(past: Optional[object]) -> Optional[Tuple]:
        """Normalize a KV cache (Cache or legacy tuple) to a legacy tuple."""
        if past is None:
            return None
        if hasattr(past, "to_legacy_cache"):
            return past.to_legacy_cache()  # type: ignore[no-any-return]
        return past  # type: ignore[return-value]

    def merge_kv(self, kv_list: List[Optional[Tuple]]) -> Optional[Tuple]:
        """
        Concatenate several KV caches along the sequence dimension (multi-source
        fan-in for graph rounds: pool all agents' latent working memory).

        NOTE (known approximation): the 2nd+ source keeps the RoPE positions it
        was computed at, i.e. positions are not re-based after concatenation.
        Acceptable for training-free latent collaboration; documented honestly.
        """
        caches = [self._as_legacy(kv) for kv in kv_list if kv is not None]
        if not caches:
            return None
        if len(caches) == 1:
            return caches[0]
        n_layers = len(caches[0])
        merged: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer in range(n_layers):
            keys = torch.cat([kv[layer][0] for kv in caches], dim=2)
            vals = torch.cat([kv[layer][1] for kv in caches], dim=2)
            merged.append((keys, vals))
        return tuple(merged)

    @torch.no_grad()
    def decode_bool(
        self,
        past_key_values: Optional[Tuple],
        question: str,
        *,
        max_new_tokens: int = 6,
    ) -> bool:
        """
        Decode a short plaintext true/false CONTROL signal conditioned on a KV
        working memory, without decoding full content (greedy, a few tokens).

        Used for control decisions (e.g. "should this dialogue stop?", "is the
        task complete?"). Operates on a legacy-tuple view so the caller's KV is
        not mutated by the generate() append.
        """
        kv = self._as_legacy(past_key_values)
        msgs = [
            {
                "role": "user",
                "content": f"{question} Reply with only 'true' or 'false'.",
            }
        ]
        prompt = self.render_chat(msgs, add_generation_prompt=True)
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        cache_position = None
        if kv is not None:
            past_len = _past_length(kv)
            cache_position = torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
            if past_len > 0:
                past_mask = torch.ones(
                    (1, past_len), dtype=attention_mask.dtype, device=self.device
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            past_key_values=kv,
            cache_position=cache_position,
        )
        gen = out[0, input_ids.shape[-1] :]
        ans = self.tokenizer.decode(gen, skip_special_tokens=True).strip().lower()
        return ans.startswith("true") or ans.startswith("yes")

    @torch.no_grad()
    def latent_dialogue(
        self,
        initiator_msgs: List[Dict],
        responder_msgs: List[Dict],
        *,
        latent_steps: int,
        max_turns: int = 5,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        """
        Latent side-dialogue between two agents (Part A).

        Mirrors the structure of BaseAgent._handle_new_communication_session
        (<=max_turns, alternating speakers, early stop), but: content travels as
        KV (no text decoded between agents), and the stop condition is a tiny
        decoded true/false (decode_bool) instead of the text "<end-of-session>".
        Returns the accumulated KV (the dialogue's latent outcome); no text
        summary is produced -- the KV itself is the takeaway.
        """
        kv = self._as_legacy(past_key_values)
        # Responder speaks first, mirroring the original (target agent answers).
        roles = [responder_msgs, initiator_msgs]
        for t in range(max_turns):
            msgs = roles[t % 2]
            _, input_ids, attention_mask, _ = self.prepare_chat_batch([msgs])
            kv = self._as_legacy(
                self.generate_latent_batch(
                    input_ids,
                    attention_mask=attention_mask,
                    latent_steps=latent_steps,
                    past_key_values=kv,
                )
            )
            if self.decode_bool(
                kv,
                "Based on the exchange so far, do you have enough to conclude the discussion?",
            ):
                break
        return kv
