# Copyright 2026 Shinapri
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Autoregressive token-generation capability."""

from __future__ import annotations

import typing as tp
from collections.abc import Iterator, Mapping, Sequence
from functools import partial

import jax
import jax.numpy as jnp

from destiny.maestro.generation.base import Generation
from destiny.utils.typing import ArrayLike

type DecodeCarry = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]
type GenerationSettings = tuple[int, jax.Array, int, str]


class AutoregressiveGeneration(Generation):
    """Reusable prefill-and-decode generation for cache-aware models."""

    def _sample(
        self,
        logits: jax.Array,
        temperature: float,
        top_k: int,
        top_p: float,
        key: jax.Array,
        seen_tokens: jax.Array | None = None,
        repetition_penalty: float = 1.0,
    ) -> jax.Array:
        if logits.ndim != 2:
            raise ValueError('logits should have shape [batch, vocabulary]')
        if seen_tokens is not None:
            if seen_tokens.shape != logits.shape:
                raise ValueError(
                    'seen_tokens should have the same shape as logits'
                )
            penalized = jnp.where(
                logits < 0,
                logits * repetition_penalty,
                logits / repetition_penalty,
            )
            logits = jnp.where(seen_tokens, penalized, logits)

        greedy_tokens = jnp.argmax(logits, axis=-1)[:, None]
        logits = logits / jnp.maximum(temperature, 1e-5)

        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            top_k_logits, _ = jax.lax.top_k(logits, top_k)
            min_top_k = top_k_logits[:, -1:]
            logits = jnp.where(logits >= min_top_k, logits, -jnp.inf)

        if top_p < 1.0:
            sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]
            sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
            cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the mask to the right to keep the first token that crosses the threshold
            sorted_indices_to_remove = jnp.roll(sorted_indices_to_remove, 1, axis=-1)
            sorted_indices_to_remove = sorted_indices_to_remove.at[:, 0].set(False)

            # Map back to original order
            indices_to_remove = jnp.empty_like(sorted_indices_to_remove)
            indices_to_remove = indices_to_remove.at[
                jnp.arange(logits.shape[0])[:, None], sorted_indices
            ].set(sorted_indices_to_remove)

            logits = jnp.where(indices_to_remove, -jnp.inf, logits)

        sampled_tokens = jax.random.categorical(key, logits)[:, None]
        return jnp.where(temperature <= 0, greedy_tokens, sampled_tokens)

    @staticmethod
    def _canonical_attention_kernel(kernel: str) -> str:
        if not isinstance(kernel, str):
            raise TypeError('attention kernel names must be strings')
        normalized = kernel.strip().lower()
        aliases = {
            'standard': 'dot_product',
            'jax': 'dot_product',
            'flash_attention': 'flash',
        }
        normalized = aliases.get(normalized, normalized)
        supported = {'auto', 'dot_product', 'flash'}
        if normalized not in supported:
            choices = ', '.join(sorted(supported))
            raise ValueError(
                f'unsupported attention kernel {kernel!r}; choose from '
                f'{choices}'
            )
        return normalized

    def _resolve_generation_attention_kernels(
        self,
        attention_kernel: str | Mapping[str, str],
    ) -> tuple[str, str]:
        if isinstance(attention_kernel, str):
            prefill = decode = self._canonical_attention_kernel(
                attention_kernel
            )
        elif isinstance(attention_kernel, Mapping):
            unknown = set(attention_kernel) - {'prefill', 'decode'}
            if unknown:
                names = ', '.join(sorted(map(str, unknown)))
                raise ValueError(
                    f'unknown attention_kernel phase keys: {names}'
                )
            prefill = self._canonical_attention_kernel(
                attention_kernel.get('prefill', 'auto')
            )
            decode = self._canonical_attention_kernel(
                attention_kernel.get('decode', 'auto')
            )
        else:
            raise TypeError(
                'attention_kernel must be a string or a mapping with '
                'prefill and decode keys'
            )

        if prefill == 'auto':
            prefill = 'dot_product'
        if decode == 'auto':
            decode = 'dot_product'
        return prefill, decode

    @partial(
        jax.jit,
        static_argnames=[
            'max_seq_len',
            'top_k',
            'top_p',
            'attention_kernel',
        ],
    )
    def _decode_step(
        self,
        carry: DecodeCarry,
        max_seq_len: int | None = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        eos_token_ids: jax.Array | None = None,
        pad_token_id: int = 0,
        attention_kernel: str = 'dot_product',
    ) -> tuple[DecodeCarry, jax.Array]:
        (
            token,
            k_cache,
            v_cache,
            pos,
            rng,
            finished,
            seen_tokens,
        ) = carry

        if max_seq_len is None:
            raise ValueError('max_seq_len is required')
        if eos_token_ids is None:
            eos_token_ids = jnp.empty((0,), dtype=token.dtype)

        position_ids = pos[:, None]
        mask = jnp.arange(max_seq_len)[None, :] <= position_ids
        mask = mask[:, None, None, :]

        output = self(
            token,
            attention_mask=mask,
            kv_cache=(k_cache, v_cache),
            position_ids=position_ids,
            cache_position=position_ids,
            is_causal=False,
            kernel=attention_kernel,
            logits_to_keep=1,
        )
        step_logits = output.logits
        updated_cache = output.kv_cache
        if updated_cache is None:
            raise ValueError('model did not return an updated KV cache')
        updated_k_cache, updated_v_cache = updated_cache

        rng, subkey = jax.random.split(rng)
        next_t = self._sample(
            step_logits[:, -1, :],
            temperature,
            top_k,
            top_p,
            subkey,
            seen_tokens=seen_tokens,
            repetition_penalty=repetition_penalty,
        )
        next_t = jnp.where(
            finished[:, None],
            jnp.asarray(pad_token_id, dtype=next_t.dtype),
            next_t,
        )

        if eos_token_ids.shape[0]:
            newly_finished = jnp.any(
                next_t == eos_token_ids[None, :],
                axis=-1,
            )
        else:
            newly_finished = jnp.zeros_like(finished)

        active = ~finished
        finished = finished | newly_finished
        batch_indices = jnp.arange(next_t.shape[0])
        seen_tokens = seen_tokens.at[
            batch_indices,
            next_t[:, 0],
        ].set(True)

        return (
            next_t,
            updated_k_cache,
            updated_v_cache,
            pos + active.astype(pos.dtype),
            rng,
            finished,
            seen_tokens,
        ), next_t

    def _prepare_generation(
        self,
        input_ids: ArrayLike,
        max_new_tokens: int,
        *,
        attention_mask: ArrayLike | None,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        eos_token_id: int | Sequence[int] | None,
        pad_token_id: int | None,
        seed: int,
        attention_kernel: str | Mapping[str, str],
    ) -> tuple[jax.Array, DecodeCarry, GenerationSettings]:
        if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError('max_new_tokens should be a positive integer')
        if not isinstance(top_k, int) or top_k < 0:
            raise ValueError('top_k should be a non-negative integer')
        if not 0 < top_p <= 1:
            raise ValueError('top_p should be in the interval (0, 1]')
        if repetition_penalty <= 0:
            raise ValueError('repetition_penalty should be positive')
        prefill_kernel, decode_kernel = (
            self._resolve_generation_attention_kernels(attention_kernel)
        )

        input_ids = jnp.asarray(input_ids)
        if input_ids.ndim != 2:
            raise ValueError('input_ids should have shape [batch, sequence]')
        batch_size, seq_len = input_ids.shape

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids, dtype=jnp.bool_)
        else:
            attention_mask = jnp.asarray(attention_mask, dtype=jnp.bool_)
            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    'attention_mask should have the same shape as input_ids'
                )

        prompt_lengths = jnp.sum(
            attention_mask,
            axis=-1,
            dtype=jnp.int32,
        )
        if bool(jnp.any(prompt_lengths == 0)):
            raise ValueError('each prompt should contain at least one token')

        compact_order = jnp.argsort(
            ~attention_mask,
            axis=-1,
            stable=True,
        )
        compact_ids = jnp.take_along_axis(
            input_ids,
            compact_order,
            axis=-1,
        )
        compact_mask = (
            jnp.arange(seq_len)[None, :] < prompt_lengths[:, None]
        )

        eos_value = (
            getattr(self.config, 'eos_token_id', None)
            if eos_token_id is None
            else eos_token_id
        )
        if eos_value is None:
            eos_values = ()
        elif isinstance(eos_value, (list, tuple)):
            eos_values = tuple(int(token_id) for token_id in eos_value)
        else:
            eos_values = (int(eos_value),)
        eos_token_ids = jnp.asarray(eos_values, dtype=input_ids.dtype)

        if pad_token_id is None:
            pad_token_id = getattr(self.config, 'pad_token_id', None)
        if pad_token_id is None:
            pad_token_id = eos_values[0] if eos_values else 0
        pad_token_id = int(pad_token_id)

        if not isinstance(seed, int):
            raise TypeError('seed should be an integer')
        key = jax.random.key(seed)

        num_layers = len(self.model.layers)
        num_heads = getattr(self.config, 'num_attention_heads', None)
        num_kv_heads = getattr(self.config, 'num_key_value_heads', None)
        hidden_size = getattr(self.config, 'hidden_size', None)
        if num_heads is None:
            raise ValueError('config.num_attention_heads is required')
        if num_kv_heads is None:
            raise ValueError('config.num_key_value_heads is required')
        if hidden_size is None:
            raise ValueError('config.hidden_size is required')

        head_dim = (
            getattr(self.config, 'head_dim', None)
            or hidden_size // num_heads
        )
        max_seq_len = seq_len + max_new_tokens
        model_dtype = jnp.dtype(self.config.dtype)
        if not jnp.issubdtype(model_dtype, jnp.inexact):
            raise TypeError(
                'model compute dtype should be floating-point, '
                f'got {model_dtype}'
            )

        layer_types = getattr(self.config, 'layer_types', None)
        cache_layouts = []
        if layer_types is not None:
            global_head_dim = (
                getattr(self.config, 'global_head_dim', None)
                or head_dim
            )
            global_num_kv_heads = (
                getattr(
                    self.config,
                    'num_global_key_value_heads',
                    None,
                )
                or num_kv_heads
            )
            for layer_type in layer_types[:num_layers]:
                if layer_type in ('full_attention', 'full'):
                    cache_layouts.append(
                        (global_num_kv_heads, global_head_dim)
                    )
                else:
                    cache_layouts.append((num_kv_heads, head_dim))
        if not cache_layouts:
            cache_layouts = [(num_kv_heads, head_dim)] * num_layers

        unique_cache_layouts = set(cache_layouts)
        if len(unique_cache_layouts) != 1:
            raise ValueError(
                'generation currently requires one KV-cache layout across all '
                'layers'
            )
        cache_num_heads, cache_head_dim = cache_layouts[0]
        cache_shape = (
            num_layers,
            batch_size,
            max_seq_len,
            cache_num_heads,
            cache_head_dim,
        )
        key_cache = jnp.zeros(cache_shape, dtype=model_dtype)
        value_cache = jnp.zeros(cache_shape, dtype=model_dtype)
        position_ids = jnp.broadcast_to(
            jnp.arange(seq_len, dtype=jnp.int32)[None, :],
            (batch_size, seq_len),
        )
        prefill_mask = (
            jnp.arange(max_seq_len)[None, :]
            < prompt_lengths[:, None]
        )
        prefill_mask = prefill_mask[:, None, None, :]
        output = self(
            compact_ids,
            attention_mask=prefill_mask,
            kv_cache=(key_cache, value_cache),
            position_ids=position_ids,
            cache_position=position_ids,
            is_causal=True,
            kernel=prefill_kernel,
            logits_to_keep=prompt_lengths - 1,
        )
        logits = output.logits
        updated_cache = output.kv_cache
        if updated_cache is None:
            raise ValueError('model did not return an updated KV cache')
        key_cache, value_cache = updated_cache

        seen_tokens = jnp.zeros(
            (batch_size, self.config.vocab_size),
            dtype=jnp.int32,
        )
        batch_indices = jnp.broadcast_to(
            jnp.arange(batch_size)[:, None],
            compact_ids.shape,
        )
        seen_tokens = seen_tokens.at[
            batch_indices,
            compact_ids,
        ].add(compact_mask.astype(jnp.int32))
        seen_tokens = seen_tokens > 0

        key, subkey = jax.random.split(key)
        next_token = self._sample(
            logits[:, -1, :],
            temperature,
            top_k,
            top_p,
            subkey,
            seen_tokens=seen_tokens,
            repetition_penalty=repetition_penalty,
        )
        if eos_token_ids.shape[0]:
            finished = jnp.any(
                next_token == eos_token_ids[None, :],
                axis=-1,
            )
        else:
            finished = jnp.zeros((batch_size,), dtype=jnp.bool_)
        seen_tokens = seen_tokens.at[
            jnp.arange(batch_size),
            next_token[:, 0],
        ].set(True)

        carry = (
            next_token,
            key_cache,
            value_cache,
            prompt_lengths,
            key,
            finished,
            seen_tokens,
        )
        settings = (
            max_seq_len,
            eos_token_ids,
            pad_token_id,
            decode_kernel,
        )
        return input_ids, carry, settings

    def generate(
        self,
        input_ids: ArrayLike,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        seed: int = 42,
        attention_mask: ArrayLike | None = None,
        repetition_penalty: float = 1.0,
        eos_token_id: int | Sequence[int] | None = None,
        pad_token_id: int | None = None,
        streamer: tp.Any = None,
        attention_kernel: str | Mapping[str, str] = 'auto',
    ) -> jax.Array:
        """Generate tokens using a direct tuple KV cache.

        ``attention_kernel`` accepts one kernel for both phases or a mapping
        such as ``{'prefill': 'flash', 'decode': 'dot_product'}``. ``'auto'``
        uses JAX dot-product attention for both phases; Flash Attention remains
        available as an explicit phase override.
        """
        if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError('max_new_tokens should be a non-negative integer')

        input_ids = jnp.asarray(input_ids)
        if input_ids.ndim != 2:
            raise ValueError('input_ids should have shape [batch, sequence]')
        if streamer is not None:
            if not callable(getattr(streamer, 'put', None)):
                raise TypeError('streamer should provide a callable put method')
            if not callable(getattr(streamer, 'end', None)):
                raise TypeError('streamer should provide a callable end method')

            streamer.put(jax.device_get(input_ids))
            generated = []
            try:
                for token_ids in self.stream_generate(
                    input_ids,
                    max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    seed=seed,
                    attention_mask=attention_mask,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    attention_kernel=attention_kernel,
                ):
                    streamer.put(jax.device_get(token_ids))
                    generated.append(token_ids)
            finally:
                streamer.end()

            if generated:
                return jnp.concatenate(
                    [input_ids, *generated],
                    axis=1,
                )
            return input_ids

        if max_new_tokens == 0:
            return input_ids

        input_ids, carry, settings = self._prepare_generation(
            input_ids,
            max_new_tokens,
            attention_mask=attention_mask,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            seed=seed,
            attention_kernel=attention_kernel,
        )
        (
            max_seq_len,
            eos_token_ids,
            pad_token_id,
            decode_kernel,
        ) = settings
        batch_size = input_ids.shape[0]
        generated = jnp.full(
            (batch_size, max_new_tokens),
            pad_token_id,
            dtype=input_ids.dtype,
        )
        generated = generated.at[:, 0].set(carry[0][:, 0])

        def cond_fn(
            loop_state: tuple[jax.Array, DecodeCarry, jax.Array],
        ) -> jax.Array:
            step, decode_carry, _ = loop_state
            return (step < max_new_tokens) & ~jnp.all(decode_carry[5])

        def body_fn(
            loop_state: tuple[jax.Array, DecodeCarry, jax.Array],
        ) -> tuple[jax.Array, DecodeCarry, jax.Array]:
            step, decode_carry, tokens = loop_state
            decode_carry, next_token = self._decode_step(
                decode_carry,
                max_seq_len=max_seq_len,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_ids=eos_token_ids,
                pad_token_id=pad_token_id,
                attention_kernel=decode_kernel,
            )
            tokens = tokens.at[:, step].set(next_token[:, 0])
            return step + 1, decode_carry, tokens

        generated_count, _, generated = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (jnp.asarray(1, dtype=jnp.int32), carry, generated),
        )
        generated_count = int(jax.device_get(generated_count))
        return jnp.concatenate(
            [input_ids, generated[:, :generated_count]],
            axis=1,
        )

    def stream_generate(
        self,
        input_ids: ArrayLike,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        seed: int = 42,
        attention_mask: ArrayLike | None = None,
        repetition_penalty: float = 1.0,
        eos_token_id: int | Sequence[int] | None = None,
        pad_token_id: int | None = None,
        attention_kernel: str | Mapping[str, str] = 'auto',
    ) -> Iterator[jax.Array]:
        """Yield generated tokens using the same kernel policy as ``generate``."""
        if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError('max_new_tokens should be a non-negative integer')
        if max_new_tokens == 0:
            return

        _, carry, settings = self._prepare_generation(
            input_ids,
            max_new_tokens,
            attention_mask=attention_mask,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            seed=seed,
            attention_kernel=attention_kernel,
        )
        (
            max_seq_len,
            eos_token_ids,
            pad_token_id,
            decode_kernel,
        ) = settings
        yield carry[0]

        for _ in range(max_new_tokens - 1):
            if bool(jax.device_get(jnp.all(carry[5]))):
                break
            carry, next_token = self._decode_step(
                carry,
                max_seq_len=max_seq_len,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_ids=eos_token_ids,
                pad_token_id=pad_token_id,
                attention_kernel=decode_kernel,
            )
            yield next_token

__all__ = ['AutoregressiveGeneration']

