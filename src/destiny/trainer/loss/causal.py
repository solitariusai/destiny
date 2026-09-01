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
"""Causal loss functions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
from taktiny.utils.typing import Array, Batch

from destiny.trainer.loss.classification import cross_entropy_loss


def _chunked_causal_lm_loss(
    model: Any,
    input_ids: jax.Array,
    labels: jax.Array,
    *,
    attention_mask: jax.Array | None = None,
    position_ids: jax.Array | None = None,
    ignore_index: int = -100,
    logits_chunk_size: int = 256,
    attention_kernel: str = 'dot_product',
    reduction: str = 'mean',
) -> jax.Array:
    """Compute causal loss without materializing full-sequence logits.

    The transformer runs once for the complete sequence. Vocabulary
    projection and cross entropy then run over rematerialized sequence
    chunks, bounding the live logits tensor to
    ``batch * logits_chunk_size * vocab_size``.
    """
    if reduction not in {'sum', 'mean'}:
        raise ValueError(
            'chunked causal loss supports only "sum" and "mean" '
            'reductions'
        )
    if not isinstance(logits_chunk_size, int) or isinstance(
        logits_chunk_size,
        bool,
    ) or logits_chunk_size <= 0:
        raise ValueError('logits_chunk_size must be a positive integer')

    input_ids = jnp.asarray(input_ids)
    labels = jnp.asarray(labels)
    if input_ids.ndim != 2:
        raise ValueError(
            'input_ids must have shape [batch, sequence], '
            f'got {input_ids.shape}'
        )
    if labels.shape != input_ids.shape:
        raise ValueError(
            'labels and input_ids must have equal shapes, got '
            f'{labels.shape} and {input_ids.shape}'
        )
    if input_ids.shape[1] < 2:
        raise ValueError('causal LM loss requires at least two tokens')

    token_mask = None
    model_attention_mask = None
    if attention_mask is not None:
        attention_mask = jnp.asarray(attention_mask, dtype=jnp.bool_)
        if attention_mask.ndim == 2:
            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    'a two-dimensional attention_mask must match input_ids'
                )
            token_mask = attention_mask
            model_attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.ndim in (3, 4):
            model_attention_mask = attention_mask
        else:
            raise ValueError(
                'attention_mask must have two, three, or four dimensions'
            )

    if position_ids is not None:
        position_ids = jnp.asarray(position_ids, dtype=jnp.int32)
        if position_ids.shape != input_ids.shape:
            raise ValueError(
                'position_ids and input_ids must have equal shapes'
            )

    model_output = model.model(
        input_ids,
        attention_mask=model_attention_mask,
        position_ids=position_ids,
        is_causal=True,
        kernel=attention_kernel,
    )
    hidden_states = model_output.x[:, :-1, :]
    shifted_labels = labels[:, 1:]
    target_mask = token_mask[:, 1:] if token_mask is not None else None
    if position_ids is not None:
        within_sequence = position_ids[:, 1:] != 0
        target_mask = (
            within_sequence
            if target_mask is None
            else target_mask & within_sequence
        )

    num_positions = hidden_states.shape[1]
    chunk_size = min(logits_chunk_size, num_positions)
    num_chunks = (num_positions + chunk_size - 1) // chunk_size
    padding = num_chunks * chunk_size - num_positions
    if padding:
        hidden_states = jnp.pad(
            hidden_states,
            ((0, 0), (0, padding), (0, 0)),
        )
        shifted_labels = jnp.pad(
            shifted_labels,
            ((0, 0), (0, padding)),
            constant_values=ignore_index,
        )
        if target_mask is not None:
            target_mask = jnp.pad(
                target_mask,
                ((0, 0), (0, padding)),
                constant_values=False,
            )

    lm_weight = model._lm_weight()

    @jax.checkpoint
    def scan_body(
        carry: tuple[jax.Array, jax.Array],
        index: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        start = index * chunk_size
        hidden_chunk = jax.lax.dynamic_slice_in_dim(
            hidden_states,
            start,
            chunk_size,
            axis=1,
        )
        label_chunk = jax.lax.dynamic_slice_in_dim(
            shifted_labels,
            start,
            chunk_size,
            axis=1,
        )
        mask_chunk = None
        if target_mask is not None:
            mask_chunk = jax.lax.dynamic_slice_in_dim(
                target_mask,
                start,
                chunk_size,
                axis=1,
            )

        logits = model.compute_logits(hidden_chunk, lm_weight)
        logits = model._process_logits(logits)
        chunk_loss = cross_entropy_loss(
            logits,
            label_chunk,
            mask=mask_chunk,
            ignore_index=ignore_index,
            reduction='sum',
        )
        selected = label_chunk != ignore_index
        if mask_chunk is not None:
            selected &= mask_chunk
        chunk_count = jnp.sum(selected, dtype=jnp.float32)
        loss_sum, count_sum = carry
        return (
            loss_sum + chunk_loss,
            count_sum + chunk_count,
        ), None

    (loss_sum, count), _ = jax.lax.scan(
        scan_body,
        (
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        ),
        jnp.arange(num_chunks, dtype=jnp.int32),
    )
    if reduction == 'sum':
        return loss_sum
    return loss_sum / jnp.maximum(count, 1.0)


def causal_lm_loss(
    model: Callable[..., Any],
    batch: Batch,
    *,
    ignore_index: int = -100,
    logits_chunk_size: int | None = None,
    attention_kernel: str = 'dot_product',
) -> Array:
    """Compute next-token loss for a TakTiny causal language model.

    ``batch`` must contain ``input_ids`` and ``labels``. Optional
    ``attention_mask`` and ``position_ids`` values are forwarded to the model.
    A two-dimensional attention mask is interpreted as a key-padding mask.
    Reset positions mark packed sequence boundaries and are excluded from the
    shifted targets.

    ``logits_chunk_size`` enables a chunked vocabulary projection: the LM
    head and cross entropy run over sequence chunks inside a rematerialized
    scan, so the ``(sequence, vocab)`` logits tensor never materializes at
    full size.
    ``attention_kernel`` selects the attention backend for the decoder
    (``'dot_product'``, ``'flash'``, ``'ragged'``, ...).

    This function has the ``loss_fn(model, batch)`` signature expected by
    :class:`~taktiny.trainer.Trainer`.
    """
    if not isinstance(batch, Mapping):
        raise TypeError('batch must be a mapping')
    missing = {'input_ids', 'labels'} - set(batch)
    if missing:
        names = ', '.join(sorted(missing))
        raise KeyError(f'causal LM batch is missing: {names}')

    input_ids = jnp.asarray(batch['input_ids'])
    labels = jnp.asarray(batch['labels'])
    if input_ids.ndim != 2:
        raise ValueError(
            'input_ids must have shape [batch, sequence], '
            f'got {input_ids.shape}'
        )
    if labels.shape != input_ids.shape:
        raise ValueError(
            'labels and input_ids must have equal shapes, got '
            f'{labels.shape} and {input_ids.shape}'
        )
    if input_ids.shape[1] < 2:
        raise ValueError('causal LM loss requires at least two tokens')

    attention_mask = batch.get('attention_mask')
    token_mask = None
    if attention_mask is not None:
        attention_mask = jnp.asarray(attention_mask, dtype=jnp.bool_)
        if attention_mask.ndim == 2:
            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    'a two-dimensional attention_mask must match input_ids'
                )
            token_mask = attention_mask
            attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.ndim not in (3, 4):
            raise ValueError(
                'attention_mask must have two, three, or four dimensions'
            )

    position_ids = batch.get('position_ids')
    if position_ids is not None:
        position_ids = jnp.asarray(position_ids, dtype=jnp.int32)
        if position_ids.shape != input_ids.shape:
            raise ValueError('position_ids and input_ids must have equal shapes')

    if logits_chunk_size is not None:
        return _chunked_causal_lm_loss(
            model,
            input_ids,
            labels,
            attention_mask=batch.get('attention_mask'),
            position_ids=batch.get('position_ids'),
            ignore_index=ignore_index,
            logits_chunk_size=logits_chunk_size,
            attention_kernel=attention_kernel,
        )

    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        is_causal=True,
        kernel=attention_kernel,
    )
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    if hasattr(logits, 'logits'):
        logits = logits.logits
    logits = jnp.asarray(logits)
    if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise ValueError(
            'model logits must have shape [batch, sequence, vocabulary], '
            f'got {logits.shape}; expected batch and sequence dimensions '
            f'{input_ids.shape}'
        )

    target_mask = None
    if token_mask is not None:
        target_mask = token_mask[:, 1:]
    if position_ids is not None:
        boundaries = position_ids[:, 1:] != 0
        target_mask = (
            boundaries
            if target_mask is None
            else target_mask & boundaries
        )

    return cross_entropy_loss(
        logits[:, :-1, :],
        labels[:, 1:],
        mask=target_mask,
        ignore_index=ignore_index,
    )


__all__ = ['causal_lm_loss']
