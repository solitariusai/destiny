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
"""Gemma3 model implementations"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from taktiny import nn

from destiny.cosette.layers import _RotaryEmbedding
from destiny.cosette.transformers.gemma import Gemma3DecoderLayer
from destiny.cosette.transformers.ordinario import PositionEmbedding, PositionEmbeddings
from destiny.maestro.symphony.gemma.config import Gemma3TextConfig
from destiny.maestro.symphony.gemma.gemma2 import Gemma2Model


class Gemma3TextModel(Gemma2Model):
    _layer_type = Gemma3DecoderLayer

    def __init__(
        self,
        config: Gemma3TextConfig,
        *,
        rngs: nn.Rngs,
        **kwargs: Any,
    ) -> None:
        pattern = config.sliding_window_pattern
        if not isinstance(pattern, int) or isinstance(pattern, bool):
            raise TypeError('sliding_window_pattern must be an integer')
        if pattern <= 0:
            raise ValueError('sliding_window_pattern must be positive')

        if config.layer_types is None:
            config.layer_types = [
                (
                    'sliding_attention'
                    if (index + 1) % pattern
                    else 'full_attention'
                )
                for index in range(config.num_hidden_layers)
            ]
        elif len(config.layer_types) != config.num_hidden_layers:
            raise ValueError(
                'config.layer_types must contain one entry per layer'
            )

        default_theta = config.default_theta
        global_theta = (
            config.rope_theta
            or default_theta.get('global')
        )
        local_theta = (
            config.rope_local_base_freq
            or default_theta.local
        )
        rope_parameters = ModelConfig(
            full_attention={
                'rope_type': 'default',
                'rope_theta': global_theta,
            },
            sliding_attention={
                'rope_type': 'default',
                'rope_theta': local_theta,
            },
        )
        if config.rope_parameters is not None:
            rope_parameters = rope_parameters.with_overrides(
                config.rope_parameters
            )
        if config.rope_scaling is not None:
            rope_parameters.full_attention = (
                rope_parameters.full_attention.with_overrides(
                    config.rope_scaling
                )
            )
        config.rope_parameters = rope_parameters

        super().__init__(config, rngs=rngs, **kwargs)
        head_dim = (
            config.head_dim
            or config.hidden_size // config.num_attention_heads
        )
        global_rope = config.rope_parameters.full_attention
        local_rope = config.rope_parameters.sliding_attention
        self.rotary_embedding = _RotaryEmbedding(
            head_dim,
            config.max_position_embeddings,
            base=global_rope.rope_theta,
            rope_scaling=global_rope,
        )
        self.local_rotary_embedding = _RotaryEmbedding(
            head_dim,
            config.max_position_embeddings,
            base=local_rope.rope_theta,
            rope_scaling=local_rope,
        )
        self.sliding_pattern = tuple(
            layer_type == 'sliding_attention'
            for layer_type in config.layer_types
        )

    def _position_embeddings(
        self,
        x: jax.Array,
        position_ids: jax.Array | None,
    ) -> Mapping[str, PositionEmbedding]:
        return {
            'full_attention': self.rotary_embedding(x, position_ids),
            'sliding_attention': self.local_rotary_embedding(x, position_ids),
        }

    def _position_embedding_for_layer(
        self,
        position_embeddings: PositionEmbeddings,
        layer_idx: jax.Array,
    ) -> PositionEmbedding:
        if not isinstance(position_embeddings, Mapping):
            cosine, sine = position_embeddings
            return cosine, sine
        use_sliding = jnp.asarray(
            self.sliding_pattern,
            dtype=jnp.bool_,
        )[layer_idx]
        local = position_embeddings['sliding_attention']
        global_ = position_embeddings['full_attention']
        return (
            jnp.where(use_sliding, local[0], global_[0]),
            jnp.where(use_sliding, local[1], global_[1]),
        )


__all__ = ['Gemma3TextModel']