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
"""Transformer abstract layer implementations"""


import typing as tp

import jax
import jax.numpy as jnp
from taktiny import nn

from destiny.cosette.layers import Attention, GateMLP
from destiny.cosette.layers.typing import (
    AttentionLike,
    GateMLPLike,
    LayerNormLike,
    RMSNormLike,
)
from destiny.maestro.utils import ModelConfig


class TransformerDecoderLayer(nn.Module):
    _norm1: RMSNormLike | LayerNormLike = nn.RMSNorm
    _norm2: RMSNormLike | LayerNormLike = nn.RMSNorm
    _attention: AttentionLike = Attention
    _ffn: GateMLPLike = GateMLP
    _attention_kwargs: tp.ClassVar = {}

    def __init__(
        self, 
        config: ModelConfig, 
        *, 
        rngs: nn.Rngs, 
        layer_idx: int | None = None, 
        **kwargs: tp.Any
    ) -> None:
        layer_types = config.layer_types
        if layer_types is not None and layer_idx is None:
            raise ValueError(f'{self.__class__.__name__} requires layer_idx')

        self.use_sliding_window = False
        self.sliding_pattern = None
        window_size = None
        if layer_types is not None and layer_idx is not None:
            if len(layer_types) != config.num_hidden_layers:
                raise ValueError(
                    'config.layer_types must contain one entry per layer'
                )
            sliding_pattern = tuple(
                layer_type in {'sliding_attention', 'sliding'}
                for layer_type in layer_types
            )
            self.sliding_pattern = sliding_pattern
            window_size = config.sliding_window
            self.use_sliding_window = jnp.asarray(
                sliding_pattern,
                dtype=jnp.bool_,
            )[layer_idx]

        _default_attention_kwargs = {
            'bias': config.attention_bias,
            'qk_norm': False,
            'window_size': window_size,
            'scaling': None,
            'softcap': None,
        }
        _default_attention_kwargs.update(self._attention_kwargs)
        apply_position_fn = kwargs.get('apply_position_fn', None)
        self.norm1 = self._norm1(
            config.hidden_size,
            config.rms_norm_eps,
            axis_names=config.rmsnorm_weight_axis_names,
            shard_mode=config.shard_mode
        )
        self.attention = self._attention(
            config.hidden_size,
            config.num_attention_heads,
            config.head_dim or config.hidden_size // config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            apply_position_fn=apply_position_fn,
            epsilon=config.rms_norm_eps,
            dropout=config.attention_dropout or 0.0,
            dtype=config.dtype,
            rngs=rngs,
            quant=config.quant,
            q_axis_names=config.attention_q_proj_axis_names,
            k_axis_names=config.attention_k_proj_axis_names,
            v_axis_names=config.attention_v_proj_axis_names,
            o_axis_names=config.attention_o_proj_axis_names,
            dot_general=config.dot_general,
            shard_mode=config.shard_mode,
            **_default_attention_kwargs,
        )
        self.norm2 = self._norm2(
            config.hidden_size,
            config.rms_norm_eps,
            axis_names=config.rmsnorm_weight_axis_names,
            shard_mode=config.shard_mode
        )
        self.ffn = self._ffn(
            config.hidden_size,
            config.intermediate_size,
            activation=config.hidden_act or config.hidden_activation,
            bias=bool(config.mlp_bias),
            dtype=config.dtype,
            rngs=rngs,
            gate_axis_names=config.gatemlp_gate_proj_axis_names,
            up_axis_names=config.gatemlp_up_proj_axis_names,
            down_axis_names=config.gatemlp_down_proj_axis_names,
            shard_mode=config.shard_mode,
            quant=config.quant,
            dot_general=config.dot_general
        )

    def __call__(
        self,
        x: jax.Array,
        attention_mask: jax.Array | None = None,
        is_causal: bool = False,
        kv_cache: tuple[jax.Array, jax.Array] | None = None,
        cache_position: jax.Array | None = None,
        position_ids: jax.Array | None = None,
        position_embedding: jax.Array | None = None,
        boundary_ids: jax.Array | None = None,
        kernel: str = 'dot_product',
        out_sharding: tp.Any = None,
        **kwargs: tp.Any,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        z = x
        x, updated_cache = self.attention(
            self.norm1(x, out_sharding=out_sharding),
            attention_mask=attention_mask,
            is_causal=is_causal,
            kv_cache=kv_cache,
            position_ids=position_ids,
            cache_position=cache_position,
            position_embedding=position_embedding,
            boundary_ids=boundary_ids,
            use_sliding_window=self.use_sliding_window,
            kernel=kernel,
            out_sharding=out_sharding
        )
        x = x + z
        x = self.ffn(
            self.norm2(x, out_sharding=out_sharding),
            out_sharding=out_sharding
        ) + x
        return x, updated_cache