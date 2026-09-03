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
import qwix
import taktiny.nn as nn
from taktiny.nn.continuo import _constrain
from taktiny.utils.typing import ShardMode

from destiny.cosette.continuo import (
    _activation,
    _config_value,
    _hidden_size,
    _model_dtype,
    _positive_int,
    _shard_mode,
)
from destiny.cosette.layers import (
    AdaXNorm,
    Attention,
    FeedForward,
    GateMLP,
    GLUMBConv,
    JointAttention,
    StackLayer,
    _RotaryEmbedding,
)
from destiny.cosette.layers import (
    ConditionalTransformerLayer as _ConditionalTransformerLayer,
)
from destiny.cosette.layers import (
    GatedParallelTransformerLayer as _GatedParallelTransformerLayer,
)
from destiny.cosette.layers import JointTransformerLayer as _JointTransformerLayer
from destiny.cosette.utils import (
    AttentionLike,
    GateMLPLike,
    LayerNormLike,
    ModelConfig,
    ModelOutput,
    RMSNormLike,
    ShardingRule,
    _validate_dtype_config,
)
from destiny.maestro.concerto import GenericHub, GenericPretrained
from destiny.maestro.generation import AutoregressiveGeneration
from destiny.utils.sharding import create_sharding

from destiny.cosette.utils import AxisName, ModuleMap

PositionEmbedding = tuple[jax.Array, jax.Array]
PositionEmbeddings = PositionEmbedding | tp.Mapping[str, PositionEmbedding]

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
        axis_names: AxisName | None = None,
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
        self.axis_names = axis_names
        norm_axis_names = (
            None if axis_names is None else axis_names.get('norm_weight')
        )
        self.norm1 = self._norm1(
            config.hidden_size,
            config.rms_norm_eps,
            axis_names=norm_axis_names,
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
            axis_names=axis_names,
            dot_general=config.dot_general,
            shard_mode=config.shard_mode,
            **_default_attention_kwargs,
        )
        self.norm2 = self._norm2(
            config.hidden_size,
            config.rms_norm_eps,
            axis_names=norm_axis_names,
            shard_mode=config.shard_mode
        )
        self.ffn = self._ffn(
            config.hidden_size,
            config.intermediate_size,
            activation=config.hidden_act or config.hidden_activation,
            bias=bool(config.mlp_bias),
            dtype=config.dtype,
            rngs=rngs,
            axis_names=axis_names,
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


class TransformerModel(nn.Module):
    _layer_type = None
    _token_embedding = nn.Embedding
    _norm = nn.RMSNorm

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nn.Rngs,
        axis_names: AxisName | None = None,
        **kwargs: tp.Any,
    ) -> None:
        if self._layer_type is None:
            raise ValueError('_layer_type cannot be None')
        self.config = config
        self.axis_names = axis_names
        head_dim = (
            config.head_dim
            or config.hidden_size // config.num_attention_heads
        )
        rope_options = config.rope_parameters or config.rope_scaling
        rope_theta = config.rope_theta
        if rope_theta is None and rope_options is not None:
            rope_theta = rope_options.get('rope_theta')

        self.rotary_embedding = _RotaryEmbedding(
            head_dim,
            config.max_position_embeddings,
            base=rope_theta or 10_000.0,
            rope_scaling=rope_options,
        )
        self.token_embedding = self._token_embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.dtype,
            rngs=rngs,
            quant=config.quant,
            axis_names=(
                None if axis_names is None else axis_names.get('token_embedding')
            ),
            shard_mode=config.shard_mode
        )
        self.layers = StackLayer.init_stack(
            self._layer_type,
            config,
            num_stacks=config.num_hidden_layers,
            stack_type=config.stack_type,
            rngs=rngs,
            apply_position_fn=self.rotary_embedding.apply_rope,
            axis_names=axis_names,
            **kwargs
        )
        self.norm = self._norm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype='float32',
            axis_names=(
                None if axis_names is None else axis_names.get('norm_weight')
            ),
            shard_mode=config.shard_mode,
        )
        self.remat = False

    def enable_remat(self) -> None:
        self.remat = True

    def disable_remat(self) -> None:
        self.remat = False

    def _position_embeddings(
        self,
        x: jax.Array,
        position_ids: jax.Array | None,
    ) -> PositionEmbedding:
        return self.rotary_embedding(x, position_ids)

    def _position_embedding_for_layer(
        self,
        position_embeddings: PositionEmbeddings,
        layer_idx: jax.Array,
    ) -> PositionEmbedding:
        del layer_idx
        if isinstance(position_embeddings, tp.Mapping):
            raise TypeError(f'{self.__class__.__name__} expects one rotary position embedding')
        cosine, sine = position_embeddings
        return cosine, sine

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        kv_cache: tuple[jax.Array, jax.Array] | None = None,
        position_ids: jax.Array | None = None,
        position_embedding: tuple[jax.Array, jax.Array] | None = None,
        is_causal: bool = False,
        kernel: str = 'dot_product',
        out_sharding: tp.Any = None,
        **kwargs: tp.Any
    ) -> ModelOutput:
        x = self.token_embedding(input_ids, out_sharding=out_sharding)
        if position_embedding is None:
            position_embedding = self._position_embeddings(x, position_ids)

        def forward(
            layer: nn.Module,
            hidden_states: jax.Array,
            layer_cache: tp.Any,
            layer_idx: jax.Array,
        ) -> tuple[jax.Array, tp.Any]:
            return layer(
                hidden_states,
                attention_mask=attention_mask,
                kv_cache=layer_cache,
                position_ids=position_ids,
                position_embedding=self._position_embedding_for_layer(
                    position_embedding, layer_idx
                ),
                layer_idx=layer_idx,
                is_causal=is_causal,
                kernel=kernel,
                out_sharding=out_sharding,
                **kwargs,
            )

        apply_layer = forward
        if self.remat:
            apply_layer = jax.checkpoint(
                forward,
                prevent_cse=isinstance(self.layers, nn.List),
            )

        x, cache = StackLayer.call_stack(
            self.layers,
            apply_layer,
            x,
            per_layer=kv_cache,
            with_layer_index=True,
        )
        x = self.norm(x, out_sharding=out_sharding)
        return ModelOutput(x=x, kv_cache=cache)


class TransformerCausalLM[T: TransformerCausalLM](
    GenericPretrained,
    AutoregressiveGeneration,
    GenericHub,
    nn.Module,
):
    _model_type = None
    _default_sharding_rules = (
        ShardingRule
        .set_logical_rp('embed', 'head_dim', 'sequence')
        .set_logical_tp('vocab', 'heads', 'kv_heads', 'mlp')
        .set_logical_dp('batch')
    )
    _default_module_map = (
        ModuleMap
        .map('model.embed_tokens.weight', 'model.token_embedding.embedding')
        .map('input_layernorm', 'norm1')
        .map('self_attn', 'attention')
        .map('post_attention_layernorm', 'norm2')
        .map('mlp', 'ffn')
    )
    _default_axis_names = (
        AxisName.set_axis_names(
            token_embedding = ('vocab', 'embed'),
            norm_weight     = ('embed',),
            q_proj          = ('embed', 'heads', 'head_dim'),
            k_proj          = ('embed', 'kv_heads', 'head_dim'),
            v_proj          = ('embed', 'kv_heads', 'head_dim'),
            o_proj          = ('heads', 'head_dim', 'embed'),
            g_proj          = ('embed', 'mlp'),
            u_proj          = ('embed', 'mlp'),
            d_proj          = ('mlp', 'embed'),
            lm_head         = ('embed', 'vocab')
        )
    )
    _default_config = None

    def __init__(
        self: T, 
        config: ModelConfig, 
        *, 
        rngs: nn.Rngs, 
        axis_names: AxisName | None = None,
        **kwargs: tp.Any
    ) -> None:
        if self._model_type is None or self._default_config is None:
            raise ValueError('_model_type and _default_config cannot be None')

        if isinstance(self._default_config, ModelConfig):
            self._default_config = type(self._default_config)

        if not issubclass(self._default_config, ModelConfig):
            raise TypeError('_default_config shoule be subclass of ModelConfig')

        # passed config is an instance
        # self._default_config is a class
        if type(config) is not self._default_config:
            config = self._default_config(**config.to_dict())
        self.config = config

        if axis_names is not None and not isinstance(axis_names, AxisName):
            raise TypeError('axis_names must be an AxisName or None')
        self.axis_names = (
            self._default_axis_names.copy()
            if axis_names is None
            else axis_names.copy()
        )
        self.quant = kwargs.pop('quant', config.quant)
        self.shard_mode = kwargs.pop(
            'shard_mode',
            config.shard_mode or ShardMode.AUTO,
        )
        self.dot_general = kwargs.pop('dot_general', config.dot_general)
        self.mesh = kwargs.pop('mesh', config.mesh)
        self.sharding_rules = kwargs.pop(
            'sharding_rules',
            config.sharding_rules,
        )
        self.stack_type = kwargs.pop('stack_type', config.stack_type)
        self.dtype = _validate_dtype_config(config)

        self.logits_out_sharding = None
        if self.mesh is not None and self.shard_mode == ShardMode.EXPLICIT:
            self.logits_out_sharding = create_sharding(
                self.mesh,
                ('batch', 'sequence', 'vocab'),
                rules=(
                    self.sharding_rules or \
                        self._default_sharding_rules
                ),
            )

        self.model = self._model_type(
            config,
            rngs=rngs,
            axis_names=self.axis_names,
            **kwargs,
        )
        self.tied_word_embeddings = bool(config.tie_word_embeddings)
        self.lm_head = None
        if not self.tied_word_embeddings:
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                dtype=config.dtype,
                rngs=rngs,
                quant=config.quant,
                dot_general=config.dot_general,
                axis_names=self.axis_names.get('lm_head'),
                shard_mode=config.shard_mode,
            )

    def enable_remat(self) -> None:
        self.model.enable_remat()

    def disable_remat(self) -> None:
        self.model.disable_remat()

    def _lm_weight(self) -> tp.Any:
        if self.tied_word_embeddings:
            return self.model.token_embedding.embedding.value.T
        if not isinstance(self.lm_head, nn.Linear):
            raise TypeError('untied language models require an nn.Linear head')
        return self.lm_head.weight

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        kv_cache: tuple[jax.Array, jax.Array] | None = None,
        position_ids: jax.Array | None = None,
        position_embedding: jax.Array | None = None,
        out_sharding: tp.Any = None,
        logits_to_keep: int | jax.Array = 0,
        **kwargs: tp.Any
    ) -> ModelOutput:
        output: ModelOutput = self.model(
            input_ids,
            attention_mask,
            kv_cache=kv_cache,
            position_ids=position_ids,
            position_embedding=position_embedding,
            out_sharding=out_sharding,
            **kwargs,
        )
        x = output.pop('x')
        if isinstance(logits_to_keep, int):
            if logits_to_keep < 0:
                raise ValueError('logits_to_keep should be non-negative')
            if logits_to_keep:
                x = x[:, -logits_to_keep:, :]
        else:
            indices = jnp.asarray(logits_to_keep, dtype=jnp.int32)
            if indices.ndim != 1 or indices.shape[0] != x.shape[0]:
                raise ValueError(
                    'logits_to_keep should contain one index per batch row'
                )
            indices = jnp.where(indices < 0, indices + x.shape[1], indices)
            x = jnp.take_along_axis(
                x,
                indices[:, None, None],
                axis=1,
            )
        logits = self.compute_logits(
            x,
            self._lm_weight(),
            out_sharding=(
                self.logits_out_sharding
                if out_sharding is None
                else out_sharding
            ),
        )
        logits = self._process_logits(logits)
        return ModelOutput(logits=logits, **output)

    @staticmethod
    def compute_logits(
        lhs: jax.Array,
        rhs: tp.Any,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """Contract the last ``lhs`` axis with the first ``rhs`` axis."""
        if isinstance(rhs, nn.Parameter):
            rhs = rhs.value
        dimension_numbers = (
            (((lhs.ndim - 1,), (0,))),
            ((), ()),
        )
        if isinstance(rhs, qwix.QArray):
            logits = qwix.dot_general(lhs, rhs, dimension_numbers)
        else:
            logits = jax.lax.dot_general(lhs, rhs, dimension_numbers)
        if out_sharding is not None:
            logits = jax.lax.with_sharding_constraint(
                logits,
                out_sharding,
            )
        return logits

    def _process_logits(self, logits: jax.Array) -> jax.Array:
        """Apply architecture-specific post-processing to vocabulary logits."""
        return logits
