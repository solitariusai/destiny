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
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
import qwix
import taktiny.nn as nn
from taktiny.maestro.overture import PretrainedModel
from taktiny.nn.continuo import _constrain

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
from destiny.utils.sharding import create_sharding
from destiny.utils.typing import (
    ArrayLike,
    LogicalRules,
    PathLike,
    ShardMode,
)

from destiny.cosette.utils import AxisName, ModuleMap

KVCache = tuple[jax.Array, jax.Array]
DecodeCarry = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]
GenerationSettings = tuple[int, jax.Array, int, str]
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


class TransformerCausalLM[T: TransformerCausalLM](PretrainedModel):
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
    _default_config = ModelConfig()

    def __init__(
        self: T, 
        config: ModelConfig, 
        *, 
        rngs: nn.Rngs, 
        axis_names: AxisName | None = None,
        **kwargs: tp.Any
    ) -> None:
        if self._model_type is None:
            raise ValueError('_model_type cannot be None')

        self.config = self._default_config.with_overrides(config)
        config = self.config
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
        kv_cache: tp.Tuple[jax.Array, jax.Array] | None = None,
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

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: PathLike,
        *,
        config: ModelConfig | None = None,
        module_map: Sequence[tuple[tp.Any, ...]] | ModuleMap | None = None,
        local: bool = False,
        **kwargs: tp.Any,
    ) -> tp.Self:
        if config is None:
            config = ModelConfig.load_config(path_or_repo, local=local)
        if config is None:
            raise ValueError(
                f'Unable to load config from {path_or_repo!r} (local={local})'
            )
        rules = cls._default_module_map.copy()
        if config.tie_word_embeddings:
            rules.map(
                'lm_head.weight',
                'model.token_embedding.embedding',
            )
        if module_map:
            rules.extend(module_map)
        return PretrainedModel.from_pretrained.__func__(
            cls,
            path_or_repo,
            config,
            module_map=rules,
            local=local,
            **kwargs,
        )

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

    def compute_causal_loss(
        self,
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

        model_output = self.model(
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

        from taktiny.trainer.loss.classification import cross_entropy_loss

        lm_weight = self._lm_weight()

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

            logits = self.compute_logits(hidden_chunk, lm_weight)
            logits = self._process_logits(logits)
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
