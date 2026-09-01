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
# WITHOUT WARRANTIES OR CONDITIONS OF tp.Any KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Common base modules for transformer architectures"""

from __future__ import annotations

import typing as tp
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
import qwix
from taktiny import nn
from taktiny.maestro.overture import ModelOutput, PretrainedModel
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
    GLUMBConv,
    JointAttention,
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
    ModelConfig,
    _validate_dtype_config,
)
from destiny.utils.sharding import create_sharding
from destiny.utils.typing import (
    ArrayLike,
    LogicalRules,
    PathLike,
)

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

@partial(
    jax.tree_util.register_dataclass,
    data_fields=['key_cache', 'value_cache', 'position_idx'],
    meta_fields=['is_causal', 'attention_kernel'],
)
@dataclass(frozen=True)
class TransformerContext:
    key_cache: jax.Array | None = None
    value_cache: jax.Array | None = None
    position_idx: jax.Array | None = None
    is_causal: bool | None = None
    attention_kernel: str = 'dot_product'


class ConditionalTransformerLayer(_ConditionalTransformerLayer):
    """Config-driven single-stream conditional transformer layer.

    The hidden stream is updated by modulated self-attention, read-only
    cross-attention context, and a modulated spatial feed-forward branch.
    Architectures may replace each compatible component while preserving this
    topology.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nn.Rngs,
        layer_idx: int | None = None,
        activation: str | Callable[[jax.Array], jax.Array] | None = None,
        ffn_dropout: float | None = None,
        attention_bias: bool | None = None,
        attention_out_bias: bool | None = None,
        cross_attention_bias: bool | None = None,
        mlp_bias: bool | None = None,
        pos_emb: nn.Module | None = None,
        cross_pos_emb: nn.Module | None = None,
        input_layernorm: nn.Module | type[nn.Module] = nn.LayerNorm,
        self_attention: nn.Module | type[nn.Module] = Attention,
        cross_attention: nn.Module | type[nn.Module] | None = Attention,
        cross_attention_layernorm: (
            nn.Module | type[nn.Module] | None
        ) = None,
        post_attention_layernorm: (
            nn.Module | type[nn.Module]
        ) = nn.LayerNorm,
        mlp: nn.Module | type[nn.Module] = GLUMBConv,
    ) -> None:
        hidden_size = _hidden_size(config)
        num_heads = _config_value(config, 'num_attention_heads')
        head_dim = _config_value(
            config,
            'head_dim',
            'attention_head_dim',
        )
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError('config must define num_attention_heads')

        context_size = _config_value(
            config,
            'cross_attention_dim',
            'context_dim',
            default=hidden_size,
        )
        cross_num_heads = _config_value(
            config,
            'num_cross_attention_heads',
            default=num_heads,
        )
        cross_head_dim = _config_value(
            config,
            'cross_attention_head_dim',
            default=head_dim,
        )
        intermediate_size = _config_value(config, 'intermediate_size')
        if intermediate_size is None:
            ratio = _config_value(
                config,
                'mlp_ratio',
                'expand_ratio',
                default=4.0,
            )
            intermediate_size = int(hidden_size * ratio)

        qk_norm = _config_value(config, 'qk_norm')
        dropout = _config_value(config, 'dropout', default=0.0)
        if ffn_dropout is None:
            ffn_dropout = _config_value(
                config,
                'ffn_dropout',
                default=0.0,
            )
        if attention_bias is None:
            attention_bias = bool(
                _config_value(config, 'attention_bias', default=False)
            )
        if attention_out_bias is None:
            attention_out_bias = bool(
                _config_value(config, 'attention_out_bias', default=True)
            )
        if cross_attention_bias is None:
            cross_attention_bias = bool(
                _config_value(config, 'cross_attention_bias', default=True)
            )
        if mlp_bias is None:
            mlp_bias = bool(
                _config_value(
                    config,
                    'glumbconv_bias',
                    'mlp_bias',
                    default=True,
                )
            )
        super().__init__(
            hidden_size=hidden_size,
            context_size=context_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            head_dim=head_dim,
            cross_num_heads=cross_num_heads,
            cross_head_dim=cross_head_dim,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
            activation=(
                _activation(config, default='gelu')
                if activation is None
                else activation
            ),
            norm_eps=_config_value(config, 'norm_eps', default=1e-6),
            norm_elementwise_affine=bool(
                _config_value(
                    config,
                    'norm_elementwise_affine',
                    default=False,
                )
            ),
            bias=attention_bias,
            attention_out_bias=attention_out_bias,
            cross_attention_bias=cross_attention_bias,
            mlp_bias=mlp_bias,
            use_qkv_norm=bool(
                _config_value(
                    config,
                    'use_qkv_norm',
                    default=qk_norm is not None,
                )
            ),
            qkv_norm_across_heads=(qk_norm == 'rms_norm_across_heads'),
            qkv_norm_eps=_config_value(
                config,
                'qkv_norm_eps',
                'norm_eps',
                default=1e-5,
            ),
            pos_emb=pos_emb,
            cross_pos_emb=cross_pos_emb,
            dtype=_model_dtype(config),
            rngs=rngs,
            shard_mode=_shard_mode(config),
            quant=_config_value(config, 'quant'),
            dot_general=_config_value(config, 'dot_general'),
            input_layernorm=input_layernorm,
            self_attention=self_attention,
            cross_attention=cross_attention,
            cross_attention_layernorm=cross_attention_layernorm,
            post_attention_layernorm=post_attention_layernorm,
            mlp=mlp,
        )
        self.layer_idx = layer_idx


class JointTransformerLayer(_JointTransformerLayer):
    """A config-driven, composable two-stream transformer layer.

    This is the joint-attention counterpart to
    :class:`TransformerDecoderLayer`. It translates architecture config values
    into the general two-stream implementation in :mod:`taktiny.layers`, while
    allowing architectures to replace each compatible component with a module
    subclass or initialized instance.

    ``layer_idx`` selects layer-dependent topology. The final layer defaults to
    context-pre-only behavior, and indices listed in
    ``config.dual_attention_layers`` receive a second hidden-stream attention.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nn.Rngs,
        layer_idx: int | None = None,
        conditioning_size: int | None = None,
        context_pre_only: bool | None = None,
        dual_attention: bool | None = None,
        project_conditioning: bool = True,
        context_project_conditioning: bool | None = None,
        use_qkv_norm: bool | None = None,
        qkv_norm_eps: float | None = None,
        context_first: bool = False,
        bias: bool | None = None,
        pos_emb: nn.Module | None = None,
        activation: str | Callable[[jax.Array], jax.Array] | None = None,
        input_layernorm: nn.Module | type[nn.Module] = AdaXNorm,
        context_input_layernorm: nn.Module | type[nn.Module] = AdaXNorm,
        joint_attention: nn.Module | type[nn.Module] = JointAttention,
        second_attention: nn.Module | type[nn.Module] = Attention,
        post_attention_layernorm: nn.Module | type[nn.Module] | None = None,
        context_post_attention_layernorm: (
            nn.Module | type[nn.Module] | None
        ) = None,
        mlp: nn.Module | type[nn.Module] = FeedForward,
        context_mlp: nn.Module | type[nn.Module] = FeedForward,
    ) -> None:
        num_heads = _config_value(config, 'num_attention_heads')
        head_dim = _config_value(config, 'head_dim', 'attention_head_dim')
        hidden_size = _config_value(config, 'hidden_size', 'inner_dim', 'dim')
        if hidden_size is None and num_heads is not None and head_dim is not None:
            hidden_size = num_heads * head_dim

        required = {
            'hidden size': hidden_size,
            'num_attention_heads': num_heads,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                'Missing required joint transformer config values: '
                + ', '.join(missing)
            )

        context_size = _config_value(
            config,
            'context_size',
            'context_dim',
            'caption_projection_dim',
            default=hidden_size,
        )
        intermediate_size = _config_value(config, 'intermediate_size')
        if intermediate_size is None:
            mlp_ratio = _config_value(config, 'mlp_ratio', default=4.0)
            intermediate_size = int(hidden_size * mlp_ratio)
        context_intermediate_size = _config_value(
            config,
            'context_intermediate_size',
            default=intermediate_size,
        )
        if conditioning_size is None:
            conditioning_size = _config_value(
                config,
                'conditioning_size',
                'time_embed_dim',
                default=hidden_size,
            )

        num_layers = _config_value(
            config,
            'num_hidden_layers',
            'num_layers',
        )
        if context_pre_only is None:
            configured_context_pre_only = _config_value(
                config,
                'context_pre_only',
            )
            context_pre_only = (
                bool(configured_context_pre_only)
                if configured_context_pre_only is not None
                else (
                    layer_idx is not None
                    and num_layers is not None
                    and layer_idx == num_layers - 1
                )
            )
        if dual_attention is None:
            dual_layers = _config_value(
                config,
                'dual_attention_layers',
                default=(),
            )
            dual_attention = layer_idx is not None and layer_idx in dual_layers

        qk_norm = _config_value(config, 'qk_norm')
        if use_qkv_norm is None:
            use_qkv_norm = bool(
                _config_value(
                    config,
                    'use_qkv_norm',
                    default=qk_norm is not None,
                )
            )
        if qkv_norm_eps is None:
            qkv_norm_eps = _config_value(
                config,
                'qkv_norm_eps',
                'norm_eps',
                default=1e-6,
            )
        if pos_emb is None:
            pos_emb = _config_value(
                config,
                'pos_emb',
                'position_embedding',
            )
        norm_type = _config_value(config, 'norm_type', default='layernorm')
        norm_type = str(norm_type).lower().replace('_', '')
        if norm_type in {'layer', 'layernormalization'}:
            norm_type = 'layernorm'
        elif norm_type in {'rms', 'rmsnormalization'}:
            norm_type = 'rmsnorm'

        super().__init__(
            hidden_size=hidden_size,
            context_size=context_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            context_intermediate_size=context_intermediate_size,
            conditioning_size=conditioning_size,
            head_dim=head_dim,
            dropout=_config_value(
                config,
                'dropout',
                'attention_dropout',
                default=0.0,
            ),
            activation=(
                activation
                if activation is not None
                else _config_value(
                    config,
                    'hidden_act',
                    'hidden_activation',
                    'activation',
                    default='gelu',
                )
            ),
            norm=norm_type,
            norm_eps=_config_value(
                config,
                'norm_eps',
                'layer_norm_eps',
                'rms_norm_eps',
                default=1e-6,
            ),
            context_pre_only=context_pre_only,
            dual_attention=dual_attention,
            bias=(
                bool(
                    _config_value(
                        config,
                        'projection_bias',
                        'attention_bias',
                        default=True,
                    )
                )
                if bias is None
                else bias
            ),
            use_qkv_norm=use_qkv_norm,
            qkv_norm_eps=qkv_norm_eps,
            context_first=context_first,
            scaling=_config_value(config, 'attention_scaling'),
            pos_emb=pos_emb,
            second_pos_emb=_config_value(config, 'second_pos_emb'),
            dtype=_model_dtype(config),
            rngs=rngs,
            shard_mode=_shard_mode(config),
            quant=_config_value(config, 'quant'),
            dot_general=_config_value(config, 'dot_general'),
            project_conditioning=project_conditioning,
            context_project_conditioning=context_project_conditioning,
            input_layernorm=input_layernorm,
            context_input_layernorm=context_input_layernorm,
            joint_attention=joint_attention,
            second_attention=second_attention,
            post_attention_layernorm=post_attention_layernorm,
            context_post_attention_layernorm=(
                context_post_attention_layernorm
            ),
            mlp=mlp,
            context_mlp=context_mlp,
        )
        self.layer_idx = layer_idx


class GatedParallelTransformerLayer(_GatedParallelTransformerLayer):
    """A config-driven transformer layer with parallel attention and FFN."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nn.Rngs,
        parallel_path: nn.Module | type[nn.Module],
        layer_idx: int | None = None,
        conditioning_size: int | None = None,
        project_conditioning: bool = True,
        pos_emb: nn.Module | None = None,
        activation: str | Callable[[jax.Array], jax.Array] | None = None,
        use_qkv_norm: bool | None = None,
        bias: bool | None = None,
        input_layernorm: nn.Module | type[nn.Module] = AdaXNorm,
    ) -> None:
        num_heads = _config_value(config, 'num_attention_heads')
        head_dim = _config_value(config, 'head_dim', 'attention_head_dim')
        hidden_size = _config_value(config, 'hidden_size', 'inner_dim', 'dim')
        if hidden_size is None and num_heads is not None and head_dim is not None:
            hidden_size = num_heads * head_dim
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError('config must define a positive hidden size')
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError('config must define num_attention_heads')

        intermediate_size = _config_value(config, 'intermediate_size')
        if intermediate_size is None:
            intermediate_size = int(
                hidden_size
                * _config_value(config, 'mlp_ratio', default=4.0)
            )
        if conditioning_size is None:
            conditioning_size = _config_value(
                config,
                'conditioning_size',
                'time_embed_dim',
                default=hidden_size,
            )
        qk_norm = _config_value(config, 'qk_norm')
        if use_qkv_norm is None:
            use_qkv_norm = bool(
                _config_value(
                    config,
                    'use_qkv_norm',
                    default=qk_norm is not None,
                )
            )
        if pos_emb is None:
            pos_emb = _config_value(
                config,
                'pos_emb',
                'position_embedding',
            )
        norm_type = _config_value(config, 'norm_type', default='layernorm')
        norm_type = str(norm_type).lower().replace('_', '')
        if norm_type in {'layer', 'layernormalization'}:
            norm_type = 'layernorm'
        elif norm_type in {'rms', 'rmsnormalization'}:
            norm_type = 'rmsnorm'

        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            conditioning_size=conditioning_size,
            parallel_path=parallel_path,
            head_dim=head_dim,
            dropout=_config_value(
                config,
                'dropout',
                'attention_dropout',
                default=0.0,
            ),
            activation=(
                activation
                if activation is not None
                else _activation(config, default='gelu')
            ),
            norm=norm_type,
            norm_eps=_config_value(
                config,
                'eps',
                'norm_eps',
                'layer_norm_eps',
                default=1e-6,
            ),
            bias=(
                bool(
                    _config_value(
                        config,
                        'projection_bias',
                        'attention_bias',
                        default=False,
                    )
                )
                if bias is None
                else bias
            ),
            pos_emb=pos_emb,
            dtype=_model_dtype(config),
            rngs=rngs,
            shard_mode=_shard_mode(config),
            quant=_config_value(config, 'quant'),
            dot_general=_config_value(config, 'dot_general'),
            project_conditioning=project_conditioning,
            use_qkv_norm=use_qkv_norm,
            scaling=_config_value(config, 'attention_scaling'),
            input_layernorm=input_layernorm,
        )
        self.use_qkv_norm = use_qkv_norm
        self.layer_idx = layer_idx


DiffusionComponent: tp.TypeAlias = nn.Module | type[nn.Module]


class DiffusionTransformerModel(PretrainedModel):
    """Composable patch-based diffusion transformer backbone.

    The class owns the model-level mechanics shared by denoising transformers:
    component construction, repeated transformer layers, rematerialization,
    optional layer skipping, ControlNet residual routing, and patch
    reconstruction. Concrete Maestro architectures select compatible
    component types while Cosette layer classes define their mathematics.

    Component types follow role-specific constructor contracts. Initialized
    module instances may be supplied when an architecture requires a different
    constructor. Subclasses can override the preparation and finalization
    hooks without reimplementing layer iteration. ``stack_type='stack'``
    stores layers in an ``nn.SeqStack``. Depth-dependent topologies are partitioned
    into maximal contiguous stack-compatible groups while preserving one
    carry and the original execution order.
    """

    _default_sharding_rules = (
        ('batch', 'fsdp'),
        ('height', None),
        ('width', None),
        ('sequence', None),
        ('embed', None),
        ('context_embed', None),
        ('heads', 'tp'),
        ('head_dim', None),
        ('mlp', 'tp'),
        ('channel', None),
    )
    _component_names = frozenset(
        {
            'patch_embedding',
            'condition_embedding',
            'context_embedding',
            'output_norm',
            'output_projection',
        }
    )

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nn.Rngs,
        transformer_layer: type[nn.Module],
        patch_embedding: DiffusionComponent,
        condition_embedding: DiffusionComponent,
        context_embedding: DiffusionComponent | None,
        output_norm: DiffusionComponent | None,
        output_projection: DiffusionComponent,
        component_kwargs: Mapping[str, Mapping[str, tp.Any]] | None = None,
        mesh: jax.sharding.Mesh | None = None,
        sharding_rules: LogicalRules | None = None,
        stack_type: tp.Literal['list', 'stack'] | None = None,
    ) -> None:
        if (
            not isinstance(transformer_layer, type)
            or not issubclass(transformer_layer, nn.Module)
        ):
            raise TypeError('transformer_layer must be an nn.Module subclass')
        stack_type = stack_type or 'stack'
        if stack_type not in {'list', 'stack'}:
            raise ValueError("stack_type must be 'list' or 'stack'")

        self.config = config
        self.num_layers = _positive_int(
            _config_value(config, 'num_layers', 'num_hidden_layers'),
            'num_layers',
        )
        self.num_attention_heads = _positive_int(
            _config_value(config, 'num_attention_heads'),
            'num_attention_heads',
        )
        self.attention_head_dim = _positive_int(
            _config_value(config, 'attention_head_dim', 'head_dim'),
            'attention_head_dim',
        )
        self.inner_dim = self.num_attention_heads * self.attention_head_dim
        self.in_channels = _positive_int(config.in_channels, 'in_channels')
        self.out_channels = _positive_int(
            _config_value(config, 'out_channels', default=self.in_channels),
            'out_channels',
        )
        self.patch_size = self._spatial_pair(
            _config_value(config, 'patch_size', default=2),
            'patch_size',
        )
        self.shard_mode = _shard_mode(config)
        self.dtype = _model_dtype(config)
        self.quant = _config_value(config, 'quant')
        self.dot_general = _config_value(config, 'dot_general')

        options = self._component_options(component_kwargs)
        sample_size = _config_value(config, 'sample_size', default=128)
        self.patch_embedding = self._instantiate_component(
            patch_embedding,
            name='patch_embedding',
            options={
                'sample_size': sample_size,
                'patch_size': self.patch_size,
                'in_channels': self.in_channels,
                'embedding_dim': self.inner_dim,
                'dtype': self.dtype,
                'rngs': rngs,
                'shard_mode': self.shard_mode,
                **options['patch_embedding'],
            },
        )

        if isinstance(condition_embedding, nn.Module):
            self.condition_embedding = condition_embedding
        else:
            pooled_projection_dim = _positive_int(
                config.pooled_projection_dim,
                'pooled_projection_dim',
            )
            self.condition_embedding = self._instantiate_component(
                condition_embedding,
                name='condition_embedding',
                options={
                    'embedding_dim': self.inner_dim,
                    'pooled_projection_dim': pooled_projection_dim,
                    'dtype': self.dtype,
                    'rngs': rngs,
                    'quant': self.quant,
                    'dot_general': self.dot_general,
                    'shard_mode': self.shard_mode,
                    **options['condition_embedding'],
                },
            )

        if context_embedding is None:
            self.context_embedding = None
        elif isinstance(context_embedding, nn.Module):
            self.context_embedding = context_embedding
        else:
            joint_attention_dim = _positive_int(
                config.joint_attention_dim,
                'joint_attention_dim',
            )
            caption_projection_dim = _positive_int(
                _config_value(
                    config,
                    'caption_projection_dim',
                    default=self.inner_dim,
                ),
                'caption_projection_dim',
            )
            self.context_embedding = self._instantiate_component(
                context_embedding,
                name='context_embedding',
                options={
                    'in_features': joint_attention_dim,
                    'out_features': caption_projection_dim,
                    'bias': True,
                    'dtype': self.dtype,
                    'rngs': rngs,
                    'quant': self.quant,
                    'dot_general': self.dot_general,
                    'axis_names': ('joint_embed', 'context_embed'),
                    'shard_mode': self.shard_mode,
                    **options['context_embedding'],
                },
            )

        layers = [
            transformer_layer(config, rngs=rngs, layer_idx=index)
            for index in range(self.num_layers)
        ]
        self.stack_type = stack_type
        if stack_type == 'list':
            self.layers = nn.List(layers)
        else:
            for layer in layers:
                if hasattr(layer, 'layer_idx'):
                    layer.layer_idx = None
            self.layers = nn.SeqStack(layers)

        if output_norm is None:
            self.output_norm = None
        else:
            self.output_norm = self._instantiate_component(
                output_norm,
                name='output_norm',
                options={
                    'embedding_dim': self.inner_dim,
                    'out_dim': 2 * self.inner_dim,
                    'norm': 'layernorm',
                    'eps': 1e-6,
                    'activation': 'silu',
                    'bias': True,
                    'dtype': self.dtype,
                    'rngs': rngs,
                    'quant': self.quant,
                    'dot_general': self.dot_general,
                    'axis_names': ('conditioning', 'output_modulation'),
                    'shard_mode': self.shard_mode,
                    **options['output_norm'],
                },
            )

        self.output_projection = self._instantiate_component(
            output_projection,
            name='output_projection',
            options={
                'in_features': self.inner_dim,
                'out_features': (
                    self.patch_size[0]
                    * self.patch_size[1]
                    * self.out_channels
                ),
                'bias': True,
                'dtype': self.dtype,
                'rngs': rngs,
                'quant': self.quant,
                'dot_general': self.dot_general,
                'axis_names': ('embed', 'patch'),
                'shard_mode': self.shard_mode,
                **options['output_projection'],
            },
        )

        if sharding_rules is None:
            sharding_rules = self._default_sharding_rules
        self.output_sharding = None
        if mesh is not None and self.shard_mode == ShardMode.EXPLICIT:
            self.output_sharding = create_sharding(
                mesh,
                ('batch', 'height', 'width', 'channel'),
                rules=sharding_rules,
            )
        self.remat = False

    @classmethod
    def _instantiate_component(
        cls,
        component: DiffusionComponent,
        *,
        name: str,
        options: Mapping[str, tp.Any],
    ) -> nn.Module:
        if isinstance(component, nn.Module):
            return component
        if not isinstance(component, type) or not issubclass(component, nn.Module):
            raise TypeError(f'{name} must be an nn.Module subclass or instance')
        return component(**options)

    @classmethod
    def _component_options(
        cls,
        component_kwargs: Mapping[str, Mapping[str, tp.Any]] | None,
    ) -> dict[str, dict[str, tp.Any]]:
        supplied = dict(component_kwargs or {})
        unknown = supplied.keys() - cls._component_names
        if unknown:
            raise ValueError(
                'unknown diffusion component options: '
                + ', '.join(sorted(unknown))
            )
        result = {name: {} for name in cls._component_names}
        for name, values in supplied.items():
            if not isinstance(values, Mapping):
                raise TypeError(f'component_kwargs[{name!r}] must be a mapping')
            result[name] = dict(values)
        return result

    @staticmethod
    def _spatial_pair(
        value: int | Sequence[int],
        name: str,
    ) -> tuple[int, int]:
        values = (value, value) if isinstance(value, int) else tuple(value)
        if len(values) != 2:
            raise ValueError(f'{name} must contain exactly two dimensions')
        for index, size in enumerate(values):
            _positive_int(size, f'{name}[{index}]')
        return tp.cast(tuple[int, int], values)

    def enable_remat(self) -> None:
        """Rematerialize transformer blocks during differentiation."""
        self.remat = True

    def disable_remat(self) -> None:
        """Disable transformer-block rematerialization."""
        self.remat = False

    def _prepare_conditioning(
        self,
        timestep: jax.Array,
        pooled_projection: jax.Array,
    ) -> jax.Array:
        return self.condition_embedding(timestep, pooled_projection)

    def _prepare_context(self, encoder_hidden_states: jax.Array) -> jax.Array:
        if self.context_embedding is None:
            return encoder_hidden_states
        return self.context_embedding(encoder_hidden_states)

    def _call_transformer_layer(
        self,
        layer: nn.Module,
        hidden_states: jax.Array,
        encoder_hidden_states: jax.Array,
        conditioning: jax.Array,
        **layer_kwargs: tp.Any,
    ) -> tuple[jax.Array, jax.Array]:
        result = layer(
            hidden_states,
            encoder_hidden_states,
            conditioning,
            **layer_kwargs,
        )
        if isinstance(result, tuple):
            if len(result) != 2:
                raise ValueError(
                    'a diffusion transformer layer tuple must contain '
                    '(context, hidden_states)'
                )
            next_context, next_hidden = result
            if next_context is None:
                next_context = encoder_hidden_states
            return next_context, next_hidden
        return encoder_hidden_states, result

    @staticmethod
    def _validate_skip_layers(
        skip_layers: Sequence[int] | None,
        num_layers: int,
    ) -> frozenset[int]:
        if skip_layers is None:
            return frozenset()
        result = frozenset(skip_layers)
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= num_layers
            for index in result
        ):
            raise ValueError('skip_layers contains an invalid layer index')
        return result

    def _apply_transformer_layers(
        self,
        hidden_states: jax.Array,
        encoder_hidden_states: jax.Array,
        conditioning: jax.Array,
        *,
        control_residuals: Sequence[jax.Array] | None,
        skip_layers: Sequence[int] | None,
        layer_kwargs: Mapping[str, tp.Any],
    ) -> tuple[jax.Array, jax.Array]:
        skipped = self._validate_skip_layers(skip_layers, self.num_layers)
        controls = () if control_residuals is None else tuple(control_residuals)
        if control_residuals is not None and not controls:
            raise ValueError('control_residuals must not be empty')

        call_layer = self._call_transformer_layer
        if self.remat:
            call_layer = jax.checkpoint(
                call_layer,
                prevent_cse=self.stack_type == 'list',
            )

        if self.stack_type == 'stack':
            control_stack = None
            if controls:
                for control in controls:
                    if control.shape != hidden_states.shape:
                        raise ValueError(
                            'each control residual must match the hidden token '
                            f'shape {hidden_states.shape}; got {control.shape}'
                        )
                control_stack = jnp.stack(
                    tuple(jnp.asarray(control) for control in controls)
                )

            skipped_indices = None
            if skipped:
                skipped_indices = jnp.asarray(
                    tuple(sorted(skipped)),
                    dtype=jnp.int32,
                )

            def apply_layer(
                layer: nn.Module,
                carry: tuple[jax.Array, jax.Array, jax.Array],
            ) -> tuple[
                tuple[jax.Array, jax.Array, jax.Array],
                None,
            ]:
                context, hidden, layer_index = carry

                def apply(
                    operands: tuple[jax.Array, jax.Array],
                ) -> tuple[jax.Array, jax.Array]:
                    current_context, current_hidden = operands
                    return call_layer(
                        layer,
                        current_hidden,
                        current_context,
                        conditioning,
                        **layer_kwargs,
                    )

                if skipped_indices is None:
                    context, hidden = apply((context, hidden))
                else:
                    should_skip = jnp.any(layer_index == skipped_indices)
                    context, hidden = jax.lax.cond(
                        should_skip,
                        lambda operands: operands,
                        apply,
                        (context, hidden),
                    )

                if (
                    control_stack is not None
                    and not getattr(layer, 'context_pre_only', False)
                ):
                    control_index = jnp.minimum(
                        layer_index * len(controls) // self.num_layers,
                        len(controls) - 1,
                    )
                    hidden = hidden + jax.lax.dynamic_index_in_dim(
                        control_stack,
                        control_index,
                        axis=0,
                        keepdims=False,
                    )

                return (context, hidden, layer_index + 1), None

            (encoder_hidden_states, hidden_states, _), _ = self.layers(
                apply_layer,
                (
                    encoder_hidden_states,
                    hidden_states,
                    jnp.asarray(0, dtype=jnp.int32),
                ),
            )
            return encoder_hidden_states, hidden_states

        for index, layer in enumerate(self.layers):
            if index not in skipped:
                encoder_hidden_states, hidden_states = call_layer(
                    layer,
                    hidden_states,
                    encoder_hidden_states,
                    conditioning,
                    **layer_kwargs,
                )

            if controls and not getattr(layer, 'context_pre_only', False):
                control_index = min(
                    int(index * len(controls) / self.num_layers),
                    len(controls) - 1,
                )
                control = jnp.asarray(controls[control_index])
                if control.shape != hidden_states.shape:
                    raise ValueError(
                        'each control residual must match the hidden token '
                        f'shape {hidden_states.shape}; got {control.shape}'
                    )
                hidden_states = hidden_states + control
        return encoder_hidden_states, hidden_states

    def _finalize_tokens(
        self,
        hidden_states: jax.Array,
        conditioning: jax.Array,
    ) -> jax.Array:
        if self.output_norm is not None:
            normalized = self.output_norm(hidden_states, conditioning)
            if isinstance(normalized, tuple):
                if len(normalized) != 2:
                    raise ValueError(
                        'output_norm tuple must contain normalized activations '
                        'and modulation'
                    )
                hidden_states, modulation = normalized
                if modulation.shape[-1] != 2 * hidden_states.shape[-1]:
                    raise ValueError(
                        'output modulation must contain one scale and shift '
                        'value per hidden feature'
                    )
                scale, shift = jnp.split(modulation, 2, axis=-1)
                hidden_states = hidden_states * (1.0 + scale[:, None, :])
                hidden_states = hidden_states + shift[:, None, :]
            else:
                hidden_states = normalized
        return self.output_projection(hidden_states)

    @staticmethod
    def _unpatchify(
        tokens: jax.Array,
        grid_size: tuple[int, int],
        patch_size: tuple[int, int],
        out_channels: int,
    ) -> jax.Array:
        batch = tokens.shape[0]
        grid_height, grid_width = grid_size
        patch_height, patch_width = patch_size
        tokens = tokens.reshape(
            batch,
            grid_height,
            grid_width,
            patch_height,
            patch_width,
            out_channels,
        )
        tokens = jnp.transpose(tokens, (0, 1, 3, 2, 4, 5))
        return tokens.reshape(
            batch,
            grid_height * patch_height,
            grid_width * patch_width,
            out_channels,
        )

    def __call__(
        self,
        x: jax.Array,
        encoder_hidden_states: jax.Array,
        pooled_projection: jax.Array,
        timestep: jax.Array,
        *,
        control_residuals: Sequence[jax.Array] | None = None,
        layer_kwargs: Mapping[str, tp.Any] | None = None,
        skip_layers: Sequence[int] | None = None,
        return_dict: bool = True,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array | tuple[jax.Array]:
        x = jnp.asarray(x)
        encoder_hidden_states = jnp.asarray(encoder_hidden_states)
        pooled_projection = jnp.asarray(pooled_projection)
        timestep = jnp.asarray(timestep)
        if x.ndim != 4 or x.shape[-1] != self.in_channels:
            raise ValueError(
                'x must have shape [batch, height, width, in_channels]'
            )
        if not jnp.issubdtype(x.dtype, jnp.floating):
            raise TypeError('x must have a floating-point dtype')
        if encoder_hidden_states.ndim != 3:
            raise ValueError(
                'encoder_hidden_states must have shape '
                '[batch, sequence, hidden]'
            )
        if pooled_projection.ndim != 2:
            raise ValueError(
                'pooled_projection must have shape [batch, hidden]'
            )
        batch = x.shape[0]
        if (
            encoder_hidden_states.shape[0] != batch
            or pooled_projection.shape[0] != batch
        ):
            raise ValueError('all inputs must share the same batch size')
        if timestep.ndim == 0:
            timestep = jnp.broadcast_to(timestep, (batch,))
        if timestep.shape != (batch,):
            raise ValueError('timestep must be a scalar or have shape [batch]')

        height, width = x.shape[1:3]
        patch_height, patch_width = self.patch_size
        if height % patch_height or width % patch_width:
            raise ValueError('latent dimensions must be divisible by patch_size')
        grid_size = (height // patch_height, width // patch_width)

        hidden_states = self.patch_embedding(x)
        conditioning = self._prepare_conditioning(
            timestep,
            pooled_projection,
        )
        encoder_hidden_states = self._prepare_context(
            encoder_hidden_states,
        )
        _, hidden_states = self._apply_transformer_layers(
            hidden_states,
            encoder_hidden_states,
            conditioning,
            control_residuals=control_residuals,
            skip_layers=skip_layers,
            layer_kwargs=dict(layer_kwargs or {}),
        )
        output = self._unpatchify(
            self._finalize_tokens(hidden_states, conditioning),
            grid_size,
            self.patch_size,
            self.out_channels,
        )
        target_sharding = (
            self.output_sharding if out_sharding is None else out_sharding
        )
        output = _constrain(output, target_sharding, self.shard_mode)
        return output if return_dict else (output,)


class TransformerMultimodalLM(PretrainedModel):
    """Unified base class for Multimodal Language Models (Conditional Generation).

    Coordinates a text language backbone (e.g., ``TransformerCausalLM``), an optional
    vision encoder/tower, an optional audio encoder/tower, and multimodal projectors.
    Supports text generation conditioned on text, image, video, and audio inputs.

    Args:
        config: Model configuration containing text, vision, and projector settings.
        rngs: Random number generator for weight initialization.
        language_model: Language model instance or module class.
        decoder: Decoder layer module class used to build a ``TransformerCausalLM``
            if ``language_model`` is not directly provided.
        vision_tower: Vision encoder instance or module class.
        multi_modal_projector: Multimodal projection layer instance or module class.
        audio_tower: Audio encoder instance or module class.
        audio_projector: Audio projection layer instance or module class.
        image_token_id: Token ID representing image placeholders in ``input_ids``.
        video_token_id: Token ID representing video placeholders in ``input_ids``.
        audio_token_id: Token ID representing audio placeholders in ``input_ids``.
        mesh: JAX device mesh for explicit sharding.
        sharding_rules: Logical-to-mesh axis mapping rules.
    """

    _default_sharding_rules = [
        ('vocab', 'tp'),
        ('embed', None),
        ('heads', 'tp'),
        ('kv_heads', 'tp'),
        ('head_dim', None),
        ('mlp', 'tp'),
        ('batch', 'fsdp'),
        ('sequence', None),
    ]

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        rngs: nn.Rngs | None = None,
        language_model: nn.Module | type[nn.Module] | None = None,
        vision_tower: nn.Module | type[nn.Module] | None = None,
        multi_modal_projector: nn.Module | type[nn.Module] | None = None,
        audio_tower: nn.Module | type[nn.Module] | None = None,
        audio_projector: nn.Module | type[nn.Module] | None = None,
        mesh: jax.sharding.Mesh | None = None,
        sharding_rules: LogicalRules | None = None,
        **kwargs: tp.Any,
    ) -> None:
        if rngs is None:
            rngs = nn.Rngs(42)

        self.config = config
        self.dtype = (
            getattr(config, 'torch_dtype', None)
            or getattr(config, 'dtype', None)
            or 'bfloat16'
        )
        self.shard_mode = getattr(config, 'shard_mode', ShardMode.AUTO)

        # 1. Text Language Model Backbone
        if language_model is not None:
            if isinstance(language_model, type) and issubclass(language_model, nn.Module):
                self.language_model = language_model(
                    config=config,
                    rngs=rngs,
                    mesh=mesh,
                    sharding_rules=sharding_rules,
                    **kwargs,
                )
            else:
                self.language_model = language_model
        else:
            self.language_model = TransformerCausalLM(
                config=config,
                rngs=rngs,
                mesh=mesh,
                sharding_rules=sharding_rules,
                **kwargs,
            )

        # 2. Vision Encoder / Tower
        if isinstance(vision_tower, type) and issubclass(vision_tower, nn.Module):
            self.vision_tower = vision_tower(config=config, rngs=rngs)
        else:
            self.vision_tower = vision_tower

        # 3. Multimodal Vision Projector
        if isinstance(multi_modal_projector, type) and issubclass(multi_modal_projector, nn.Module):
            self.multi_modal_projector = multi_modal_projector(config=config, rngs=rngs)
        else:
            self.multi_modal_projector = multi_modal_projector

        # 4. Audio Tower & Audio Projector
        if isinstance(audio_tower, type) and issubclass(audio_tower, nn.Module):
            self.audio_tower = audio_tower(config=config, rngs=rngs)
        else:
            self.audio_tower = audio_tower

        if isinstance(audio_projector, type) and issubclass(audio_projector, nn.Module):
            self.audio_projector = audio_projector(config=config, rngs=rngs)
        else:
            self.audio_projector = audio_projector

        # 5. Media Token IDs
        self.image_token_id = (
            kwargs.get('image_token_id')
            or getattr(config, 'image_token_id', None)
            or getattr(getattr(config, 'vision_config', None), 'image_token_id', None)
        )
        self.video_token_id = kwargs.get('video_token_id') or getattr(config, 'video_token_id', None)
        self.audio_token_id = kwargs.get('audio_token_id') or getattr(config, 'audio_token_id', None)

    def get_input_embeddings(self) -> nn.Module | None:
        if self.language_model is not None and hasattr(self.language_model, 'get_input_embeddings'):
            return self.language_model.get_input_embeddings()
        elif self.language_model is not None and hasattr(self.language_model, 'model'):
            return self.language_model.model.embed_tokens
        elif hasattr(self, 'embed_tokens'):
            return self.embed_tokens
        return None

    def get_output_embeddings(self) -> nn.Module | None:
        if self.language_model is not None and hasattr(self.language_model, 'get_output_embeddings'):
            return self.language_model.get_output_embeddings()
        elif self.language_model is not None and hasattr(self.language_model, 'lm_head'):
            return self.language_model.lm_head
        elif hasattr(self, 'lm_head'):
            return self.lm_head
        return None

    def get_language_model(self) -> nn.Module | None:
        return self.language_model

    def get_vision_tower(self) -> nn.Module | None:
        return self.vision_tower

    def get_multi_modal_projector(self) -> nn.Module | None:
        return self.multi_modal_projector

    def enable_remat(self) -> None:
        if self.language_model is not None and hasattr(self.language_model, 'enable_remat'):
            self.language_model.enable_remat()
        if self.vision_tower is not None and hasattr(self.vision_tower, 'enable_remat'):
            self.vision_tower.enable_remat()

    def encode_vision(self, pixel_values: jax.Array, **kwargs: tp.Any) -> jax.Array:
        """Encode vision inputs and project features into hidden dimension."""
        if self.vision_tower is None:
            raise ValueError("vision_tower is not configured for this model")
        vision_outputs = self.vision_tower(pixel_values, **kwargs)
        if isinstance(vision_outputs, tuple):
            vision_features = vision_outputs[0]
        else:
            vision_features = vision_outputs

        if self.multi_modal_projector is not None:
            vision_features = self.multi_modal_projector(vision_features)
        return vision_features

    def encode_audio(self, input_features: jax.Array, **kwargs: tp.Any) -> jax.Array:
        """Encode audio inputs and project features into hidden dimension."""
        if self.audio_tower is None:
            raise ValueError("audio_tower is not configured for this model")
        audio_outputs = self.audio_tower(input_features, **kwargs)
        if isinstance(audio_outputs, tuple):
            audio_features = audio_outputs[0]
        else:
            audio_features = audio_outputs

        if self.audio_projector is not None:
            audio_features = self.audio_projector(audio_features)
        return audio_features

    def merge_multimodal_embeddings(
        self,
        input_ids: jax.Array,
        inputs_embeds: jax.Array,
        multimodal_features: jax.Array,
        media_token_id: int,
    ) -> jax.Array:
        """Merge/splice multimodal feature vectors into inputs_embeds at media_token_id positions."""
        if media_token_id is None or multimodal_features is None:
            return inputs_embeds

        mask = (input_ids == media_token_id)
        flat_features = multimodal_features.reshape(-1, multimodal_features.shape[-1])
        target_shape = inputs_embeds.shape
        reshaped_features = flat_features[:target_shape[0] * target_shape[1]].reshape(target_shape)
        return jnp.where(mask[..., None], reshaped_features, inputs_embeds)

    def __call__(
        self,
        input_ids: jax.Array | None = None,
        pixel_values: jax.Array | None = None,
        input_features: jax.Array | None = None,
        pixel_attention_mask: jax.Array | None = None,
        image_sizes: jax.Array | None = None,
        inputs_embeds: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
        ctx: TransformerContext | None = None,
        logits_to_keep: int | jax.Array = 0,
        image_token_id: int | None = None,
        audio_token_id: int | None = None,
        **kwargs: tp.Any,
    ) -> tuple[jax.Array, TransformerContext | None]:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You should specify either input_ids or inputs_embeds")

            embed_fn = self.get_input_embeddings()
            if embed_fn is not None:
                inputs_embeds = embed_fn(input_ids)
            else:
                inputs_embeds = input_ids

            # Process vision features
            if pixel_values is not None and self.vision_tower is not None:
                vision_features = self.encode_vision(pixel_values, **kwargs)
                img_tok_id = image_token_id or self.image_token_id
                if img_tok_id is not None:
                    inputs_embeds = self.merge_multimodal_embeddings(
                        input_ids, inputs_embeds, vision_features, img_tok_id
                    )

            # Process audio features
            if input_features is not None and self.audio_tower is not None:
                audio_features = self.encode_audio(input_features, **kwargs)
                aud_tok_id = audio_token_id or self.audio_token_id
                if aud_tok_id is not None:
                    inputs_embeds = self.merge_multimodal_embeddings(
                        input_ids, inputs_embeds, audio_features, aud_tok_id
                    )

        # Delegate to language model backbone
        if self.language_model is not None:
            return self.language_model(
                inputs_embeds,
                attention_mask=attention_mask,
                ctx=ctx,
                logits_to_keep=logits_to_keep,
            )
        elif hasattr(self, 'model'):
            x, new_cache = self.model(
                inputs_embeds,
                attention_mask=attention_mask,
                kv_cache=(ctx.key_cache, ctx.value_cache) if ctx and ctx.key_cache is not None else None,
                position_idx=ctx.position_idx if ctx else None,
                is_causal=ctx.is_causal if ctx else False,
            )
            out_embed = self.get_output_embeddings()
            logits = out_embed(x) if out_embed is not None else x
            if ctx is not None and new_cache is not None:
                ctx = replace(ctx, key_cache=new_cache[0], value_cache=new_cache[1])
            return logits, ctx
        else:
            raise NotImplementedError("Subclass should implement forward pass or provide language_model / model")

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: tp.Any,
        mesh: tp.Any = None,
        sharding_rules: tp.Any = None,
        local: bool = False,
        module_map: tp.List | None = None,
        **kwargs: tp.Any,
    ) -> tp.Any:
        kwargs = dict(kwargs)
        if 'config' in kwargs:
            config = kwargs.pop('config')
        else:
            config = ModelConfig.load_config(path_or_repo, local=local)
        if config is None:
            raise ValueError(
                f'Unable to load config from {path_or_repo!r} (local={local})'
            )

        rules = [
            ("model.language_model.", "language_model.model."),
            ("embed_tokens.weight", "embed_tokens.embedding"),
        ]

        if getattr(config, 'tie_word_embeddings', False):
            rules.append(
                ('lm_head.weight', 'language_model.model.embed_tokens.embedding')
            )

        if module_map is not None:
            rules.extend(module_map)

        return super().from_pretrained(
            path_or_repo,
            config=config,
            module_map=rules,
            local=local,
            mesh=mesh,
            sharding_rules=sharding_rules,
            **kwargs,
        )

    def generate(
        self,
        input_ids: jax.Array,
        max_new_tokens: int = 20,
        pixel_values: jax.Array | None = None,
        input_features: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        eos_token_id: int | list[int] | tuple[int, ...] | None = None,
        pad_token_id: int | None = None,
        seed: int = 42,
        streamer: tp.Any = None,
        attention_kernel: str | Mapping[str, str] = 'auto',
    ) -> jax.Array:
        """Autoregressively generate tokens conditioned on text and optional multimodal inputs."""
        if self.language_model is not None and hasattr(self.language_model, 'generate'):
            if pixel_values is not None or input_features is not None:
                logits, ctx = self(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    input_features=input_features,
                    attention_mask=attention_mask,
                )
            return self.language_model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                attention_mask=attention_mask,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                seed=seed,
                streamer=streamer,
                attention_kernel=attention_kernel,
            )
        else:
            raise NotImplementedError("Generation requires a configured language_model")





__all__ = [
    'ModelOutput',
    'PositionEmbedding',
    'PositionEmbeddings',
    'TransformerDecoderLayer',
    'TransformerModel',
    'TransformerCausalLM',
    'ConditionalTransformerLayer',
    'JointTransformerLayer',
    'GatedParallelTransformerLayer',
    'DiffusionTransformerModel',
    'TransformerContext',
    'TransformerMultimodalLM',
]
