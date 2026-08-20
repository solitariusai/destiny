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
"""Gemma architectures"""

from __future__ import annotations
from typing import Any
import typing as tp
import jax.numpy as jnp, jax

from taktiny.maestro.livret import repertoire
from taktiny.cosettes.transformers.ordinario import (
    TransformerCausalLM,
    TransformerMultimodalLM,
    TransformerContext
)
from taktiny.cosettes.transformers.gemma import (
    GemmaTextScaledWordEmbedding,
    GemmaRMSNorm,
    GemmaDecoderLayer,
    Gemma2DecoderLayer,
    Gemma3TextScaledWordEmbedding,
    Gemma3RMSNorm,
    Gemma3DecoderLayer,
    Gemma4DecoderLayer,
)
from taktiny import nn
from taktiny.maestro.config import ModelConfig
from taktiny.utils.typing import PathLike, LogicalRules


class Gemma(TransformerCausalLM):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        config.tie_word_embeddings = True
        super().__init__(
            config,
            embedding=GemmaTextScaledWordEmbedding,
            decoder=GemmaDecoderLayer,
            norm=GemmaRMSNorm,
            **kwargs
        )

    @classmethod
    def from_pretrained(
        cls, path_or_repo: Any,
        mesh: Any=None,
        sharding_rules: Any=None,
        local: bool=False,
        module_map: Any = None,
        **kwargs: Any
    ) -> Any:
        kwargs = dict(kwargs)
        extra_module_map = kwargs.pop('module_map', None)
        if 'config' in kwargs:
            config = kwargs.pop('config')
        else:
            config = ModelConfig.load_config(path_or_repo, local=local)
        if config is None:
            raise ValueError(
                f'Unable to load config from {path_or_repo!r} (local={local})'
            )
        config.tie_word_embeddings = True
        rules = list(module_map or [])
        if extra_module_map:
            rules.extend(extra_module_map)
        return super().from_pretrained(
            path_or_repo,
            mesh=mesh,
            sharding_rules=sharding_rules,
            local=local,
            config=config,
            module_map=rules if rules else None,
            **kwargs,
        )

class Gemma2(TransformerCausalLM):
    def __init__(self, config: ModelConfig, **kwargs: Any) -> None:
        config.tie_word_embeddings = True
        if getattr(config, 'layer_types', None) is None:
            config.layer_types = [
                (
                    'sliding_attention'
                    if (layer_idx + 1) % 2
                    else 'full_attention'
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        super().__init__(
            config,
            embedding=GemmaTextScaledWordEmbedding,
            decoder=Gemma2DecoderLayer,
            norm=GemmaRMSNorm,
            **kwargs
        )
        self.final_logit_softcapping = config.final_logit_softcapping

    def __call__(
        self,
        x: jax.Array,
        attention_mask: jax.Array | None = None,
        position_ids: jax.Array | None = None,
        ctx: TransformerContext | None = None,
        logits_to_keep: int = 0,
    ) -> tuple[Any, ...]:
        logits, ctx = super().__call__(
            x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            ctx=ctx,
            logits_to_keep=logits_to_keep,
        )
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = cap * jnp.tanh(logits / cap)
        return logits, ctx


class Gemma3(TransformerCausalLM):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        if bool(getattr(config, 'use_bidirectional_attention', False)):
            raise NotImplementedError(
                'Gemma3 bidirectional attention is not supported'
            )

        config.tie_word_embeddings      = True
        config.head_dim                 = config.head_dim or config.hidden_size // config.num_attention_heads
        config.num_key_value_heads      = config.num_key_value_heads or config.num_attention_heads
        config.rope_theta               = config.rope_theta or 1_000_000.0
        config.rope_local_base_freq     = config.rope_local_base_freq or 10_000.0
        config.query_pre_attn_scalar    = config.query_pre_attn_scalar or 256
        config.attention_bias           = config.attention_bias or False
        config.rms_norm_eps             = config.rms_norm_eps or 1e-6

        if config.layer_types is None:
            pattern = config.sliding_window_pattern or 6
            config.layer_types = [
                (
                    'sliding_attention'
                    if (layer_idx + 1) % pattern
                    else 'full_attention'
                )
                for layer_idx in range(config.num_hidden_layers)
            ]

        super().__init__(
            config,
            embedding=Gemma3TextScaledWordEmbedding,
            decoder=Gemma3DecoderLayer,
            norm=Gemma3RMSNorm,
            **kwargs
        )
        self.final_logit_softcapping = config.final_logit_softcapping

    def __call__(
        self,
        x: jax.Array,
        attention_mask: jax.Array | None = None,
        position_ids: jax.Array | None = None,
        ctx: TransformerContext | None = None,
        logits_to_keep: int = 0,
    ) -> tuple[Any, ...]:
        logits, ctx = super().__call__(
            x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            ctx=ctx,
            logits_to_keep=logits_to_keep,
        )
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = cap * jnp.tanh(logits / cap)
        return logits, ctx

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: PathLike,
        mesh: jax.sharding.Mesh | None = None,
        sharding_rules: LogicalRules | None = None,
        local: bool = False,
        module_map: Any = None,
        **kwargs: tp.Any,
    ) -> tp.Self:
        kwargs = dict(kwargs)
        extra_module_map = kwargs.pop('module_map', None)
        if 'config' in kwargs:
            config = kwargs.pop('config')
        else:
            config = ModelConfig.load_config(path_or_repo, local=local)
        if config is None:
            raise ValueError(
                f'Unable to load config from {path_or_repo!r} (local={local})'
            )
        config.tie_word_embeddings = True
        rules = list(module_map or [])
        if extra_module_map:
            rules.extend(extra_module_map)
        return super().from_pretrained(
            path_or_repo,
            mesh=mesh,
            sharding_rules=sharding_rules,
            local=local,
            config=config,
            module_map=rules if rules else None,
            **kwargs,
        )

# TODO: rewrite
class Gemma3ConditionalGeneration(TransformerMultimodalLM):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        language_model = TransformerCausalLM(
            config=config,
            decoder=Gemma3DecoderLayer,
            norm=nn.RMSNorm,
            **kwargs,
        )
        super().__init__(
            config,
            language_model=language_model,
            **kwargs,
        )

def _split_gemma4_gate_up(tensor: jax.Array) -> tuple[jax.Array, jax.Array]:
    # The checkpoint merges w1 and w3 along the intermediate axis:
    # (num_experts, 2 * moe_intermediate, hidden). Taktiny stores them as
    # (num_experts, hidden, moe_intermediate), so split on axis -2 and
    # transpose the trailing pair.
    dim = tensor.shape[-2] // 2
    w1 = tensor[..., :dim, :].transpose(0, 2, 1)
    w3 = tensor[..., dim:, :].transpose(0, 2, 1)
    return w1, w3


def _transpose_expert_weight(tensor: jax.Array) -> jax.Array:
    # (num_experts, hidden, moe_intermediate) -> (num_experts, moe_intermediate, hidden)
    return tensor.transpose(0, 2, 1)


_GEMMA4_MODULE_MAP = [
    ('experts.gate_up_proj', ['experts.w1', 'experts.w3'], _split_gemma4_gate_up),
    ('experts.down_proj', 'experts.w2', _transpose_expert_weight),
    ('router.proj.weight', 'experts.router.proj.weight'),
    ('router.scale', 'experts.router.scale'),
    ('router.per_expert_scale', 'experts.router.per_expert_scale'),
]


def _gemma4_module_map(
    config: ModelConfig,
    module_map: tp.Any,
    extra_module_map: tp.Any,
) -> list[tp.Any]:
    text_config = getattr(config, 'text_config', config)
    enable_moe = getattr(text_config, 'enable_moe_block', False)
    rules = [*_GEMMA4_MODULE_MAP]
    if not enable_moe:
        rules.append(
            (
                'post_feedforward_layernorm.weight',
                'post_feedforward_layernorm_1.weight',
            )
        )
    layer_types = getattr(text_config, 'layer_types', None)
    if layer_types is not None:
        for idx, ltype in enumerate(layer_types):
            if ltype in ('full_attention', 'full'):
                rules.append((
                    f'layers.{idx}.self_attn.k_proj.weight',
                    [f'layers.{idx}.self_attn.k_proj.weight', f'layers.{idx}.self_attn.v_proj.weight'],
                    lambda k: (k, k),
                ))
    rules.extend(module_map or [])
    if extra_module_map:
        rules.extend(extra_module_map)
    return rules


class Gemma4(TransformerCausalLM):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__(
            config,
            decoder=Gemma4DecoderLayer,
            norm=nn.RMSNorm,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: PathLike,
        mesh: jax.sharding.Mesh | None = None,
        sharding_rules: LogicalRules | None = None,
        local: bool = False,
        module_map: Any = None,
        **kwargs: tp.Any,
    ) -> tp.Self:
        kwargs = dict(kwargs)
        extra_module_map = kwargs.pop('module_map', None)
        if 'config' in kwargs:
            config = kwargs.pop('config')
        else:
            config = ModelConfig.load_config(path_or_repo, local=local)
        if config is None:
            raise ValueError(
                f'Unable to load config from {path_or_repo!r} (local={local})'
            )
        config.tie_word_embeddings = True
        rules = _gemma4_module_map(config, module_map, extra_module_map)
        return super().from_pretrained(
            path_or_repo,
            mesh=mesh,
            sharding_rules=sharding_rules,
            local=local,
            config=config,
            module_map=rules,
            **kwargs,
        )


class Gemma4Multimodal(TransformerMultimodalLM):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        language_model = Gemma4(
            config=config,
            **kwargs,
        )
        super().__init__(
            config,
            language_model=language_model,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: PathLike,
        mesh: jax.sharding.Mesh | None = None,
        sharding_rules: LogicalRules | None = None,
        local: bool = False,
        module_map: Any = None,
        **kwargs: tp.Any,
    ) -> tp.Self:
        kwargs = dict(kwargs)
        extra_module_map = kwargs.pop('module_map', None)
        if 'config' in kwargs:
            config = kwargs.pop('config')
        else:
            config = ModelConfig.load_config(path_or_repo, local=local)
        if config is None:
            raise ValueError(
                f'Unable to load config from {path_or_repo!r} (local={local})'
            )
        config.tie_word_embeddings = True
        rules = _gemma4_module_map(config, module_map, extra_module_map)
        return super().from_pretrained(
            path_or_repo,
            mesh=mesh,
            sharding_rules=sharding_rules,
            local=local,
            config=config,
            module_map=rules,
            **kwargs,
        )

# TODO
class Gemma4Unified(TransformerMultimodalLM):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        language_model = TransformerCausalLM(
            config=config,
            decoder=Gemma4DecoderLayer,
            norm=nn.RMSNorm,
            **kwargs,
        )
        super().__init__(
            config,
            language_model=language_model,
            **kwargs,
        )

# TODO
class DiffusionGemma(nn.Module):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        raise NotImplementedError(f'There is a plan to implement {self.__class__.__name__}.')

class_map = [
    ('GemmaForCausalLM', Gemma),
    ('Gemma2ForCausalLM', Gemma2),
    ('Gemma3ForCausalLM', Gemma3),
    ('Gemma3ForConditionalGeneration', Gemma3ConditionalGeneration),
    ('Gemma4ForCausalLM', Gemma4),
    ('Gemma4ForConditionalGeneration', Gemma4Multimodal),
    ('Gemma4UnifiedForConditionalGeneration', Gemma4Unified),
    ('DiffusionGemmaForBlockDiffusion', DiffusionGemma),
]

__all__ = []
for name, cls in class_map:
    repertoire.register(name, cls)
    __all__.append(cls.__name__)
