# Copyright 2026 Shinapri
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import typing as tp

from taktiny.cosettes.transformers._ordinario import TransformerDecoderLayer
from taktiny import nn
from taktiny.cosettes.layers.ffn import MoeFFN
from taktiny.cosettes.layers import Attention

class GptOssDecoderLayer(TransformerDecoderLayer):
    def __init__(self, config: tp.Any, rngs: nn.Rngs, layer_idx: int | None=None) -> None:
        self.config = config
        super().__init__(
            config,
            rngs=rngs,
            layer_idx=layer_idx,
            input_layernorm=nn.RMSNorm,
            self_attn=Attention,
            post_attention_layernorm=nn.RMSNorm,
            mlp=MoeFFN,
        )
        
        import jax.numpy as jnp
        window_size = self.self_attn.window_size
        if window_size is None:
            window_size = getattr(config, 'max_position_embeddings', 4096)
        window_size = jnp.asarray(window_size, dtype=jnp.int32)
        self.sliding_window = window_size
        self.self_attn.window_size = window_size

    def _create_module(
        self,
        *,
        name: str,
        module_type: type[nn.Module] | nn.Module,
        **kwargs
    ) -> tuple[nn.Module, str]:
        if isinstance(module_type, type) and issubclass(module_type, MoeFFN):
            module = module_type(
                hidden_size=kwargs['hidden_size'],
                intermediate_size=kwargs['intermediate_size'],
                num_experts=self.config.num_local_experts,
                num_experts_per_tok=self.config.num_experts_per_tok,
                bias=kwargs['mlp_bias'],
                dtype=kwargs.get('dtype', None),
                rngs=kwargs['rngs'],
            )
            return module, 'residual'
        return TransformerDecoderLayer._create_module(name=name, module_type=module_type, **kwargs)

__all__ = [
    'GptOssDecoderLayer',
]
