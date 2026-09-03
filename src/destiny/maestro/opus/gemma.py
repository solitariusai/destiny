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
"""Gemma implementations"""

from __future__ import annotations

from typing import Any, Self

import jax
import jax.numpy as jnp

from destiny.cosette.utils import ModuleMap
from destiny.maestro.transformer import TransformerCausalLM
from destiny.maestro.symphony.gemma import Gemma2Model, Gemma3TextModel, GemmaModel
from destiny.maestro.symphony.gemma.config import (
    Gemma2Config,
    Gemma3TextConfig,
    GemmaConfig,
)
from destiny.maestro.utils import destiny
from destiny.utils.typing import PathLike


# ┏━╸┏━╸┏┳┓┏┳┓┏━┓
# ┃╺┓┣╸ ┃┃┃┃┃┃┣━┫
# ┗━┛┗━╸╹ ╹╹ ╹╹ ╹
# ╻┏┳┓┏━┓╻  ┏━╸┏┳┓┏━╸┏┓╻╺┳╸┏━┓╺┳╸╻┏━┓┏┓╻┏━┓
# ┃┃┃┃┣━┛┃  ┣╸ ┃┃┃┣╸ ┃┗┫ ┃ ┣━┫ ┃ ┃┃ ┃┃┗┫┗━┓
# ╹╹ ╹╹  ┗━╸┗━╸╹ ╹┗━╸╹ ╹ ╹ ╹ ╹ ╹ ╹┗━┛╹ ╹┗━┛
@destiny
class GemmaForCausalLM(TransformerCausalLM):
    _model_type = GemmaModel
    _default_config = GemmaConfig


# ┏━╸┏━╸┏┳┓┏┳┓┏━┓
# ┃╺┓┣╸ ┃┃┃┃┃┃┏━┛
# ┗━┛┗━╸╹ ╹╹ ╹┗━╸
# ╻┏┳┓┏━┓╻  ┏━╸┏┳┓┏━╸┏┓╻╺┳╸┏━┓╺┳╸╻┏━┓┏┓╻┏━┓
# ┃┃┃┃┣━┛┃  ┣╸ ┃┃┃┣╸ ┃┗┫ ┃ ┣━┫ ┃ ┃┃ ┃┃┗┫┗━┓
# ╹╹ ╹╹  ┗━╸┗━╸╹ ╹┗━╸╹ ╹ ╹ ╹ ╹ ╹ ╹┗━┛╹ ╹┗━┛
@destiny
class Gemma2ForCausalLM(GemmaForCausalLM):
    _model_type = Gemma2Model
    _default_config = Gemma2Config
    _default_module_map = (
        GemmaForCausalLM._default_module_map.copy()
        .map('pre_feedforward_layernorm', 'norm3')
        .map('post_feedforward_layernorm', 'norm4')
    )

    def _process_logits(self, logits: jax.Array) -> jax.Array:
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = jnp.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
        return logits


# ┏━╸┏━╸┏┳┓┏┳┓┏━┓┏━┓
# ┃╺┓┣╸ ┃┃┃┃┃┃┣━┫╺━┫
# ┗━┛┗━╸╹ ╹╹ ╹╹ ╹┗━┛
# ╻┏┳┓┏━┓╻  ┏━╸┏┳┓┏━╸┏┓╻╺┳╸┏━┓╺┳╸╻┏━┓┏┓╻┏━┓
# ┃┃┃┃┣━┛┃  ┣╸ ┃┃┃┣╸ ┃┗┫ ┃ ┣━┫ ┃ ┃┃ ┃┃┗┫┗━┓
# ╹╹ ╹╹  ┗━╸┗━╸╹ ╹┗━╸╹ ╹ ╹ ╹ ╹ ╹ ╹┗━┛╹ ╹┗━┛
@destiny
class Gemma3ForCausalLM(Gemma2ForCausalLM):
    _model_type = Gemma3TextModel
    _default_module_map = (
        ModuleMap
        .map('model.language_model.', 'model.')
        .extend(Gemma2ForCausalLM._default_module_map)
    )
    _default_config = Gemma3TextConfig

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: PathLike,
        *,
        config: Gemma3TextConfig | None = None,
        local: bool = False,
        **kwargs: Any,
    ) -> Self:
        if config is None:
            config = Gemma3TextConfig.load_config(path_or_repo, local=local)
        if config is None:
            raise ValueError(
                f'Unable to load config from {path_or_repo!r} (local={local})'
            )
        text_config = vars(config).get('text_config')
        if text_config is not None:
            config = text_config
        return super().from_pretrained(
            path_or_repo,
            config=config,
            local=local,
            **kwargs,
        )


# TODO
# ┏━╸┏━╸┏┳┓┏┳┓┏━┓┏━┓┏┓╻
# ┃╺┓┣╸ ┃┃┃┃┃┃┣━┫╺━┫┃┗┫
# ┗━┛┗━╸╹ ╹╹ ╹╹ ╹┗━┛╹ ╹
# ╻┏┳┓┏━┓╻  ┏━╸┏┳┓┏━╸┏┓╻╺┳╸┏━┓╺┳╸╻┏━┓┏┓╻┏━┓
# ┃┃┃┃┣━┛┃  ┣╸ ┃┃┃┣╸ ┃┗┫ ┃ ┣━┫ ┃ ┃┃ ┃┃┗┫┗━┓
# ╹╹ ╹╹  ┗━╸┗━╸╹ ╹┗━╸╹ ╹ ╹ ╹ ╹ ╹ ╹┗━┛╹ ╹┗━┛


# TODO
# ┏━╸┏━╸┏┳┓┏┳┓┏━┓╻ ╻
# ┃╺┓┣╸ ┃┃┃┃┃┃┣━┫┗━┫
# ┗━┛┗━╸╹ ╹╹ ╹╹ ╹  ╹
# ╻┏┳┓┏━┓╻  ┏━╸┏┳┓┏━╸┏┓╻╺┳╸┏━┓╺┳╸╻┏━┓┏┓╻┏━┓
# ┃┃┃┃┣━┛┃  ┣╸ ┃┃┃┣╸ ┃┗┫ ┃ ┣━┫ ┃ ┃┃ ┃┃┗┫┗━┓
# ╹╹ ╹╹  ┗━╸┗━╸╹ ╹┗━╸╹ ╹ ╹ ╹ ╹ ╹ ╹┗━┛╹ ╹┗━┛


__all__ = [
    'Gemma2ForCausalLM',
    'Gemma3ForCausalLM',
    'GemmaForCausalLM',
]
