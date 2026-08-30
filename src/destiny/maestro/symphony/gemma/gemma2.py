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
"""Gemma2 model implementations"""

from __future__ import annotations

from typing import Any

from taktiny import nn

from destiny.cosette.transformers.gemma import Gemma2DecoderLayer
from destiny.maestro.symphony.gemma.config import Gemma2Config
from destiny.maestro.symphony.gemma.gemma import GemmaModel


class Gemma2Model(GemmaModel):
    _layer_type = Gemma2DecoderLayer

    def __init__(
        self,
        config: Gemma2Config,
        *,
        rngs: nn.Rngs,
        **kwargs: Any,
    ) -> None:
        if config.layer_types is None:
            config.layer_types = [
                (
                    'sliding_attention'
                    if layer_idx % 2 == 0
                    else 'full_attention'
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        super().__init__(config, rngs=rngs, **kwargs)


__all__ = ["Gemma2Model"]