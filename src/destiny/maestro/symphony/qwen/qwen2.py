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
"""Qwen2 model implementations"""

from __future__ import annotations

from destiny.maestro.transformer import TransformerModel
from destiny.cosette.transformers.qwen import Qwen2DecoderLayer


class Qwen2Model(TransformerModel):
    _layer_type = Qwen2DecoderLayer

    def __init__(self, config, *, rngs, **kwargs):
        if config.layer_types is None and config.use_sliding_window:
            max_window_layers = config.max_window_layers or 0
            config.layer_types = [
                (
                    'full_attention'
                    if layer_idx < max_window_layers
                    else 'sliding_attention'
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        super().__init__(config, rngs=rngs, **kwargs)


__all__ = ['Qwen2Model']
