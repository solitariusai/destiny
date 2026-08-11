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
from __future__ import annotations
import typing as tp

from taktiny import nn, layers as ly
from taktiny.cosettes.diffusions._ordinario import DiffusionTransformerModel
from taktiny.maestro.config import ModelConfig
from taktiny.layers import PatchEmbedding, AttentionPooling


class StableDiffusion3(DiffusionTransformerModel):
    def __init__(self, config: ModelConfig):
        super().__init__(
            patch_layer, 
            time_step_embedding, 
            context_embedding, 
            transformer_blocks, 
            norm_out, proj_out, 
            use_scale_shift_table
        )