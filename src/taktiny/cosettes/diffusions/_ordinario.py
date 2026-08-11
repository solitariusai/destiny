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
from taktiny import nn, layers as ly
import typing as tp


class PatchEmbedding(nn.Module):
    def __init__(self, patch_module, position_module):
        ...


class DiffusionTransformerLayer(nn.Module):
    def __init__(self):
        ...


class DiffusionTransformerModel(nn.Module):
    def __init__(
        self, 
        patch_layer: nn.Module, 
        time_step_embedding: nn.Module, 
        context_embedding: nn.Module, 
        transformer_blocks: tp.List[nn.Module], 
        norm_out: nn.Module, 
        proj_out: nn.Module, 
        use_scale_shift_table: bool,
    ):
        self.patch_layer = patch_layer
        self.time_embed = time_step_embedding
        self.transformer_blocks = transformer_blocks # cannot None
        self.norm_out = proj_out or ly.AdaXNorm() # adalyernorm # cannot None maybe
        self.proj_out = proj_out or nn.Linear() # cannot None maybe
        if use_scale_shift_table:
            scale_shift_table = nn.Parameter()


class DiffusionTransformer1D(DiffusionTransformerModel):
    def __init__(self):
        ...


class DiffusionTransformer2D(DiffusionTransformerModel):
    def __init__(self):
        ...


class DiffusionTransformer3D(DiffusionTransformerModel):
    def __init__(self):
        ...
