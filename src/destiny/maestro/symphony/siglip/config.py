# Copyright 2026 Shinapri
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Siglip configurations"""

from __future__ import annotations

import logging
from typing import ClassVar

from destiny.cosette.utils import ModelConfig

logger = logging.getLogger(__name__)


# ┏━┓╻┏━╸╻  ╻┏━┓
# ┗━┓┃┃╺┓┃  ┃┣━┛
# ┗━┛╹┗━┛┗━╸╹╹  
# ┏━╸┏━┓┏┓╻┏━╸╻┏━╸╻ ╻┏━┓┏━┓╺┳╸╻┏━┓┏┓╻┏━┓
# ┃  ┃ ┃┃┗┫┣╸ ┃┃╺┓┃ ┃┣┳┛┣━┫ ┃ ┃┃ ┃┃┗┫┗━┓
# ┗━╸┗━┛╹ ╹╹  ╹┗━┛┗━┛╹┗╸╹ ╹ ╹ ╹┗━┛╹ ╹┗━┛
class SiglipTextConfig(ModelConfig):

    model_type = "siglip_text_model"
    base_config_key = "text_config"

    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 64
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float | int = 0.0
    pad_token_id: int | None = 1
    bos_token_id: int | None = 49406
    eos_token_id: int | list[int] | None = 49407
    projection_size: int | None = None

    def __post_init__(self, **kwargs):
        self.projection_size = self.projection_size if self.projection_size is not None else self.hidden_size
        super().__post_init__(**kwargs)


class SiglipVisionConfig(ModelConfig):
    model_type = "siglip_vision_model"
    base_config_key = "vision_config"

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 16
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float | int = 0.0


class SiglipConfig(ModelConfig):

    model_type = "siglip"
    sub_configs: ClassVar = {"text_config": SiglipTextConfig, "vision_config": SiglipVisionConfig}

    text_config: dict | ModelConfig | None = None
    vision_config: dict | ModelConfig | None = None
    initializer_factor: float = 1.0

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = SiglipTextConfig()
            logger.info("`text_config` is `None`. Initializing the `SiglipTextConfig` with default values.")
        elif isinstance(self.text_config, dict):
            self.text_config = SiglipTextConfig(**self.text_config)

        if self.vision_config is None:
            self.vision_config = SiglipVisionConfig()
            logger.info("`vision_config` is `None`. initializing the `SiglipVisionConfig` with default values.")
        elif isinstance(self.vision_config, dict):
            self.vision_config = SiglipVisionConfig(**self.vision_config)

        super().__post_init__(**kwargs)


__all__ = [
    "SiglipConfig", 
    "SiglipTextConfig", 
    "SiglipVisionConfig"
]