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
"""Qwen implementations"""

from __future__ import annotations

from destiny.maestro.transformer import TransformerCausalLM
from destiny.maestro.symphony.qwen.config import Qwen2Config, Qwen3Config
from destiny.maestro.symphony.qwen.qwen2 import Qwen2Model
from destiny.maestro.symphony.qwen.qwen3 import Qwen3Model
from destiny.maestro.utils import destiny


# ┏━┓╻ ╻┏━╸┏┓╻┏━┓
# ┃┓┃┃╻┃┣╸ ┃┗┫┏━┛
# ┗┻┛┗┻┛┗━╸╹ ╹┗━╸
# ╻┏┳┓┏━┓╻  ┏━╸┏┳┓┏━╸┏┓╻╺┳╸┏━┓╺┳╸╻┏━┓┏┓╻┏━┓
# ┃┃┃┃┣━┛┃  ┣╸ ┃┃┃┣╸ ┃┗┫ ┃ ┣━┫ ┃ ┃┃ ┃┃┗┫┗━┓
# ╹╹ ╹╹  ┗━╸┗━╸╹ ╹┗━╸╹ ╹ ╹ ╹ ╹ ╹ ╹┗━┛╹ ╹┗━┛
@destiny
class Qwen2ForCausalLM(TransformerCausalLM):
    _model_type = Qwen2Model
    _default_config = Qwen2Config()


# ┏━┓╻ ╻┏━╸┏┓╻┏━┓
# ┃┓┃┃╻┃┣╸ ┃┗┫╺━┫
# ┗┻┛┗┻┛┗━╸╹ ╹┗━┛
# ╻┏┳┓┏━┓╻  ┏━╸┏┳┓┏━╸┏┓╻╺┳╸┏━┓╺┳╸╻┏━┓┏┓╻┏━┓
# ┃┃┃┃┣━┛┃  ┣╸ ┃┃┃┣╸ ┃┗┫ ┃ ┣━┫ ┃ ┃┃ ┃┃┗┫┗━┓
# ╹╹ ╹╹  ┗━╸┗━╸╹ ╹┗━╸╹ ╹ ╹ ╹ ╹ ╹ ╹┗━┛╹ ╹┗━┛
@destiny
class Qwen3ForCausalLM(TransformerCausalLM):
    _model_type = Qwen3Model
    _default_config = Qwen3Config()


__all__ = [
    'Qwen2ForCausalLM',
    'Qwen3ForCausalLM',
]
