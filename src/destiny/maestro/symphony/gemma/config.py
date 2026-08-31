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
"""Gemma configurations"""

from __future__ import annotations

from destiny.cosette.layers.positional_embedding import RopeParameters
from destiny.cosette.utils import ModelConfig


# ┏━╸┏━╸┏┳┓┏┳┓┏━┓
# ┃╺┓┣╸ ┃┃┃┃┃┃┣━┫
# ┗━┛┗━╸╹ ╹╹ ╹╹ ╹
# ┏━╸┏━┓┏┓╻┏━╸╻┏━╸╻ ╻┏━┓┏━┓╺┳╸╻┏━┓┏┓╻┏━┓
# ┃  ┃ ┃┃┗┫┣╸ ┃┃╺┓┃ ┃┣┳┛┣━┫ ┃ ┃┃ ┃┃┗┫┗━┓
# ┗━╸┗━┛╹ ╹╹  ╹┗━┛┗━┛╹┗╸╹ ╹ ╹ ╹┗━┛╹ ╹┗━┛
class GemmaConfig(ModelConfig):
    model_type = "gemma"
    vocab_size: int = 256000
    hidden_size: int = 3072
    intermediate_size: int = 24576
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 256
    hidden_act: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 8192
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    bos_token_id: int | None = 2
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    use_bidirectional_attention: bool | None = None


# ┏━╸┏━╸┏┳┓┏┳┓┏━┓┏━┓
# ┃╺┓┣╸ ┃┃┃┃┃┃┣━┫┏━┛
# ┗━┛┗━╸╹ ╹╹ ╹╹ ╹┗━╸
# ┏━╸┏━┓┏┓╻┏━╸╻┏━╸╻ ╻┏━┓┏━┓╺┳╸╻┏━┓┏┓╻┏━┓
# ┃  ┃ ┃┃┗┫┣╸ ┃┃╺┓┃ ┃┣┳┛┣━┫ ┃ ┃┃ ┃┃┗┫┗━┓
# ┗━╸┗━┛╹ ╹╹  ╹┗━┛┗━┛╹┗╸╹ ╹ ╹ ╹┗━┛╹ ╹┗━┛
class Gemma2Config(ModelConfig):
    r"""
    query_pre_attn_scalar (`float`, *optional*, defaults to 256):
        scaling factor used on the attention scores
    final_logit_softcapping (`float`, *optional*, defaults to 30.0):
        scaling factor when applying tanh softcapping on the logits.
    attn_logit_softcapping (`float`, *optional*, defaults to 50.0):
        scaling factor when applying tanh softcapping on the attention scores.
    use_bidirectional_attention (`bool`, *optional*):
        If True, the model will attend to all text tokens instead of using a causal mask.
    """
    model_type = "gemma2"
    vocab_size: int = 256000
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 26
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 8192
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    bos_token_id: int | None = 2
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: int | float | None = 0.0
    query_pre_attn_scalar: int = 256
    sliding_window: int | None = 4096
    layer_types: list[str] | None = None
    final_logit_softcapping: float | None = 30.0
    attn_logit_softcapping: float | None = 50.0
    use_bidirectional_attention: bool | None = None

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})."
            )


# ┏━╸┏━╸┏┳┓┏┳┓┏━┓┏━┓
# ┃╺┓┣╸ ┃┃┃┃┃┃┣━┫╺━┫
# ┗━┛┗━╸╹ ╹╹ ╹╹ ╹┗━┛
# ┏━╸┏━┓┏┓╻┏━╸╻┏━╸╻ ╻┏━┓┏━┓╺┳╸╻┏━┓┏┓╻┏━┓
# ┃  ┃ ┃┃┗┫┣╸ ┃┃╺┓┃ ┃┣┳┛┣━┫ ┃ ┃┃ ┃┃┗┫┗━┓
# ┗━╸┗━┛╹ ╹╹  ╹┗━┛┗━┛╹┗╸╹ ╹ ╹ ╹┗━┛╹ ╹┗━┛
class Gemma3TextConfig(ModelConfig):
    r"""
    query_pre_attn_scalar (`float`, *optional*, defaults to 256):
        scaling factor used on the attention scores
    final_logit_softcapping (`float`, *optional*):
        Scaling factor when applying tanh softcapping on the logits.
    attn_logit_softcapping (`float`, *optional*):
        Scaling factor when applying tanh softcapping on the attention scores.
    use_bidirectional_attention (`bool`, *optional*, defaults to `False`):
        If True, the model will attend to all text tokens instead of using a causal mask. This does not change
        behavior for vision tokens.
    """

    model_type = "gemma3_text"
    vocab_size: int = 262_208
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 26
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 131_072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    bos_token_id: int | None = 2
    tie_word_embeddings: bool = True
    rope_parameters: dict | None = None
    attention_bias: bool = False
    attention_dropout: int | float | None = 0.0
    query_pre_attn_scalar: int = 256
    sliding_window: int | None = 4096
    layer_types: list[str] | None = None
    final_logit_softcapping: float | None = None
    attn_logit_softcapping: float | None = None
    use_bidirectional_attention: bool | None = False
    default_theta = {"global": 1_000_000.0, "local": 10_000.0}

    def __post_init__(self, **kwargs):
        if self.use_bidirectional_attention:
            self.sliding_window = (self.sliding_window // 2) + 1  # due to fa we set exclusive bounds

        # BC -> the pattern used to be a simple int, and it's still present in configs on the Hub
        self._sliding_window_pattern = kwargs.get("sliding_window_pattern", 6)

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % self._sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})."
            )

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        # Serialize the value `__post_init__` converted *from*, so that a reload converts once more
        # instead of halving the already-halved window. Configs that inherit this one without the
        # flag never convert, hence the `False` fallback.
        if getattr(self, "use_bidirectional_attention", False):
            output["sliding_window"] = (self.sliding_window - 1) * 2
        return output

    def convert_rope_params_to_dict(self, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)

        # Try to set `rope_scaling` if available, otherwise use `rope_parameters`. If we find `rope_parameters`
        # as arg in the inputs, we can safely assume that it is in the new format. New naming used -> new format
        default_rope_params = {
            "sliding_attention": {"rope_type": "default"},
            "full_attention": {"rope_type": "default"},
        }
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else default_rope_params
        if rope_scaling is not None:
            self.rope_parameters["full_attention"].update(rope_scaling)

        # Set default values if not present
        if self.rope_parameters.get("full_attention") is None:
            self.rope_parameters["full_attention"] = {"rope_type": "default"}
        self.rope_parameters["full_attention"].setdefault(
            "rope_theta", kwargs.pop("rope_theta", self.default_theta["global"])
        )
        if self.rope_parameters.get("sliding_attention") is None:
            self.rope_parameters["sliding_attention"] = {"rope_type": "default"}
        self.rope_parameters["sliding_attention"].setdefault(
            "rope_theta", kwargs.pop("rope_local_base_freq", self.default_theta["local"])
        )

        # Standardize and validate the correctness of rotary position embeddings parameters
        self.standardize_rope_params()
        return kwargs


__all__ = [
    "Gemma2Config",
    "Gemma3TextConfig",
    "GemmaConfig",
]