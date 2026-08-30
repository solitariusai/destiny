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
"""Type annotations for layers"""

import collections.abc as cab
import typing as tp

import jax
import jax.numpy as jnp
from taktiny import nn

from destiny.utils.typing import (
    Activation,
    Axes,
    AxisNames,
    DType,
    Initializer,
    ShardMode,
)


class AttentionLike(tp.Protocol):
    def __call__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        *,
        num_kv_heads: int | None = None,
        context_dim: int | None = None,
        apply_position_fn: cab.Callable | None = None,
        bias: bool | cab.Sequence[bool] = False,
        q_norm: bool | nn.Module = False,
        k_norm: bool | nn.Module = False,
        qk_norm: bool = False,
        qk_norm_across_heads: bool | cab.Sequence[bool] = False,
        epsilon: float = 1e-5,
        window_size: int | None = None,
        scaling: float | None = None,
        softcap: float | None = None,
        dropout: float = 0.0,
        dtype: DType | None = None,
        rngs: nn.Rngs,
        quant: tp.Any = None,
        q_axis_names: AxisNames | None = None,
        k_axis_names: AxisNames | None = None,
        v_axis_names: AxisNames | None = None,
        o_axis_names: AxisNames | None = None,
        dot_general: tp.Any = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> nn.Module: ...

class LayerNormLike(tp.Protocol):
    def __call__(
        self,
        normalized_shape: int | tp.Sequence[int] | None,
        eps: float = 1e-5,
        *,
        elementwise_affine: bool = True,
        dtype: DType = jnp.float32,
        bias: bool = True,
        axes: Axes | None = None,
        initializer: Initializer = jnp.ones,
        bias_initializer: Initializer = jnp.zeros,
        axis_names: AxisNames | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> nn.Module: ...

class RMSNormLike(tp.Protocol):
    def __call__(
        self,
        shape: int | tp.Sequence[int] | None,
        epsilon: float = 1e-5,
        *,
        dtype: DType | None = None,
        with_scale: bool = True,
        bias: bool = False,
        axis_names: AxisNames | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
        initializer: Initializer = jnp.ones,
        bias_initializer: Initializer = jnp.zeros,
        axes: Axes | None = None,
    ) -> nn.Module: ...

class GateMLPLike(tp.Protocol):
    def __call__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        activation: Activation = jax.nn.silu,
        bias: bool = False,
        dtype: DType | None = None,
        rngs: nn.Rngs | None = None,
        gate_axis_names: AxisNames | None = None,
        up_axis_names: AxisNames | None = None,
        down_axis_names: AxisNames | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
        quant: tp.Any = None,
        dot_general: tp.Any = None,
    ) -> nn.Module: ...