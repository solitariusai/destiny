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
"""Conditioned normalization layers."""
from __future__ import annotations
from collections.abc import Callable, Sequence
import math
from typing import Any, Literal, TypeAlias
import jax
import jax.numpy as jnp

from taktiny import nn
from taktiny.nn.linear import default_linear_initializer
from taktiny.utils.typing import Axes, AxisNames, DType, Initializer, ShardMode

NormType: TypeAlias = Literal['layernorm', 'rmsnorm']
NormModule: TypeAlias = (
    nn.LayerNorm | nn.RMSNorm | nn.GroupNorm | nn.BatchNorm
)
Normalizer: TypeAlias = (
    NormType | NormModule | Callable[[jax.Array], jax.Array]
)
Activation: TypeAlias = str | Callable[[jax.Array], jax.Array] | None

def _positive_integer(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value

def _canonical_axes(axes: Axes, ndim: int) -> int | tuple[int, ...]:
    values = (axes,) if isinstance(axes, int) else tuple(axes)
    if not values:
        raise ValueError('axes must contain at least one axis')
    if not all(isinstance(axis, int) for axis in values):
        raise TypeError('axes must contain only integers')

    canonical = tuple(axis + ndim if axis < 0 else axis for axis in values)
    if any(axis < 0 or axis >= ndim for axis in canonical):
        raise ValueError(f'axes {values} are invalid for an array of rank {ndim}')
    if len(set(canonical)) != len(canonical):
        raise ValueError('axes must not contain duplicates')
    return canonical[0] if len(canonical) == 1 else canonical

def _resolve_activation(activation: Activation) -> Callable[[jax.Array], jax.Array]:
    if activation is None:
        return lambda value: value
    if callable(activation):
        return activation
    if not isinstance(activation, str):
        raise TypeError('activation must be a string, callable, or None')

    function = getattr(jax.nn, activation, None)
    if function is None or not callable(function):
        raise ValueError(f'unsupported activation: {activation!r}')
    return function

def _constrain(
    value: jax.Array,
    sharding: jax.sharding.Sharding | None,
    shard_mode: ShardMode,
) -> jax.Array:
    if shard_mode == ShardMode.EXPLICIT and sharding is not None:
        return jax.lax.with_sharding_constraint(value, sharding)
    return value

class AdaXNorm(nn.Module):
    """
    Normalize activations and project a conditioning tensor.

    ``norm`` may be ``"layernorm"``, ``"rmsnorm"``, or any module or
    callable that maps the input activation to an equally shaped array. The
    built-in normalizers are parameter-free and may reduce over any requested
    axes. The projected modulation is deliberately left unsplit so an
    architecture can interpret it as scale, shift, gate, or another signal.
    """

    def __init__(
        self,
        embedding_dim: int,
        out_dim: int | Sequence[int],
        norm: Normalizer = 'layernorm',
        eps: float = 1e-6,
        *,
        axes: Axes = -1,
        activation: Activation = 'silu',
        bias: bool = True,
        dtype: DType = jnp.float32,
        rngs: nn.Rngs,
        initializer: Initializer = default_linear_initializer,
        quant: Any = None,
        dot_general: Any = None,
        axis_names: AxisNames | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        self.embedding_dim = _positive_integer(embedding_dim, 'embedding_dim')
        if isinstance(out_dim, int):
            _positive_integer(out_dim, 'out_dim')
            self.out_dim = (out_dim,)
        else:
            self.out_dim = tuple(out_dim)
            if not self.out_dim:
                raise ValueError('out_dim must contain at least one dimension')
            for index, dimension in enumerate(self.out_dim):
                _positive_integer(dimension, f'out_dim[{index}]')

        if not math.isfinite(eps) or eps <= 0:
            raise ValueError('eps must be finite and positive')

        if isinstance(norm, str):
            normalized_name = norm.lower().replace('-', '_')
            aliases = {
                'layer': 'layernorm',
                'layer_norm': 'layernorm',
                'rms': 'rmsnorm',
                'rms_norm': 'rmsnorm',
            }
            normalized_name = aliases.get(normalized_name, normalized_name)
            if normalized_name not in {'layernorm', 'rmsnorm'}:
                raise ValueError(f'unsupported norm: {norm!r}')
            self.norm_type = normalized_name
            self.normalizer = None
        elif isinstance(norm, nn.Module) or callable(norm):
            self.norm_type = 'custom'
            self.normalizer = norm
        else:
            raise TypeError('norm must be a supported string, module, or callable')

        self.eps = float(eps)
        self.axes = axes
        self.activation = _resolve_activation(activation)
        self.shard_mode = shard_mode
        self.linear = nn.Linear(
            self.embedding_dim,
            self.out_dim,
            bias=bias,
            dtype=dtype,
            rngs=rngs,
            initializer=initializer,
            quant=quant,
            dot_general=dot_general,
            axis_names=axis_names,
            shard_mode=shard_mode,
        )

    def _normalize(self, x: jax.Array) -> jax.Array:
        if self.normalizer is not None:
            normalized = self.normalizer(x)
            if normalized.shape != x.shape:
                raise ValueError(
                    'a custom normalizer must preserve the input shape; '
                    f'got {x.shape} -> {normalized.shape}'
                )
            return normalized

        axes = _canonical_axes(self.axes, x.ndim)
        if self.norm_type == 'layernorm':
            mean = jnp.mean(x, axis=axes, keepdims=True)
            variance = jnp.var(x, axis=axes, keepdims=True)
            return (x - mean) * jax.lax.rsqrt(variance + self.eps)

        variance = jnp.mean(jnp.square(x), axis=axes, keepdims=True)
        return x * jax.lax.rsqrt(variance + self.eps)

    def __call__(
        self,
        x: jax.Array,
        conditioning: jax.Array,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
        modulation_sharding: jax.sharding.Sharding | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        x = jnp.asarray(x)
        conditioning = jnp.asarray(conditioning)
        if not jnp.issubdtype(x.dtype, jnp.inexact):
            raise TypeError('x must have a floating-point or complex dtype')
        if conditioning.shape[-1:] != (self.embedding_dim,):
            raise ValueError(
                'conditioning has an incompatible trailing dimension: '
                f'expected {self.embedding_dim}, got {conditioning.shape}'
            )

        normalized = _constrain(
            self._normalize(x),
            out_sharding,
            self.shard_mode,
        )
        modulation = self.linear(
            self.activation(conditioning),
            out_sharding=modulation_sharding,
        )
        return normalized, modulation

    def extra_repr(self) -> str:
        output = 'x'.join(map(str, self.out_dim))
        return f'{self.embedding_dim} -> {output}, norm={self.norm_type}'


class SpatialNorm(nn.Module):
    """
    Apply group normalization modulated by a spatial conditioning tensor.

    Inputs use channels-last layout ``[batch, ..., channels]``. The conditioning
    tensor is resized to the feature tensor's non-channel dimensions, then two
    pointwise projections produce multiplicative and additive modulation. This
    formulation supports sequences, images, volumes, and higher-rank spatial
    arrays with the same parameters.
    """

    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
        *,
        num_groups: int = 32,
        eps: float = 1e-6,
        interpolation: str = 'nearest',
        affine: bool = True,
        bias: bool = True,
        dtype: DType = jnp.float32,
        rngs: nn.Rngs,
        initializer: Initializer = default_linear_initializer,
        quant: Any = None,
        dot_general: Any = None,
        axis_names: AxisNames | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
    ) -> None:
        self.f_channels = _positive_integer(f_channels, 'f_channels')
        self.zq_channels = _positive_integer(zq_channels, 'zq_channels')
        self.num_groups = _positive_integer(num_groups, 'num_groups')
        if self.f_channels % self.num_groups != 0:
            raise ValueError('f_channels must be divisible by num_groups')
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError('eps must be finite and positive')
        if not isinstance(interpolation, str):
            raise TypeError('interpolation must be a string')
        if axis_names is not None and len(axis_names) != 2:
            raise ValueError(
                'axis_names must contain conditioning and feature axes'
            )

        self.eps = float(eps)
        self.interpolation = interpolation
        self.shard_mode = shard_mode
        norm_axis_names = (
            None if axis_names is None else (axis_names[-1],)
        )
        self.norm_layer = nn.GroupNorm(
            self.num_groups,
            self.f_channels,
            eps=self.eps,
            affine=affine,
            bias=bias,
            dtype=dtype,
            axis_names=norm_axis_names,
            shard_mode=shard_mode,
        )
        projection_options = {
            'bias': bias,
            'dtype': dtype,
            'rngs': rngs,
            'initializer': initializer,
            'quant': quant,
            'dot_general': dot_general,
            'axis_names': axis_names,
            'shard_mode': shard_mode,
        }
        self.scale = nn.Linear(
            self.zq_channels,
            self.f_channels,
            **projection_options,
        )
        self.shift = nn.Linear(
            self.zq_channels,
            self.f_channels,
            **projection_options,
        )

    def __call__(
        self,
        features: jax.Array,
        conditioning: jax.Array,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
        modulation_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        features = jnp.asarray(features)
        conditioning = jnp.asarray(conditioning)
        if features.ndim < 2 or conditioning.ndim != features.ndim:
            raise ValueError(
                'features and conditioning must have equal rank and use '
                '[batch, ..., channels] layout'
            )
        if features.shape[-1] != self.f_channels:
            raise ValueError(
                f'expected {self.f_channels} feature channels, '
                f'got {features.shape[-1]}'
            )
        if conditioning.shape[-1] != self.zq_channels:
            raise ValueError(
                f'expected {self.zq_channels} conditioning channels, '
                f'got {conditioning.shape[-1]}'
            )
        if conditioning.shape[0] != features.shape[0]:
            raise ValueError('features and conditioning must share a batch size')

        target_shape = (*features.shape[:-1], self.zq_channels)
        if conditioning.shape != target_shape:
            conditioning = jax.image.resize(
                conditioning,
                shape=target_shape,
                method=self.interpolation,
            )

        normalized = self.norm_layer(features)
        scale = self.scale(
            conditioning,
            out_sharding=modulation_sharding,
        )
        shift = self.shift(
            conditioning,
            out_sharding=modulation_sharding,
        )
        output = normalized * scale + shift
        return _constrain(output, out_sharding, self.shard_mode)

    def extra_repr(self) -> str:
        return (
            f'{self.f_channels}, condition={self.zq_channels}, '
            f'groups={self.num_groups}'
        )

__all__ = [
    'NormType',
    'NormModule',
    'Normalizer',
    'AdaXNorm',
    'SpatialNorm',
]
