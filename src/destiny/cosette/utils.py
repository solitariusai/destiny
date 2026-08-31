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
"""Type annotations and reusable configuration helpers for Cosette."""

from __future__ import annotations

import collections.abc as cab
import json
import typing as tp
from collections.abc import Iterator, Mapping, Sequence, Callable
from functools import update_wrapper
from pathlib import Path
from typing import Any, Concatenate, Self, overload

import jax
import jax.numpy as jnp
from huggingface_hub import hf_hub_download
from taktiny import nn

from destiny.utils.typing import (
    Activation,
    Axes,
    DType,
    Initializer,
    PathLike,
    ShardMode, MeshAxis,
)

_MISSING = object()



class ModelConfig:
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = ModelConfig(**v)
            setattr(self, k, v)
        self.__post_init__(**kwargs)

    def __post_init__(self, **kwargs):
        ...

    def __getattr__(self, name: str) -> Any:
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)

        # 1. Check nested text_config (common in HuggingFace multimodal models like Gemma 3/4, Qwen VL, Llama Vision)
        text_cfg = self.__dict__.get('text_config', None)
        if text_cfg is not None and text_cfg is not self:
            val = getattr(text_cfg, name, None)
            if val is not None:
                return val

        # 2. Check nested sub-configs (vision_config, encoder, decoder)
        for sub_cfg_name in ('vision_config', 'encoder', 'decoder'):
            sub_cfg = self.__dict__.get(sub_cfg_name, None)
            if sub_cfg is not None and sub_cfg is not self:
                val = getattr(sub_cfg, name, None)
                if val is not None:
                    return val

        # 3. Gracefully return None for missing keys
        return None

    def to_dict(self) -> dict[str, tp.Any]:
        """Recursively serialize ModelConfig to a plain dictionary."""
        output = {}
        for k, v in self.__dict__.items():
            if isinstance(v, ModelConfig):
                output[k] = v.to_dict()
            elif isinstance(v, (list, tuple)):
                output[k] = [
                    elem.to_dict() if isinstance(elem, ModelConfig) else elem
                    for elem in v
                ]
            else:
                output[k] = v
        return output

    def with_overrides(self, overrides: Self) -> Self:
        """Return a new config with ``overrides`` layered over this config.

        Nested configuration dictionaries are merged recursively. Neither the
        defaults nor the overrides are mutated, so class-level default configs
        remain reusable across model instances.
        """
        if not isinstance(overrides, ModelConfig):
            raise TypeError('overrides must be a ModelConfig')

        def clone(value: tp.Any) -> tp.Any:
            if isinstance(value, dict):
                return {key: clone(item) for key, item in value.items()}
            if isinstance(value, list):
                return [clone(item) for item in value]
            if isinstance(value, tuple):
                return tuple(clone(item) for item in value)
            return value

        def merge(
            defaults: dict[str, tp.Any],
            supplied: dict[str, tp.Any],
        ) -> dict[str, tp.Any]:
            result = clone(defaults)
            for key, value in supplied.items():
                if (
                    key in result
                    and isinstance(result[key], dict)
                    and isinstance(value, dict)
                ):
                    result[key] = merge(result[key], value)
                else:
                    result[key] = clone(value)
            return result

        return type(self)(
            **merge(
                self.to_dict(), 
                overrides.to_dict()
            )
        )

    def get(self, key: tp.Any, default: tp.Any=None) -> tp.Any:
        """Return a configuration value using mapping-style semantics."""
        value = getattr(self, key, None)
        return default if value is None else value

    @classmethod
    def load_config(
        cls: type[Self], 
        path_or_repo: PathLike,
        filename: str = 'config.json',
        subfolder: tp.Any = None,
        local: bool = False
    ) -> Self | None:
        if local:
            config_path = Path(path_or_repo).resolve()
            if subfolder:
                config_path = config_path / subfolder

            config_path = config_path / filename
            if not config_path.is_file():
                raise FileNotFoundError(
                    f'config file not found: {config_path}. With local=True, '
                    f'{path_or_repo!r} must be a local checkpoint directory '
                    f'containing {filename!r}.'
                )
        else:
            try:
                config_path = hf_hub_download(
                    repo_id=str(path_or_repo),
                    subfolder=subfolder if subfolder else None,
                    filename=filename
                )

            except Exception as e:
                print(f'config.json not found in repo: {e}')
                return None

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

        except Exception as e:
            print(f'Error loading config.json: {e}')
            return None

        return cls(**config)

    def __repr__(self) -> str:
        config_str = json.dumps(self.__dict__, indent=2, default=str)
        return f"{self.__class__.__name__} {config_str}"


def _validate_dtype_config(config: ModelConfig):
    dtype = config.dtype or config.torch_dtype
    if dtype is None:
        import warnings
        warnings.warn('Not found `dtype` or `torch_dtype` in model config fallback to float32')
        dtype = 'float32'
    return dtype


def _verify_required_config_attributes(config: ModelConfig, config_attributes: tp.Sequence[str] | None) -> None:
    missing = []
    if config_attributes is None:
        return

    for attr in config_attributes:
        if not hasattr(config, attr):
            missing.append(attr)
    if len(missing) > 0:
        raise ValueError(f'Missing config attributes: {', '.join(missing)}.')


@jax.tree_util.register_pytree_node_class
class ModelOutput(Mapping[str, tp.Any]):
    """Attribute-accessible output PyTree for model call results.

    Field names are static PyTree metadata and field values are dynamic leaves.
    Consequently, a compiled function must preserve the same output fields, but
    their array values may change normally between calls.
    """

    __slots__ = ('_keys', '_values')

    def __init__(self, **fields: tp.Any) -> None:
        if not fields:
            raise ValueError('ModelOutput requires at least one field')
        invalid = [name for name in fields if not name.isidentifier()]
        if invalid:
            names = ', '.join(repr(name) for name in invalid)
            raise ValueError(f'ModelOutput field names must be identifiers: {names}')
        object.__setattr__(self, '_keys', tuple(fields))
        object.__setattr__(self, '_values', tuple(fields.values()))

    def __getitem__(self, key: str) -> tp.Any:
        if not isinstance(key, str):
            raise TypeError('ModelOutput keys must be strings')
        try:
            index = self._keys.index(key)
        except ValueError as error:
            raise KeyError(key) from error
        return self._values[index]

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __getattr__(self, name: str) -> tp.Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value: tp.Any) -> None:
        raise AttributeError('ModelOutput fields cannot be assigned directly')

    def pop(self, key: str, default: tp.Any = _MISSING) -> tp.Any:
        """Remove ``key`` and return its value using dictionary semantics."""
        if not isinstance(key, str):
            raise TypeError('ModelOutput keys must be strings')
        try:
            index = self._keys.index(key)
        except ValueError as error:
            if default is _MISSING:
                raise KeyError(key) from error
            return default

        value = self._values[index]
        object.__setattr__(
            self,
            '_keys',
            self._keys[:index] + self._keys[index + 1:],
        )
        object.__setattr__(
            self,
            '_values',
            self._values[:index] + self._values[index + 1:],
        )
        return value

    def tree_flatten(self) -> tuple[tuple[tp.Any, ...], tuple[str, ...]]:
        return self._values, self._keys

    @classmethod
    def tree_unflatten(
        cls,
        keys: tuple[str, ...],
        values: tuple[tp.Any, ...],
    ) -> tp.Self:
        return cls(**dict(zip(keys, values, strict=True)))

    def __repr__(self) -> str:
        fields = ', '.join(
            f'{name}={value!r}'
            for name, value in zip(self._keys, self._values, strict=True)
        )
        return f'{type(self).__name__}({fields})'


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
        axis_names: AxisName | None = None,
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
        axis_names: tuple[str | None, ...] | None = None,
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
        axis_names: tuple[str | None, ...] | None = None,
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
        axis_names: AxisName | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
        quant: tp.Any = None,
        dot_general: tp.Any = None,
    ) -> nn.Module: ...


class dualmethod[T, **P, R]:
    def __init__(
        self,
        method: Callable[Concatenate[type[T], P], R],
    ) -> None:
        self.class_method = method
        self.instance_method: Callable[Concatenate[T, P], R] | None = None
        self.name = method.__name__
        update_wrapper(self, method)

    def __set_name__(self, owner: type[T], name: str) -> None:
        self.name = name

    def instance(
        self,
        method: Callable[Concatenate[T, P], R],
    ) -> "dualmethod[T, P, R]":
        self.instance_method = method
        return self

    @overload
    def __get__(
        self,
        obj: None,
        cls: type[T],
    ) -> Callable[P, R]: ...

    @overload
    def __get__(
        self,
        obj: T,
        cls: type[T] | None = None,
    ) -> Callable[P, R]: ...

    def __get__(
        self,
        obj: T | None,
        cls: type[T] | None = None,
    ) -> Callable[P, R]:
        if cls is None:
            cls = type(obj)

        if obj is None:
            return self.class_method.__get__(cls, cls)

        if self.instance_method is None:
            raise AttributeError(
                f'{type(obj).__name__}.{self.name} has no instance implementation'
            )

        return self.instance_method.__get__(obj, cls)


class ModuleMap[M: ModuleMap](Sequence[tuple[tp.Any, ...]]):
    def __init__(
        self: M,
        mapping: cab.Iterable[tuple[tp.Any, ...]] = (),
    ) -> None:
        self.mapping: list[tuple[tp.Any, ...]] = []
        self.extend(mapping)

    def __iter__(self) -> Iterator[tuple[tp.Any, ...]]:
        return iter(self.mapping)

    def __len__(self) -> int:
        return len(self.mapping)

    @overload
    def __getitem__(self, index: int) -> tuple[tp.Any, ...]: ...

    @overload
    def __getitem__(self, index: slice) -> list[tuple[tp.Any, ...]]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> tuple[tp.Any, ...] | list[tuple[tp.Any, ...]]:
        return self.mapping[index]

    def copy(self: M) -> M:
        return type(self)(self.mapping)

    def extend(
        self: M,
        mapping: cab.Iterable[tuple[tp.Any, ...]],
    ) -> M:
        for rule in mapping:
            if not isinstance(rule, tuple) or len(rule) not in {2, 3}:
                raise ValueError('module-map rules must be 2- or 3-tuples')
            if len(rule) == 3 and not callable(rule[2]):
                raise TypeError('the third module-map item must be callable')
            self.mapping.append(rule)
        return self

    @staticmethod
    def _validate_name(name: str, label: str) -> None:
        if not isinstance(name, str):
            raise TypeError(f'{label} must be a string')
        if not name:
            raise ValueError(f'{label} cannot be empty')

    @dualmethod
    def map(cls: type[M], source: str, target: str) -> M:
        obj = cls()
        return obj.map(source, target)

    @map.instance
    def map(self: M, source: str, target: str) -> M:
        self._validate_name(source, 'source')
        self._validate_name(target, 'target')
        self.mapping.append((source, target))
        return self

    @dualmethod
    def split(
        cls: type[M],
        source: str,
        targets: Sequence[str],
        *,
        axis: int = 0,
    ) -> M:
        obj = cls()
        return obj.split(source, targets, axis=axis)

    @split.instance
    def split(
        self: M,
        source: str,
        targets: Sequence[str],
        *,
        axis: int = 0,
    ) -> M:
        self._validate_name(source, 'source')
        if isinstance(targets, str):
            raise TypeError('targets must be a sequence of strings')
        normalized_targets = tuple(targets)
        if not normalized_targets:
            raise ValueError('targets cannot be empty')
        for target in normalized_targets:
            self._validate_name(target, 'target')
        if not isinstance(axis, int):
            raise TypeError('axis must be an integer')

        def split_value(value: tp.Any) -> tuple[tp.Any, ...]:
            return tuple(jnp.split(value, len(normalized_targets), axis=axis))

        self.mapping.append((source, normalized_targets, split_value))
        return self

    @dualmethod
    def concat(
        cls: type[M],
        sources: Sequence[str],
        target: str,
        *,
        axis: int = 0,
    ) -> M:
        obj = cls()
        return obj.concat(sources, target, axis=axis)

    @concat.instance
    def concat(
        self: M,
        sources: Sequence[str],
        target: str,
        *,
        axis: int = 0,
    ) -> M:
        if isinstance(sources, str):
            raise TypeError('sources must be a sequence of strings')
        normalized_sources = tuple(sources)
        if not normalized_sources:
            raise ValueError('sources cannot be empty')
        for source in normalized_sources:
            self._validate_name(source, 'source')
        self._validate_name(target, 'target')
        if not isinstance(axis, int):
            raise TypeError('axis must be an integer')

        def concat_values(*values: tp.Any) -> tp.Any:
            return jnp.concatenate(values, axis=axis)

        self.mapping.append((normalized_sources, target, concat_values))
        return self

    def __repr__(self: Self) -> str:
        return f'{type(self).__name__}({self.mapping!r})'


class AxisName[A: AxisName](Mapping[str, tuple[str | None, ...] | None]):
    def __init__(self, **axis_names: Sequence[str | None] | None) -> None:
        self._axis_names: dict[str, tuple[str | None, ...] | None] = {}
        self.set_axis_names(**axis_names)

    @staticmethod
    def _normalize(
        name: str,
        value: Sequence[str | None] | None,
    ) -> tuple[str | None, ...] | None:
        if not isinstance(name, str):
            raise TypeError('axis-name attributes must be strings')
        if not name or not name.isidentifier():
            raise ValueError(f'invalid axis-name attribute: {name!r}')
        if value is None:
            return None
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise TypeError(f'{name} must be a sequence of strings or None')
        normalized = tuple(value)
        if any(
            axis is not None and not isinstance(axis, str)
            for axis in normalized
        ):
            raise TypeError(f'{name} axes must be strings or None')
        return normalized

    @dualmethod
    def set_axis_names(
        cls: type[A],
        **axis_names: Sequence[str | None] | None,
    ) -> A:
        obj = cls()
        obj.set_axis_names(**axis_names)
        return obj

    @set_axis_names.instance
    def set_axis_names(
        self: A,
        **axis_names: Sequence[str | None] | None,
    ) -> A:
        for name, value in axis_names.items():
            self._axis_names[name] = self._normalize(name, value)
        return self

    def __getattr__(self, name: str) -> tuple[str | None, ...] | None:
        try:
            return self._axis_names[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __contains__(self, name: str) -> bool:
        return name in self._axis_names

    def __iter__(self) -> Iterator[str]:
        return iter(self._axis_names)

    def __len__(self) -> int:
        return len(self._axis_names)

    def __getitem__(self, name: str) -> tuple[str | None, ...] | None:
        return self._axis_names[name]

    def get(
        self,
        name: str,
        default: tuple[str | None, ...] | None = None,
    ) -> tuple[str | None, ...] | None:
        return self._axis_names.get(name, default)

    def copy(self: A) -> A:
        return type(self)(**self._axis_names)

    def __repr__(self) -> str:
        fields = ', '.join(
            f'{name}={value!r}' for name, value in self._axis_names.items()
        )
        return f'{type(self).__name__}({fields})'


class ShardingRule[S: ShardingRule](Sequence[tuple[str, str | None]]):
    def __init__(
        self: S,
        rules: cab.Iterable[tuple[str, MeshAxis]] = (),
    ) -> None:
        self._sharding_rules: list[tuple[str, MeshAxis]] = []
        for names, mesh_axis in rules:
            self.set_logical([names], mesh_axis)

    def __iter__(self) -> Iterator[tuple[str, str | None]]:
        return iter(tuple(
            (name, mesh_axis.value)
            for name, mesh_axis in self._sharding_rules
        ))

    def __len__(self) -> int:
        return len(self._sharding_rules)

    @overload
    def __getitem__(self, index: int) -> tuple[str, str | None]: ...

    @overload
    def __getitem__(self, index: slice) -> list[tuple[str, str | None]]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> tuple[str, str | None] | list[tuple[str, str | None]]:
        rules = list(self)
        return rules[index]

    def copy(self: S) -> S:
        return type(self)(self._sharding_rules)

    def set_logical(
        self: S, 
        names: list[str], 
        mesh_axis: MeshAxis
    ) -> S:
        if not isinstance(mesh_axis, MeshAxis):
            raise TypeError('mesh_axis must be a MeshAxis')
        if not names:
            raise ValueError('at least one logical axis name is required')
        for name in names:
            if not isinstance(name, str):
                raise TypeError('logical axis names must be strings')
            if not name:
                raise ValueError('logical axis names cannot be empty')
            if any(existing == name for existing, _ in self._sharding_rules):
                raise ValueError(f'logical axis {name!r} is already mapped')
            self._sharding_rules.append((name, mesh_axis))
        return self

    @dualmethod
    def set_logical_tp(
        cls: type[S], 
        *names: str
    ) -> S:
        obj = cls()
        obj.set_logical_tp(*names)
        return obj

    @set_logical_tp.instance
    def set_logical_tp(
        self: S, 
        *names: str
    ) -> S:
        self.set_logical(list(names), MeshAxis.TP)
        return self

    @dualmethod
    def set_logical_dp(
        cls: type[S], 
        *names: str
    ) -> S:
        obj = cls()
        obj.set_logical_dp(*names)
        return obj

    @set_logical_dp.instance
    def set_logical_dp(
        self: S, 
        *names: str
    ) -> S:
        self.set_logical(list(names), MeshAxis.DP)
        return self

    @dualmethod
    def set_logical_rp(
        cls: type[S], 
        *names: str
    ) -> S:
        obj = cls()
        obj.set_logical_rp(*names)
        return obj

    @set_logical_rp.instance
    def set_logical_rp(
        self: S, 
        *names: str
    ) -> S:
        self.set_logical(list(names), MeshAxis.RP)
        return self

    def __repr__(self) -> str:
        return f'{type(self).__name__}({list(self)!r})'
