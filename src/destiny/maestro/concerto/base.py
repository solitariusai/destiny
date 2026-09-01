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

"""Low-level codecs and state-dict persistence."""

from __future__ import annotations

import json
import os
import typing as tp
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import jax
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from destiny.utils.typing import PathLike


@dataclass(frozen=True, slots=True)
class DecodedValue[T]:
    """A decoded payload and the storage metadata that accompanied it."""

    value: T
    metadata: Mapping[str, str]


class PretrainedCodec(tp.Protocol):
    """Storage codec accepted anywhere a serialization format is accepted."""

    def serialize(
        self,
        value: tp.Any,
        path: PathLike,
        *,
        metadata: Mapping[str, str] | None = None,
    ) -> None: ...

    def deserialize(self, path: PathLike) -> DecodedValue[tp.Any]: ...


type SerializationFormat = str | PretrainedCodec


class _JsonCodec:
    _metadata_key = '__destiny_metadata__'
    _value_key = '__destiny_value__'

    def serialize(
        self,
        value: tp.Any,
        path: PathLike,
        *,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        payload = value
        if metadata:
            payload = {
                self._metadata_key: dict(metadata),
                self._value_key: value,
            }
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(payload, file, indent=2, default=str)

    def deserialize(self, path: PathLike) -> DecodedValue[tp.Any]:
        with open(path, encoding='utf-8') as file:
            payload = json.load(file)
        if (
            isinstance(payload, dict)
            and set(payload) == {self._metadata_key, self._value_key}
        ):
            metadata = payload[self._metadata_key]
            if not isinstance(metadata, dict):
                raise ValueError('JSON metadata must be an object')
            return DecodedValue(payload[self._value_key], metadata)
        return DecodedValue(payload, {})


class _SafetensorsCodec:
    @staticmethod
    def _tensors(value: tp.Any) -> dict[str, np.ndarray]:
        if not isinstance(value, Mapping):
            raise TypeError('Safetensors payloads must be mappings')
        tensors = {}
        for name, tensor in value.items():
            if not isinstance(name, str):
                raise TypeError('Safetensors keys must be strings')
            array = np.asarray(jax.device_get(tensor))
            if array.dtype == np.dtype('O'):
                raise TypeError(
                    f'Safetensors value {name!r} is not an array payload'
                )
            tensors[name] = array
        return tensors

    def serialize(
        self,
        value: tp.Any,
        path: PathLike,
        *,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        normalized_metadata = None
        if metadata is not None:
            normalized_metadata = dict(metadata)
            invalid = [
                key
                for key, item in normalized_metadata.items()
                if not isinstance(key, str) or not isinstance(item, str)
            ]
            if invalid:
                raise TypeError(
                    'Safetensors metadata keys and values must be strings'
                )
        save_file(
            self._tensors(value),
            os.fspath(path),
            metadata=normalized_metadata,
        )

    def deserialize(
        self,
        path: PathLike,
    ) -> DecodedValue[dict[str, np.ndarray]]:
        tensors = {}
        with safe_open(path, framework='np', device='cpu') as file:
            metadata = file.metadata() or {}
            for name in file.keys():
                tensors[name] = file.get_tensor(name)
        return DecodedValue(tensors, metadata)


class _PretrainedIO:
    """Generic path, metadata, format, and codec handling."""

    _codecs: tp.ClassVar[dict[str, PretrainedCodec]] = {
        'json': _JsonCodec(),
        'safetensors': _SafetensorsCodec(),
    }

    @classmethod
    def _resolve_codec(cls, format: SerializationFormat) -> PretrainedCodec:
        if isinstance(format, str):
            normalized = format.strip().lower()
            codec = cls._codecs.get(normalized)
            if codec is None:
                choices = ', '.join(sorted(cls._codecs))
                raise ValueError(
                    f'unsupported serialization format {format!r}; '
                    f'choose from {choices} or supply a codec'
                )
            return codec
        if not callable(getattr(format, 'serialize', None)):
            raise TypeError('custom codecs must provide serialize()')
        if not callable(getattr(format, 'deserialize', None)):
            raise TypeError('custom codecs must provide deserialize()')
        return format

    @classmethod
    def serialize_to_disk(
        cls,
        path: PathLike,
        value: tp.Any,
        *,
        metadata: Mapping[str, str] | None = None,
        format: SerializationFormat = 'json',
    ) -> str:
        destination = os.path.abspath(os.fspath(path))
        if os.path.isdir(destination):
            raise IsADirectoryError(destination)
        parent = os.path.dirname(destination)
        if parent:
            os.makedirs(parent, exist_ok=True)
        cls._resolve_codec(format).serialize(
            value,
            destination,
            metadata=metadata,
        )
        return destination

    @classmethod
    def deserialize_from_disk(
        cls,
        path: PathLike,
        *,
        format: SerializationFormat = 'json',
        with_metadata: bool = False,
    ) -> tp.Any:
        source = os.path.abspath(os.fspath(path))
        if not os.path.isfile(source):
            raise FileNotFoundError(source)
        decoded = cls._resolve_codec(format).deserialize(source)
        if not isinstance(decoded, DecodedValue):
            raise TypeError(
                'custom codec deserialize() must return DecodedValue'
            )
        if with_metadata:
            return decoded
        return decoded.value


class StateDictPretrained(_PretrainedIO):
    """Persist state dictionaries that have already been preprocessed."""

    @staticmethod
    def _select_keys(
        state_dict: Mapping[str, tp.Any],
        keys: Iterable[str] | None,
    ) -> dict[str, tp.Any]:
        if not isinstance(state_dict, Mapping):
            raise TypeError('state_dict must be a mapping')
        if keys is None:
            return dict(state_dict)
        if isinstance(keys, str):
            raise TypeError('keys must be an iterable of state-dict keys')
        selected = {}
        for key in keys:
            if not isinstance(key, str):
                raise TypeError('state-dict keys must be strings')
            if key not in state_dict:
                raise KeyError(key)
            selected[key] = state_dict[key]
        return selected

    @classmethod
    def serialize_state_dict(
        cls,
        path: PathLike,
        state_dict: Mapping[str, tp.Any],
        *,
        keys: Iterable[str] | None = None,
        format: SerializationFormat = 'safetensors',
        metadata: Mapping[str, str] | None = None,
    ) -> str:
        selected = cls._select_keys(state_dict, keys)
        return cls.serialize_to_disk(
            path,
            selected,
            metadata=metadata,
            format=format,
        )

    @classmethod
    def deserialize_state_dict(
        cls,
        path: PathLike,
        *,
        keys: Iterable[str] | None = None,
        format: SerializationFormat = 'safetensors',
    ) -> dict[str, tp.Any]:
        state_dict = cls.deserialize_from_disk(path, format=format)
        if not isinstance(state_dict, Mapping):
            raise TypeError('decoded state_dict must be a mapping')
        return cls._select_keys(state_dict, keys)


__all__ = [
    'DecodedValue',
    'PretrainedCodec',
    'SerializationFormat',
    'StateDictPretrained',
    '_PretrainedIO',
]
