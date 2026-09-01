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
"""Persistence for already-prepared quantized state."""

from __future__ import annotations

import typing as tp
from collections.abc import Mapping
from dataclasses import dataclass, field

from destiny.maestro.concerto.base import (
    DecodedValue,
    SerializationFormat,
    _PretrainedIO,
)
from destiny.utils.typing import PathLike


@dataclass(frozen=True, slots=True)
class QuantState:
    """Storage-ready quantized tensors and reconstruction metadata."""

    tensors: Mapping[str, tp.Any]
    metadata: Mapping[str, str] = field(default_factory=dict)


class QuantPretrained(_PretrainedIO):
    """Persist quantized state without performing quantization."""

    @classmethod
    def serialize_quantized(
        cls,
        path: PathLike,
        state: QuantState,
        *,
        format: SerializationFormat = 'safetensors',
    ) -> str:
        if not isinstance(state, QuantState):
            raise TypeError('state must be a QuantState')
        return cls.serialize_to_disk(
            path,
            state.tensors,
            metadata=state.metadata,
            format=format,
        )

    @classmethod
    def deserialize_quantized(
        cls,
        path: PathLike,
        *,
        format: SerializationFormat = 'safetensors',
    ) -> QuantState:
        decoded = cls.deserialize_from_disk(
            path,
            format=format,
            with_metadata=True,
        )
        if not isinstance(decoded, DecodedValue):
            raise TypeError('decoded quantized payload is invalid')
        if not isinstance(decoded.value, Mapping):
            raise TypeError('decoded quantized tensors must be a mapping')
        return QuantState(dict(decoded.value), dict(decoded.metadata))


__all__ = ['QuantPretrained', 'QuantState']
