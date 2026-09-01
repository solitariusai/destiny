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

"""Canonical PEFT persistence and LoRA preprocessing."""

from __future__ import annotations

import json
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
class PeftState:
    """Method-agnostic PEFT tensor state and reconstruction metadata."""

    tensors: Mapping[str, tp.Any]
    metadata: Mapping[str, str] = field(default_factory=dict)


class PeftPretrained(_PretrainedIO):
    """Persistence for canonical :class:`PeftState` values."""

    @classmethod
    def serialize_peft(
        cls,
        path: PathLike,
        state: PeftState,
        *,
        format: SerializationFormat = 'safetensors',
    ) -> str:
        if not isinstance(state, PeftState):
            raise TypeError('state must be a PeftState')
        return cls.serialize_to_disk(
            path,
            state.tensors,
            metadata=state.metadata,
            format=format,
        )

    @classmethod
    def deserialize_peft(
        cls,
        path: PathLike,
        *,
        format: SerializationFormat = 'safetensors',
    ) -> PeftState:
        decoded = cls.deserialize_from_disk(
            path,
            format=format,
            with_metadata=True,
        )
        if not isinstance(decoded, DecodedValue):
            raise TypeError('decoded PEFT payload is invalid')
        if not isinstance(decoded.value, Mapping):
            raise TypeError('decoded PEFT tensors must be a mapping')
        return PeftState(dict(decoded.value), dict(decoded.metadata))


class LoraPretrained(PeftPretrained):
    """Convert LoRA-specific tensor state to canonical PEFT state."""

    @staticmethod
    def prepare_lora_state(
        state_dict: Mapping[str, tp.Any],
        *,
        config: Mapping[str, tp.Any] | None = None,
    ) -> PeftState:
        if not isinstance(state_dict, Mapping):
            raise TypeError('state_dict must be a mapping')
        tensors = {
            name: value
            for name, value in state_dict.items()
            if name.endswith(('.lora_A', '.lora_B'))
        }
        if not tensors:
            raise ValueError('state_dict contains no LoRA tensors')
        metadata = {'destiny.peft_type': 'LORA'}
        if config is not None:
            metadata['destiny.peft_config'] = json.dumps(
                dict(config),
                default=str,
            )
        return PeftState(tensors, metadata)

    @classmethod
    def serialize_lora(
        cls,
        path: PathLike,
        state_dict: Mapping[str, tp.Any],
        *,
        config: Mapping[str, tp.Any] | None = None,
        format: SerializationFormat = 'safetensors',
    ) -> str:
        """Preprocess LoRA tensors, then delegate canonical persistence."""
        state = cls.prepare_lora_state(state_dict, config=config)
        return cls.serialize_peft(path, state, format=format)


__all__ = ['LoraPretrained', 'PeftPretrained', 'PeftState']
