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

"""Model-facing pretrained persistence facade."""

from __future__ import annotations

import json
import os
import re
import typing as tp
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import qwix
from huggingface_hub import split_state_dict_into_shards_factory

from destiny.maestro.concerto.base import (
    DecodedValue,
    StateDictPretrained,
)
from destiny.maestro.concerto.peft import (
    LoraPretrained,
    PeftPretrained,
    PeftState,
)
from destiny.maestro.concerto.quantization import (
    QuantPretrained,
    QuantState,
)
from destiny.utils.typing import PathLike

type PreparedState = dict[str, tp.Any] | PeftState | QuantState


class GenericPretrained(
    StateDictPretrained,
    PeftPretrained,
    QuantPretrained,
):
    """Prepare a model's current state and dispatch it to Concerto storage."""

    @staticmethod
    def _config_as_dict(config: tp.Any) -> dict[str, tp.Any]:
        if config is None:
            return {}
        if isinstance(config, Mapping):
            return dict(config)
        to_dict = getattr(config, 'to_dict', None)
        if callable(to_dict):
            value = to_dict()
            if not isinstance(value, Mapping):
                raise TypeError('config.to_dict() must return a mapping')
            return dict(value)
        if hasattr(config, '__dict__'):
            return {
                name: value
                for name, value in vars(config).items()
                if not name.startswith('_')
            }
        raise TypeError('model config must be mapping-like')

    @staticmethod
    def _qtype_name(qtype: tp.Any) -> str:
        if isinstance(qtype, str):
            return qtype
        return getattr(qtype, 'name', None) or str(qtype)

    @classmethod
    def _prepare_quant_state(
        cls,
        state_dict: Mapping[str, tp.Any],
    ) -> QuantState:
        tensors = {}
        parameters = {}
        for name, value in state_dict.items():
            if not isinstance(value, qwix.QArray):
                tensors[name] = value
                continue
            prefix = f'{name}.__qwix__'
            tensors[f'{prefix}.qvalue'] = value.qvalue
            tensors[f'{prefix}.scale'] = value.scale
            zero_point_name = None
            if value.zero_point is not None:
                zero_point_name = f'{prefix}.zero_point'
                tensors[zero_point_name] = value.zero_point
            parameters[name] = {
                'qtype': cls._qtype_name(value.qtype),
                'qvalue_dtype': value.qvalue.dtype.name,
                'qvalue': f'{prefix}.qvalue',
                'scale': f'{prefix}.scale',
                'zero_point': zero_point_name,
            }
        if not parameters:
            raise ValueError('state_dict contains no quantized values')
        metadata = {
            'destiny.quantization_format': 'qwix',
            'destiny.quantization_version': '1',
            'destiny.quantization_parameters': json.dumps(parameters),
        }
        return QuantState(tensors, metadata)

    @staticmethod
    def _restore_quant_state(state: QuantState) -> dict[str, tp.Any]:
        if state.metadata.get('destiny.quantization_format') != 'qwix':
            raise ValueError('unsupported quantized checkpoint format')
        if state.metadata.get('destiny.quantization_version') != '1':
            raise ValueError('unsupported quantized checkpoint version')
        encoded = state.metadata.get('destiny.quantization_parameters')
        if encoded is None:
            raise ValueError('quantized checkpoint has no parameter metadata')
        parameters = json.loads(encoded)
        if not isinstance(parameters, dict):
            raise TypeError('quantized parameter metadata must be an object')

        component_names = {
            component
            for specification in parameters.values()
            for component in (
                specification.get('qvalue'),
                specification.get('scale'),
                specification.get('zero_point'),
            )
            if component is not None
        }
        restored = {
            name: value
            for name, value in state.tensors.items()
            if name not in component_names
        }
        for name, specification in parameters.items():
            qvalue_name = specification.get('qvalue')
            scale_name = specification.get('scale')
            if qvalue_name not in state.tensors or scale_name not in state.tensors:
                raise ValueError(
                    f'quantized parameter {name!r} is missing components'
                )
            qvalue_dtype = jnp.dtype(specification['qvalue_dtype'])
            zero_point_name = specification.get('zero_point')
            restored[name] = qwix.QArray(
                qvalue=jnp.asarray(
                    state.tensors[qvalue_name],
                    dtype=qvalue_dtype,
                ),
                scale=jnp.asarray(state.tensors[scale_name]),
                zero_point=(
                    jnp.asarray(
                        state.tensors[zero_point_name],
                        dtype=qvalue_dtype,
                    )
                    if zero_point_name is not None
                    else None
                ),
                qtype=specification.get('qtype'),
            )
        return restored

    def prepare_pretrained_state(self) -> PreparedState:
        flat_state_dict = getattr(self, 'flat_state_dict', None)
        if not callable(flat_state_dict):
            raise TypeError(
                f'{type(self).__name__} must provide flat_state_dict()'
            )
        state = dict(flat_state_dict())
        peft_config = getattr(self, 'peft_config', None)
        if peft_config is not None:
            if not isinstance(peft_config, Mapping):
                raise TypeError('model peft_config must be a mapping')
            peft_type = str(peft_config.get('peft_type', '')).upper()
            if peft_type != 'LORA':
                raise NotImplementedError(
                    f'unsupported PEFT state: {peft_type or "<missing>"}'
                )
            return LoraPretrained.prepare_lora_state(
                state,
                config=peft_config,
            )
        if any(isinstance(value, qwix.QArray) for value in state.values()):
            return self._prepare_quant_state(state)
        return state

    @staticmethod
    def _host_prepared_state(state: PreparedState) -> PreparedState:
        def snapshot(tensors: Mapping[str, tp.Any]) -> dict[str, tp.Any]:
            hosted = jax.device_get(dict(tensors))
            return {
                name: (
                    np.array(value, copy=True)
                    if isinstance(value, np.ndarray)
                    else value
                )
                for name, value in hosted.items()
            }

        if isinstance(state, PeftState):
            return PeftState(snapshot(state.tensors), dict(state.metadata))
        if isinstance(state, QuantState):
            return QuantState(snapshot(state.tensors), dict(state.metadata))
        return snapshot(state)

    def _checkpoint_snapshot(self) -> dict[str, tp.Any]:
        return {
            'state': self._host_prepared_state(
                self.prepare_pretrained_state()
            ),
            'config': self._config_as_dict(getattr(self, 'config', None)),
        }

    @staticmethod
    def _prepared_tensors(state: PreparedState) -> Mapping[str, tp.Any]:
        if isinstance(state, (PeftState, QuantState)):
            return state.tensors
        if isinstance(state, Mapping):
            return state
        raise TypeError('checkpoint state is not a supported representation')

    @classmethod
    def _serialize_prepared(
        cls,
        path: PathLike,
        state: PreparedState,
    ) -> str:
        if isinstance(state, PeftState):
            return cls.serialize_peft(path, state)
        if isinstance(state, QuantState):
            return cls.serialize_quantized(path, state)
        return cls.serialize_state_dict(
            path,
            state,
            metadata={'destiny.format': 'native'},
        )

    @classmethod
    def _slice_prepared(
        cls,
        state: PreparedState,
        names: list[str],
    ) -> PreparedState:
        tensors = cls._prepared_tensors(state)
        selected = {name: tensors[name] for name in names}
        if isinstance(state, PeftState):
            return PeftState(selected, state.metadata)
        if isinstance(state, QuantState):
            return QuantState(selected, state.metadata)
        return selected

    @classmethod
    def _save_weight_files(
        cls,
        directory: str,
        state: PreparedState,
        *,
        filename: str,
        max_shard_size: int | str,
    ) -> tuple[str, ...]:
        tensors = dict(cls._prepared_tensors(state))
        if not tensors:
            raise ValueError('cannot save an empty pretrained state')
        stem, extension = os.path.splitext(filename)
        split = split_state_dict_into_shards_factory(
            tensors,
            get_storage_size=lambda value: int(value.nbytes),
            filename_pattern=f'{stem}{{suffix}}{extension}',
            max_shard_size=max_shard_size,
        )
        paths = []
        for shard_name, names in split.filename_to_tensors.items():
            shard_path = os.path.join(directory, shard_name)
            cls._serialize_prepared(
                shard_path,
                cls._slice_prepared(state, names),
            )
            paths.append(shard_path)
        if split.is_sharded:
            index_path = os.path.join(
                directory,
                f'{filename}.index.json',
            )
            cls.serialize_to_disk(
                index_path,
                {
                    'metadata': split.metadata,
                    'weight_map': split.tensor_to_filename,
                },
            )
            paths.append(index_path)

        active_names = {os.path.basename(path) for path in paths}
        index_name = f'{filename}.index.json'
        shard_pattern = re.compile(
            rf'{re.escape(stem)}-\d{{5}}-of-\d{{5}}'
            rf'{re.escape(extension)}'
        )
        for existing_name in os.listdir(directory):
            belongs_to_checkpoint = (
                existing_name == filename
                or existing_name == index_name
                or shard_pattern.fullmatch(existing_name) is not None
            )
            if (
                belongs_to_checkpoint
                and existing_name not in active_names
            ):
                os.remove(os.path.join(directory, existing_name))
        return tuple(paths)

    @classmethod
    def _save_pretrained_snapshot(
        cls,
        snapshot: Mapping[str, tp.Any],
        path: PathLike,
        *,
        max_shard_size: int | str = '10GB',
    ) -> tuple[str, ...]:
        if not isinstance(snapshot, Mapping):
            raise TypeError('checkpoint snapshot must be a mapping')
        state = snapshot.get('state')
        if not isinstance(state, (Mapping, PeftState, QuantState)):
            raise TypeError('checkpoint snapshot has no prepared state')
        directory = os.path.abspath(os.fspath(path))
        os.makedirs(directory, exist_ok=True)
        config_path = cls.serialize_to_disk(
            os.path.join(directory, 'config.json'),
            snapshot.get('config', {}),
        )

        extra_paths = []
        filename = 'model.safetensors'
        if isinstance(state, PeftState):
            filename = 'adapter_model.safetensors'
            encoded = state.metadata.get('destiny.peft_config')
            adapter_config = (
                json.loads(encoded)
                if encoded is not None
                else {
                    'peft_type': state.metadata.get(
                        'destiny.peft_type',
                        'UNKNOWN',
                    )
                }
            )
            extra_paths.append(
                cls.serialize_to_disk(
                    os.path.join(directory, 'adapter_config.json'),
                    adapter_config,
                )
            )
        elif isinstance(state, QuantState):
            encoded = state.metadata.get(
                'destiny.quantization_parameters',
                '{}',
            )
            extra_paths.append(
                cls.serialize_to_disk(
                    os.path.join(directory, 'quantization_config.json'),
                    {
                        'format': state.metadata.get(
                            'destiny.quantization_format'
                        ),
                        'version': state.metadata.get(
                            'destiny.quantization_version'
                        ),
                        'parameters': json.loads(encoded),
                    },
                )
            )
        weight_paths = cls._save_weight_files(
            directory,
            state,
            filename=filename,
            max_shard_size=max_shard_size,
        )
        stale_sidecars = []
        if not isinstance(state, PeftState):
            stale_sidecars.append('adapter_config.json')
        if not isinstance(state, QuantState):
            stale_sidecars.append('quantization_config.json')
        for name in stale_sidecars:
            stale_path = os.path.join(directory, name)
            if os.path.isfile(stale_path):
                os.remove(stale_path)
        return (config_path, *extra_paths, *weight_paths)

    def _save_config(self, path: PathLike) -> str:
        return self.serialize_to_disk(
            os.path.join(os.fspath(path), 'config.json'),
            self._config_as_dict(getattr(self, 'config', None)),
        )

    def save_pretrained(
        self,
        path: PathLike,
        *,
        max_shard_size: int | str = '10GB',
    ) -> tuple[str, ...]:
        """Save the current model, adapter, or quantized state to a directory."""
        return type(self)._save_pretrained_snapshot(
            self._checkpoint_snapshot(),
            path,
            max_shard_size=max_shard_size,
        )

    @classmethod
    def _checkpoint_files(
        cls,
        directory: str,
        filename: str,
    ) -> list[str]:
        index_path = os.path.join(directory, f'{filename}.index.json')
        if not os.path.isfile(index_path):
            path = os.path.join(directory, filename)
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            return [path]
        index = cls.deserialize_from_disk(index_path)
        if not isinstance(index, Mapping):
            raise TypeError('checkpoint index must be an object')
        weight_map = index.get('weight_map')
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ValueError('checkpoint index has no weight_map')
        return [
            os.path.join(directory, name)
            for name in dict.fromkeys(weight_map.values())
        ]

    @classmethod
    def _load_prepared_files(
        cls,
        paths: list[str],
        kind: str,
    ) -> PreparedState:
        tensors = {}
        metadata: Mapping[str, str] = {}
        for path in paths:
            if kind == 'peft':
                decoded = cls.deserialize_peft(path)
            elif kind == 'quant':
                decoded = cls.deserialize_quantized(path)
            else:
                decoded = DecodedValue(
                    cls.deserialize_state_dict(path),
                    {},
                )
            shard_tensors = (
                decoded.tensors
                if isinstance(decoded, (PeftState, QuantState))
                else decoded.value
            )
            shard_metadata = decoded.metadata
            duplicates = set(tensors).intersection(shard_tensors)
            if duplicates:
                name = min(duplicates)
                raise ValueError(f'duplicate checkpoint tensor: {name}')
            tensors.update(shard_tensors)
            if metadata and dict(metadata) != dict(shard_metadata):
                raise ValueError('checkpoint shards have inconsistent metadata')
            metadata = shard_metadata
        if kind == 'peft':
            return PeftState(tensors, metadata)
        if kind == 'quant':
            return QuantState(tensors, metadata)
        return tensors

    def _apply_flat_state_dict(
        self,
        state_dict: Mapping[str, tp.Any],
        *,
        expected_keys: set[str],
        strict: bool,
    ) -> None:
        flat_parameter_dict = getattr(self, 'flat_parameter_dict', None)
        if not callable(flat_parameter_dict):
            raise TypeError(
                f'{type(self).__name__} must provide flat_parameter_dict()'
            )
        parameters = dict(flat_parameter_dict())
        unexpected = sorted(set(state_dict) - set(parameters))
        missing = sorted(expected_keys - set(state_dict))
        if strict and unexpected:
            raise ValueError(
                'checkpoint contains unexpected tensors: '
                + ', '.join(unexpected[:8])
            )
        if strict and missing:
            raise ValueError(
                'checkpoint is missing tensors: ' + ', '.join(missing[:8])
            )

        for name, value in state_dict.items():
            parameter = parameters.get(name)
            if parameter is None:
                continue
            target = parameter.value
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f'checkpoint tensor {name!r} has shape {value.shape}, '
                    f'expected {target.shape}'
                )
            if isinstance(value, qwix.QArray):
                if isinstance(target, qwix.QArray):
                    value = jax.tree.map(
                        lambda source, destination: jax.device_put(
                            source,
                            getattr(destination, 'sharding', None),
                        ),
                        value,
                        target,
                    )
                parameter.value = value
                continue
            array = jnp.asarray(value, dtype=target.dtype)
            sharding = getattr(target, 'sharding', None)
            if sharding is not None:
                array = jax.device_put(array, sharding)
            parameter.value = array

    def load_pretrained(
        self,
        path: PathLike,
        *,
        strict: bool = True,
    ) -> tp.Self:
        """Load a local Concerto checkpoint into this model in place."""
        directory = os.path.abspath(os.fspath(path))
        if not os.path.isdir(directory):
            raise NotADirectoryError(directory)
        if os.path.isfile(os.path.join(directory, 'adapter_config.json')):
            kind = 'peft'
            filename = 'adapter_model.safetensors'
        elif os.path.isfile(
            os.path.join(directory, 'quantization_config.json')
        ):
            kind = 'quant'
            filename = 'model.safetensors'
        else:
            kind = 'state_dict'
            filename = 'model.safetensors'

        prepared = type(self)._load_prepared_files(
            type(self)._checkpoint_files(directory, filename),
            kind,
        )
        parameters = set(self.flat_parameter_dict())
        if isinstance(prepared, PeftState):
            state_dict = dict(prepared.tensors)
            expected = {
                name
                for name in parameters
                if name.endswith(('.lora_A', '.lora_B'))
            }
        elif isinstance(prepared, QuantState):
            state_dict = self._restore_quant_state(prepared)
            expected = parameters
        else:
            state_dict = prepared
            expected = parameters
        self._apply_flat_state_dict(
            state_dict,
            expected_keys=expected,
            strict=strict,
        )
        return self

    @classmethod
    def from_pretrained_directory(
        cls,
        path: PathLike,
        config: tp.Any,
        **kwargs: tp.Any,
    ) -> tp.Self:
        """Construct ``cls`` from a resolved local checkpoint directory."""
        from destiny.maestro.concerto.materialization import (
            materialize_pretrained,
        )

        return materialize_pretrained(
            cls,
            path,
            config,
            **kwargs,
        )


__all__ = ['GenericPretrained', 'PreparedState']
