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
from __future__ import annotations
import typing as tp
import time
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
import os
import json
import re
import tempfile
import copy
import collections
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import qwix
from huggingface_hub import (
    HfApi,
    hf_hub_download,
    split_state_dict_into_shards_factory,
)
from safetensors.numpy import save_file
import numpy as np
from safetensors import safe_open
from ..utils.quantization import (
    quantize_embedding_weight,
    quantize_linear_weight,
    resolve_quantization_rule,
)
from huggingface_hub import repo_info

from ..utils.quantization import merge_quantization
from ..utils.weights import map_state_dict
from ..utils.logging import is_jax_rank_zero, tqdm
from ..utils.sharding import create_sharding

from taktiny import nn
from taktiny.nn.module import iter_children
from taktiny.utils.format import parse_size
from taktiny.utils.typing import AxisNames, DType, PathLike, LogicalRules
from taktiny.nn.lora import LoRALinear


_MISSING = object()

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


def _grouped_stack_layout(
    state: tp.Mapping[str, tp.Any],
) -> dict[tuple[str, ...], tuple[int, ...]]:
    sizes: dict[tuple[str, ...], dict[int, int]] = {}
    for name, value in state.items():
        parts = name.split('.')
        for position in range(len(parts) - 2):
            if (
                parts[position] != 'groups'
                or not parts[position + 1].isdigit()
                or parts[position + 2] != 'stacked'
            ):
                continue
            shape = getattr(value, 'shape', ())
            if not shape:
                raise ValueError(
                    f'Grouped stacked parameter {name!r} has no layer axis'
                )
            root = tuple(parts[:position])
            group_index = int(parts[position + 1])
            group_sizes = sizes.setdefault(root, {})
            previous = group_sizes.setdefault(group_index, shape[0])
            if previous != shape[0]:
                raise ValueError(
                    f'Grouped stack {".".join(root)!r} has inconsistent '
                    f'size for group {group_index}'
                )
            break

    layouts = {}
    for root, indexed_sizes in sizes.items():
        indices = sorted(indexed_sizes)
        if indices != list(range(len(indices))):
            raise ValueError(
                f'Grouped stack {".".join(root)!r} has non-contiguous groups'
            )
        layouts[root] = tuple(indexed_sizes[index] for index in indices)
    return layouts


def _resolve_stacked_parameter(
    name: str,
    parameters: tp.Mapping[str, tp.Any],
    grouped_layouts: tp.Mapping[tuple[str, ...], tuple[int, ...]],
) -> tuple[str, int] | None:
    parts = name.split('.')
    for position, part in enumerate(parts):
        if not part.isdigit():
            continue
        layer_index = int(part)

        stacked_parts = list(parts)
        stacked_parts[position] = 'stacked'
        stacked_name = '.'.join(stacked_parts)
        if stacked_name in parameters:
            if layer_index < parameters[stacked_name].shape[0]:
                return stacked_name, layer_index
            continue

        root = tuple(parts[:position])
        group_sizes = grouped_layouts.get(root)
        if group_sizes is None:
            continue
        offset = 0
        for group_index, group_size in enumerate(group_sizes):
            if layer_index < offset + group_size:
                local_index = layer_index - offset
                grouped_parts = [
                    *root,
                    'groups',
                    str(group_index),
                    'stacked',
                    *parts[position + 1:],
                ]
                grouped_name = '.'.join(grouped_parts)
                if grouped_name in parameters:
                    return grouped_name, local_index
                break
            offset += group_size
    return None


class PretrainedModel(nn.Module):
    """
    Base class for models that load and save pretrained checkpoints.

    Full models are serialized as Safetensors together with a weight index.
    Qwix arrays retain their quantized components and reconstruction metadata.
    LoRA-transformed models instead save adapter tensors and reconstruction
    metadata. Loading first constructs an abstract parameter tree with
    ``jax.eval_shape``, then maps checkpoint names to module paths, applies any
    requested quantization, and places arrays using parameter sharding metadata.

    Subclasses are expected to accept ``config`` and ``rngs`` in their
    constructor. They may provide module-mapping rules to translate external
    checkpoint names and may expose default logical sharding rules.
    """

    @classmethod
    def _resolve_sharding_rules(cls) -> tp.Any:
        """Resolve the class's default logical-to-mesh sharding rules.

        Architectures declare their rules under ``_default_sharding_rules``;
        loading consults this attribute so multi-device placement uses the
        same rules the modules were built with.
        """
        return getattr(cls, '_default_sharding_rules', None)

    @staticmethod
    def _config_as_dict(config: tp.Any) -> tp.Any:
        if config is None:
            return {}
        if isinstance(config, dict):
            return dict(config)
        to_dict = getattr(config, 'to_dict', None)
        if callable(to_dict):
            return to_dict()
        return {
            key: value
            for key, value in vars(config).items()
            if not key.startswith('_')
        }

    def _config_dict(self) -> tp.Any:
        return self._config_as_dict(getattr(self, 'config', None))

    def _checkpoint_config(self) -> tp.Any:
        """Config written by ``save_pretrained``.

        Models materialized through ``from_pretrained`` keep a pristine copy
        of the configuration they were loaded with, untouched by library
        options such as quantization shortcuts, and that original
        configuration is what gets saved back. The only amendment is the
        floating dtype the parameters were materialized with, so that
        reloading casts correctly. Directly instantiated models fall back to
        their working config.
        """
        original = getattr(self, 'original_config_dict', None)
        if not (isinstance(original, dict) and original):
            return copy.deepcopy(self._config_dict())
        saved = copy.deepcopy(original)
        override = getattr(self, 'loaded_dtype_override', None)
        if override:
            dtype_name = (
                override
                if isinstance(override, str)
                else getattr(override, 'name', None) or str(override)
            )
            saved['torch_dtype'] = dtype_name
            saved['dtype'] = dtype_name
        return saved

    def _save_config(self, path: str) -> tp.Any:
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'w') as config_file:
            json.dump(
                self._config_dict(),
                config_file,
                indent=2,
                default=str,
            )
        return config_path

    @staticmethod
    def _qtype_name(qtype: tp.Any) -> tp.Any:
        if isinstance(qtype, str):
            return qtype
        return jnp.dtype(qtype).name

    @staticmethod
    def _safetensors_qvalue(array: tp.Any) -> tp.Any:
        dtype = array.dtype
        if jnp.issubdtype(dtype, jnp.signedinteger):
            storage_dtype = np.int8
        elif jnp.issubdtype(dtype, jnp.unsignedinteger):
            storage_dtype = np.uint8
        elif jnp.issubdtype(dtype, jnp.floating):
            storage_dtype = np.float16
        else:
            raise TypeError(
                f'Unsupported Qwix qvalue dtype for serialization: {dtype}'
            )
        return np.asarray(jax.device_get(array), dtype=storage_dtype)

    @classmethod
    def _encode_qwix_state(cls, state: tp.Any) -> tuple[tp.Any, ...]:
        encoded = {}
        parameters = {}

        for name, value in state.items():
            if not isinstance(value, qwix.QArray):
                encoded[name] = value
                continue

            component_prefix = f'{name}.__qwix__'
            qvalue_name = f'{component_prefix}.qvalue'
            scale_name = f'{component_prefix}.scale'
            zero_point_name = (
                f'{component_prefix}.zero_point'
                if value.zero_point is not None
                else None
            )
            encoded[qvalue_name] = cls._safetensors_qvalue(value.qvalue)
            encoded[scale_name] = value.scale
            if zero_point_name is not None:
                encoded[zero_point_name] = cls._safetensors_qvalue(
                    value.zero_point
                )
            parameters[name] = {
                'qtype': cls._qtype_name(value.qtype),
                'qvalue_dtype': value.qvalue.dtype.name,
                'qvalue': qvalue_name,
                'scale': scale_name,
                'zero_point': zero_point_name,
            }

        metadata = None
        if parameters:
            metadata = {
                'format': 'taktiny-qwix',
                'version': 1,
                'parameters': parameters,
            }
        return encoded, metadata

    def _lora_state_dict(self) -> tp.Any:
        state = {}

        def collect(module: tp.Any, prefix: str='') -> None:
            for name, child in iter_children(module):
                full_name = f'{prefix}.{name}' if prefix else name
                if isinstance(child, LoRALinear):
                    state[f'{full_name}.lora_A'] = child.lora_A.value
                    state[f'{full_name}.lora_B'] = child.lora_B.value
                elif isinstance(child, nn.Module):
                    collect(child, full_name)

        collect(self)
        return state

    @staticmethod
    def _host_state_snapshot(state: tp.Any) -> tp.Any:
        # A single batched transfer avoids one blocking round trip per
        # tensor on accelerator backends.
        hosted = jax.device_get(state)

        def stabilize_leaf(value: tp.Any) -> tp.Any:
            # Fetched arrays may view device buffers that later operations
            # can reuse, so views are copied to keep the snapshot stable.
            if isinstance(value, np.ndarray) and not value.flags['OWNDATA']:
                return np.array(value, copy=True)
            return value

        return jax.tree.map(stabilize_leaf, hosted)

    # TODO: refactor _invert_checkpoint_names
    @staticmethod
    def _invert_checkpoint_names(state: tp.Dict, module_map: tp.List) -> tp.Any:
        """Restore source-format checkpoint tensor names for saving.

        The mapping rules applied while loading are undone in reverse order.
        Only plain renames (two-element rules) are invertible; rules that
        carry a transform are left exactly as they were loaded.
        """
        if not module_map:
            return state
        inverted = dict(state)
        for rule in reversed(module_map):
            if len(rule) != 2:
                continue
            source_pattern, target_pattern = rule
            if (
                not isinstance(source_pattern, str)
                or not isinstance(target_pattern, str)
                or source_pattern == target_pattern
            ):
                continue
            remapped = {}
            for key, value in inverted.items():
                new_key = key.replace(target_pattern, source_pattern)
                if new_key in remapped:
                    raise ValueError(
                        'Reversing module_map collapsed two checkpoint '
                        f'tensors onto the same name {new_key!r}'
                    )
                remapped[new_key] = value
            inverted = remapped
        return inverted

    @staticmethod
    def _fired_rename_rules(
        checkpoint_keys: tp.Any,
        module_map: tp.Any,
    ) -> tp.Any:
        """Select rename rules that matched the loaded checkpoint's names.

        ``from_pretrained`` remembers its module map so saving can restore
        source spellings, but a rule whose source pattern never appears in
        the checkpoint (e.g. a multimodal ``model.language_model.`` rule used
        to load a text-only checkpoint) must not be inverted at save time;
        doing so would rename tensors that loading never touched.
        """
        simulated = set(checkpoint_keys)
        fired = []
        for rule in module_map:
            if len(rule) != 2:
                continue
            source_pattern, target_pattern = rule
            if (
                not isinstance(source_pattern, str)
                or source_pattern == target_pattern
            ):
                continue
            if any(source_pattern in key for key in simulated):
                fired.append((source_pattern, target_pattern))
                simulated = {
                    key.replace(source_pattern, target_pattern)
                    for key in simulated
                }
        return fired

    def _checkpoint_snapshot(self) -> dict[tp.Any, tp.Any]:
        """Capture stable host state for background checkpoint writing."""
        adapter_state = self._expand_stacked_state_dict(
            self._host_state_snapshot(self._lora_state_dict())
        )
        if adapter_state:
            peft_config = getattr(self, 'peft_config', None)
            if peft_config is None:
                raise ValueError(
                    'LoRA modules were found but PEFT configuration metadata '
                    'is missing; apply LoRA through Takt.apply_peft'
                )
            return {
                'kind': 'adapter',
                'config': self._checkpoint_config(),
                'peft_config': copy.deepcopy(peft_config),
                'module_map': list(
                    getattr(self, 'checkpoint_module_map', None) or []
                ),
                'state': adapter_state,
            }

        # Host tensors are fetched before stacked parameters are expanded
        # into numbered layers so that each stacked tensor transfers once.
        state = self._expand_stacked_state_dict(
            self._host_state_snapshot(self.flat_state_dict())
        )
        return {
            'kind': 'model',
            'config': self._checkpoint_config(),
            'module_map': list(
                getattr(self, 'checkpoint_module_map', None) or []
            ),
            'state': state,
        }

    @classmethod
    def _save_pretrained_snapshot(
        cls,
        snapshot: tp.Dict,
        path: str,
        *,
        max_shard_size: str='10GB',
        module_map: tp.Any=None,
    ) -> tuple[tp.Any, ...]:
        os.makedirs(path, exist_ok=True)
        model_config_path = os.path.join(path, 'config.json')
        with open(model_config_path, 'w') as config_file:
            json.dump(
                snapshot['config'],
                config_file,
                indent=2,
                default=str,
            )

        if module_map is None:
            module_map = snapshot.get('module_map')
        state = cls._invert_checkpoint_names(
            snapshot['state'],
            module_map,
        )

        if snapshot['kind'] == 'adapter':
            config_path = os.path.join(path, 'adapter_config.json')
            with open(config_path, 'w') as config_file:
                json.dump(snapshot['peft_config'], config_file, indent=2)
            adapter_paths = cls._save_safetensors(
                state,
                path,
                'adapter_model.safetensors',
                max_shard_size=max_shard_size,
            )
            return (
                model_config_path,
                config_path,
                *adapter_paths,
            )

        state_dict, quantization_metadata = cls._encode_qwix_state(
            state
        )
        quantization_path = os.path.join(
            path,
            'quantization_config.json',
        )
        if quantization_metadata is not None:
            with open(quantization_path, 'w') as quantization_file:
                json.dump(
                    quantization_metadata,
                    quantization_file,
                    indent=2,
                )
        elif os.path.isfile(quantization_path):
            os.remove(quantization_path)
            quantization_path = None
        else:
            quantization_path = None
        checkpoint_paths = cls._save_safetensors(
            state_dict,
            path,
            'model.safetensors',
            max_shard_size=max_shard_size,
            always_write_index=False,
        )
        return (
            model_config_path,
            *((quantization_path,) if quantization_path else ()),
            *checkpoint_paths,
        )

    @classmethod
    def _expand_stacked_entries(
        cls,
        state: tp.Any,
    ) -> list[tuple[str, str, int | None]]:
        """Ordered ``(expanded_name, source_name, layer_index)`` triples.

        ``layer_index`` is ``None`` for parameters outside a ``SeqStack``.
        The name rules mirror ``_expand_stacked_state_dict``, which consumes
        this layout so both stay in lockstep.
        """
        grouped_layouts = _grouped_stack_layout(state)
        layout: list[tuple[str, tp.Any]] = []
        stacked_groups: dict = {}

        for name, value in state.items():
            parts = name.split('.')
            if 'stacked' not in parts:
                layout.append(('parameter', (name, name, None)))
                continue

            stacked_index = parts.index('stacked')
            group_key = (tuple(parts[:stacked_index]), stacked_index)
            if group_key not in stacked_groups:
                stacked_groups[group_key] = []
                layout.append(('group', group_key))
            stacked_groups[group_key].append((parts, value))

        def group_entries(
            group_key: tuple,
        ) -> list[tuple[str, str, int]]:
            group = stacked_groups[group_key]
            stacked_index = group_key[1]
            num_layers = None
            for parts, value in group:
                name = '.'.join(parts)
                if not getattr(value, 'shape', ()):
                    raise ValueError(
                        f'Stacked parameter {name!r} has no leading layer axis'
                    )
                if num_layers is None:
                    num_layers = value.shape[0]
                elif value.shape[0] != num_layers:
                    raise ValueError(
                        'Parameters in the same stack have inconsistent '
                        f'layer counts: expected {num_layers}, found '
                        f'{value.shape[0]} for {name!r}'
                    )

            entries = []
            for layer_index in range(num_layers):
                for parts, value in group:
                    layer_parts = list(parts)
                    if (
                        stacked_index >= 2
                        and parts[stacked_index - 2] == 'groups'
                        and parts[stacked_index - 1].isdigit()
                    ):
                        root = tuple(parts[:stacked_index - 2])
                        group_index = int(parts[stacked_index - 1])
                        group_sizes = grouped_layouts[root]
                        global_index = (
                            sum(group_sizes[:group_index]) + layer_index
                        )
                        layer_parts = [
                            *parts[:stacked_index - 2],
                            str(global_index),
                            *parts[stacked_index + 1:],
                        ]
                    else:
                        layer_parts[stacked_index] = str(layer_index)
                    entries.append((
                        '.'.join(layer_parts),
                        '.'.join(parts),
                        layer_index,
                    ))
            return entries

        expanded_entries: list[tuple[str, str, int | None]] = []
        for kind, payload in layout:
            if kind == 'parameter':
                expanded_entries.append(payload)
            else:
                expanded_entries.extend(group_entries(payload))

        return expanded_entries

    @classmethod
    def _expand_stacked_state_dict(cls, state: tp.Any) -> tp.Any:
        return {
            name: (
                state[source]
                if index is None
                else state[source][index]
            )
            for name, source, index in cls._expand_stacked_entries(state)
        }


    @staticmethod
    def _save_safetensors_exp(
        state: tp.Dict,
        path: PathLike,
        filename: str,
        *,
        max_shard_byte_size: float,
    ) -> tp.Tuple[str]:
        filename, extension = os.path.splitext(filename)
        
        import math
        save_arrays = {}
        paths = []
        weight_map = {}
        accm_byte_size = 0
        num_shards = 1
        shard_index = 0
        for k, v in state.items():
            accm_byte_size += v.nbytes

        if accm_byte_size > max_shard_byte_size:
            num_shards = math.ceil(accm_byte_size / max_shard_byte_size)

        accm_byte_size = 0

        def _get_format_number(n):
            n_str = str(n)
            remain_zeros = 5 - len(n_str)
            return f"{"0" * remain_zeros}{n_str}"
        
        def _get_current_shard_name():
            if num_shards > 1:
                curr_index_str = _get_format_number(shard_index)
                num_shard_str = _get_format_number(num_shards)
                return f"{filename}-{curr_index_str}-of-{num_shard_str}.{extension}"

            return f"{filename}.{extension}"

        def _serialize(path: PathLike, state_dict: tp.Dict):
            save_file(state_dict, path)
            paths.append(path.__str__())

        for k, v in tqdm(state.items()):
            if 'stacked' in k:
                num_stacks = v.shape[0]
                byte_size_avg = v.nbytes / num_stacks
            
                for i in range(num_stacks):
                    array = v.at[i]
                    p = os.path.join(path, _get_current_shard_name())
                    if accm_byte_size > max_shard_byte_size:
                        _serialize(p, save_arrays)
                        accm_byte_size = 0
                        shard_index += 1
                        save_arrays.clear()

                    layer_key = k.replace('stacked', str(i))
                    weight_map[layer_key] = p.__str__()
                    save_arrays[layer_key] = array
                    accm_byte_size += byte_size_avg
                    
                continue

            p = os.path.join(path, _get_current_shard_name())
            if accm_byte_size > max_shard_byte_size:
                _serialize(p, save_arrays)
                accm_byte_size = 0
                shard_index += 1
                save_arrays.clear()

            weight_map[k] = p.__str__()
            save_arrays[k] = v
            accm_byte_size += v.nbytes

        if len(save_arrays) > 0:
            p = os.path.join(path, _get_current_shard_name())
            _serialize(p, save_arrays)

        if num_shards > 1:
            p_index = os.path.join(path, f'{filename}.{extension}.index.json')
            with open(p_index, 'w') as f:
                json.dump({
                    'weight_map': weight_map
                }, f, indent=2)

        return tuple(paths)

    @staticmethod
    def _save_safetensors(
        state: tp.Dict,
        path: PathLike,
        filename: str,
        *,
        max_shard_size: str,
        always_write_index: bool=False,
    ) -> tp.Any:
        stem, extension = os.path.splitext(filename)
        split = split_state_dict_into_shards_factory(
            state,
            get_storage_size=lambda value: int(value.nbytes),
            filename_pattern=f'{stem}{{suffix}}{extension}',
            max_shard_size=max_shard_size,
        )

        shard_pattern = re.compile(
            rf'{re.escape(stem)}-\d{{5}}-of-\d{{5}}'
            rf'{re.escape(extension)}'
        )
        for existing_filename in os.listdir(path):
            if (
                existing_filename == filename
                or shard_pattern.fullmatch(existing_filename)
                or existing_filename == f'{filename}.index.json'
            ):
                os.remove(os.path.join(path, existing_filename))

        saved_paths = []
        shard_items = list(split.filename_to_tensors.items())

        def write_shard(item: tp.Any) -> str:
            shard_filename, tensor_names = item
            shard_path = os.path.join(path, shard_filename)
            save_file(
                {name: state[name] for name in tensor_names},
                shard_path,
            )
            return shard_path

        # Shards are serialized concurrently; the Rust writer releases the
        # GIL, so large sharded checkpoints overlap disk writes.
        workers = max(1, min(len(shard_items), 8))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for shard_path in executor.map(write_shard, shard_items):
                saved_paths.append(shard_path)

        if split.is_sharded or always_write_index:
            index_path = os.path.join(
                path,
                f'{filename}.index.json',
            )
            with open(index_path, 'w') as index_file:
                json.dump(
                    {
                        'metadata': split.metadata,
                        'weight_map': split.tensor_to_filename,
                    },
                    index_file,
                    indent=2,
                )
            saved_paths.append(index_path)

        return tuple(saved_paths)

    @staticmethod
    def _resident_nbytes(value: tp.Any) -> int:
        """Host/device bytes actually occupied by ``value``."""
        if isinstance(value, qwix.QArray):
            total = int(value.qvalue.nbytes) + int(value.scale.nbytes)
            if value.zero_point is not None:
                total += int(value.zero_point.nbytes)
            return total
        return int(value.nbytes)

    @staticmethod
    def _storage_nbytes(value: tp.Any) -> int:
        """Bytes ``_encode_qwix_state``/safetensors will emit for ``value``."""
        if isinstance(value, qwix.QArray):
            qvalue = value.qvalue
            dtype = qvalue.dtype
            if jnp.issubdtype(dtype, jnp.signedinteger):
                itemsize = 1
            elif jnp.issubdtype(dtype, jnp.unsignedinteger):
                itemsize = 1
            elif jnp.issubdtype(dtype, jnp.floating):
                itemsize = 2
            else:
                raise TypeError(
                    'Unsupported Qwix qvalue dtype for serialization: '
                    f'{dtype}'
                )
            total = qvalue.size * itemsize + value.scale.nbytes
            if value.zero_point is not None:
                total += value.zero_point.size * itemsize
            return int(total)
        return int(value.nbytes)

    def _stream_save_pretrained(
        self,
        path: str,
        *,
        max_shard_size: str='10GB',
        module_map: tp.Any=None,
    ) -> tuple[tp.Any, ...]:
        """Write a full checkpoint shard-by-shard without a full host copy.

        Device tensors are fetched one shard at a time, and stacked
        parameters transfer only the contiguous layer ranges each shard
        uses, so peak host memory stays at a couple of shards instead of
        the whole model; the fetch of one shard overlaps with the disk
        write of the previous one.
        """
        from taktiny.utils.logging import is_jax_rank_zero

        started = time.perf_counter()
        os.makedirs(path, exist_ok=True)
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'w') as config_file:
            json.dump(
                self._checkpoint_config(),
                config_file,
                indent=2,
                default=str,
            )

        if module_map is None:
            module_map = list(
                getattr(self, 'checkpoint_module_map', None) or []
            )

        flat = self.flat_state_dict()
        entries = self._expand_stacked_entries(flat)
        expanded = {
            name: (
                flat[source]
                if index is None
                else flat[source][index]
            )
            for name, source, index in entries
        }
        provenance = {
            name: (source, index) for name, source, index in entries
        }
        # Restore source-format names before planning shards so the written
        # index matches what `_save_pretrained_snapshot` produces. The
        # inversion preserves key order, so display and internal names stay
        # aligned positionally.
        internal_names = list(expanded.keys())
        renamed = self._invert_checkpoint_names(expanded, module_map)
        internal_of_display = dict(zip(renamed.keys(), internal_names))

        split = split_state_dict_into_shards_factory(
            renamed,
            get_storage_size=self._storage_nbytes,
            filename_pattern='model{suffix}.safetensors',
            max_shard_size=max_shard_size,
        )

        stem = 'model'
        extension = '.safetensors'
        shard_pattern = re.compile(
            rf'{re.escape(stem)}-\d{{5}}-of-\d{{5}}'
            rf'{re.escape(extension)}'
        )
        for existing_filename in os.listdir(path):
            if (
                existing_filename == f'{stem}{extension}'
                or shard_pattern.fullmatch(existing_filename)
                or existing_filename == f'{stem}{extension}.index.json'
                or existing_filename == 'quantization_config.json'
            ):
                os.remove(os.path.join(path, existing_filename))

        shard_items = list(split.filename_to_tensors.items())
        # Cache keys are (source, layer_index) pairs, or (source, None) for
        # parameters outside a SeqStack; eviction is keyed the same way so a
        # stacked tensor only stays resident while some layer of it is still
        # needed by a future shard.
        last_use = {}
        for shard_index, (_, tensor_names) in enumerate(shard_items):
            for name in tensor_names:
                key = provenance[internal_of_display[name]]
                last_use[key] = shard_index

        total_bytes = sum(self._storage_nbytes(v) for v in renamed.values())
        saved_paths = [config_path]
        quantization_parameters: dict = {}
        host_cache: dict = {}
        write_start = time.perf_counter()

        # Writes go straight to the Rust safetensors writer one shard at a
        # time. Optional JAX-native prefetch issues shard N+1's device
        # copies before shard N is written, overlapping DMA with disk I/O;
        # enable with TAKTINY_SAVE_PREFETCH=1 after verifying it wins on the
        # target accelerator, since CPU backends measure slower with it.
        use_prefetch = (
            len(shard_items) > 1
            and os.environ.get('TAKTINY_SAVE_PREFETCH', '0') == '1'
            and hasattr(jax.Array, 'copy_to_host_async')
        )
        if is_jax_rank_zero() and len(shard_items) > 1:
            print(
                f'[taktiny] planning save: {len(shard_items)} shards, '
                f'~{total_bytes / 1e9:.2f} GB '
                f'(prefetch {"on" if use_prefetch else "off"})'
            )

        def stabilize(values: tp.Any) -> tp.Any:
            def copy_leaf(value: tp.Any) -> tp.Any:
                # Per-layer slices of stacked tensors are contiguous views
                # that pin their base buffer, so safetensors can read them
                # in place; copying every view would duplicate the whole
                # shard on host. Only materialize non-contiguous arrays.
                if (
                    isinstance(value, np.ndarray)
                    and not value.flags['C_CONTIGUOUS']
                ):
                    return np.ascontiguousarray(value)
                return value

            return jax.tree.map(copy_leaf, values)

        def begin_shard(shard_index: int, tensor_names: list) -> dict:
            """Resolve exact device runs and optionally start their D2H."""
            requested_bytes = sum(
                self._storage_nbytes(expanded[internal_of_display[name]])
                for name in tensor_names
            )
            needs: dict = {}
            for name in tensor_names:
                key = provenance[internal_of_display[name]]
                needs.setdefault(key[0], set()).add(key[1])

            # Fetch only the layer ranges this shard actually uses: a
            # stacked tensor is transferred as contiguous [lo:hi] device
            # slices so shard size bounds host residency even when a stack
            # spans several shards.
            to_fetch: dict = {}
            fetched_bytes = 0
            for source, indices in needs.items():
                if None in indices:
                    if (source, None) not in host_cache:
                        to_fetch[(source, None)] = flat[source]
                        fetched_bytes += self._resident_nbytes(flat[source])
                    continue
                run_start = None
                previous = None
                for index in sorted(indices):
                    if (source, index) in host_cache:
                        continue
                    if previous is None or index != previous + 1:
                        if previous is not None:
                            to_fetch[(
                                source,
                                run_start,
                                previous,
                            )] = flat[source][run_start:previous + 1]
                            fetched_bytes += self._resident_nbytes(
                                flat[source][run_start:previous + 1]
                            )
                        run_start = index
                    previous = index
                if previous is not None:
                    to_fetch[(source, run_start, previous)] = flat[source][
                        run_start:previous + 1
                    ]
                    fetched_bytes += self._resident_nbytes(
                        flat[source][run_start:previous + 1]
                    )

            prefetched = False
            if use_prefetch and to_fetch:
                try:
                    for block in to_fetch.values():
                        block.copy_to_host_async()
                    prefetched = True
                except Exception:
                    prefetched = False
            return {
                'index': shard_index,
                'to_fetch': to_fetch,
                'requested_bytes': requested_bytes,
                'fetched_bytes': fetched_bytes,
                'prefetched': prefetched,
            }

        def materialize_shard(handle: dict) -> None:
            shard_index = handle['index']
            to_fetch = handle['to_fetch']
            handle['d2h_seconds'] = 0.0
            handle['sync_seconds'] = 0.0
            if to_fetch:
                blocks = list(to_fetch.values())
                # Charge pending device work to 'sync', not to the transfer,
                # so the D2H figure reflects transfer throughput alone.
                sync_start = time.perf_counter()
                jax.block_until_ready(blocks)
                handle['sync_seconds'] = time.perf_counter() - sync_start

                transfer_start = time.perf_counter()
                fetched = jax.device_get(to_fetch)
                handle['d2h_seconds'] = (
                    time.perf_counter() - transfer_start
                )
                for key, block in fetched.items():
                    if len(key) == 2:
                        host_cache[key] = block
                    else:
                        _, lo, hi = key
                        for offset in range(hi - lo + 1):
                            host_cache[(key[0], lo + offset)] = block[offset]

            if to_fetch and is_jax_rank_zero() and (
                len(shard_items) > 1 or handle['d2h_seconds'] >= 0.5
            ):
                run_sizes = sorted(
                    self._resident_nbytes(block) for block in to_fetch.values()
                )
                cache_bytes = sum(
                    self._resident_nbytes(value)
                    for value in host_cache.values()
                )
                print(
                    f'[taktiny] shard {shard_index + 1}/'
                    f'{len(shard_items)}: D2H {handle["d2h_seconds"]:.1f}s '
                    f'(+sync {handle["sync_seconds"]:.1f}s) | '
                    f'{len(run_sizes)} runs, largest '
                    f'{run_sizes[-1] / 1e6:.0f}MB, median '
                    f'{run_sizes[len(run_sizes) // 2] / 1e6:.0f}MB | '
                    f'fetched {handle["fetched_bytes"] / 1e6:.0f}MB of '
                    f'{handle["requested_bytes"] / 1e6:.0f}MB requested | '
                    f'host cache {cache_bytes / 1e6:.0f}MB'
                )

        def build_shard(shard_filename: str, tensor_names: list) -> dict:
            # Build under internal spellings; the single inversion below
            # restores source-format names exactly like the eager path.
            shard_state = {}
            for name in tensor_names:
                internal = internal_of_display[name]
                source, index = provenance[internal]
                shard_state[internal] = host_cache[(source, index)]
            stabilized = stabilize(shard_state)
            inverted = self._invert_checkpoint_names(stabilized, module_map)
            encoded, metadata = self._encode_qwix_state(inverted)
            if metadata is not None:
                quantization_parameters.update(metadata['parameters'])
            return encoded

        pending = None
        for shard_index, (shard_filename, tensor_names) in enumerate(
            shard_items,
        ):
            current = (
                pending
                if pending is not None
                else begin_shard(shard_index, tensor_names)
            )
            materialize_shard(current)
            encoded = build_shard(shard_filename, tensor_names)

            # Kick off shard N+1's DMA before handing shard N to the Rust
            # writer so the transfer overlaps serialization and disk I/O.
            if shard_index + 1 < len(shard_items):
                pending = begin_shard(
                    shard_index + 1,
                    shard_items[shard_index + 1][1],
                )
            else:
                pending = None

            shard_path = os.path.join(path, shard_filename)
            write_shard_start = time.perf_counter()
            save_file(encoded, shard_path)
            del encoded
            if is_jax_rank_zero() and len(shard_items) > 1:
                print(
                    f'[taktiny] wrote {shard_filename} in '
                    f'{time.perf_counter() - write_shard_start:.1f}s'
                )
            saved_paths.append(shard_path)

            # Release host cache entries no future shard references. The
            # write above consumed this shard's views, and views held by an
            # in-flight prefetch pin their own base buffers, so dropping
            # stale entries here bounds residency at roughly two shards.
            host_cache = {
                key: value
                for key, value in host_cache.items()
                if last_use.get(key, -1) > shard_index
            }

        if quantization_parameters:
            with open(
                os.path.join(path, 'quantization_config.json'),
                'w',
            ) as quantization_file:
                json.dump(
                    {
                        'format': 'taktiny-qwix',
                        'version': 1,
                        'parameters': quantization_parameters,
                    },
                    quantization_file,
                    indent=2,
                )
            saved_paths.insert(1, os.path.join(path, 'quantization_config.json'))

        if split.is_sharded:
            index_path = os.path.join(
                path,
                f'{stem}{extension}.index.json',
            )
            with open(index_path, 'w') as index_file:
                json.dump(
                    {
                        'metadata': split.metadata,
                        'weight_map': split.tensor_to_filename,
                    },
                    index_file,
                    indent=2,
                )
            saved_paths.append(index_path)

        if is_jax_rank_zero():
            print(
                f'[taktiny] checkpoint written to {path}: '
                f'{total_bytes / 1e9:.2f} GB in '
                f'{time.perf_counter() - write_start:.1f}s '
                f'(total {time.perf_counter() - started:.1f}s)'
            )
        return tuple(saved_paths)

    def save_pretrained(
        self,
        path: str,
        max_shard_size: str='10GB',
        module_map: tp.Any=None,
    ) -> tp.Any:
        """Save a full model checkpoint or the model's LoRA adapters.

        Models containing ``LoRALinear`` modules save only adapter tensors and
        their reconstruction metadata. Models without LoRA save their complete
        parameter state and a Safetensors index. Parameters held by a
        ``SeqStack`` are expanded into conventional numbered layer keys.

        Tensor names are restored to their source checkpoint spelling by
        reversing the ``module_map`` rules that were applied while loading;
        plain rename rules map back to their original names, while rules with
        transforms cannot be inverted and keep the loaded names.

        Args:
            path: Directory in which to write the checkpoint.
            max_shard_size: Maximum tensor data size per Safetensors file,
                expressed as an integer byte count or a string using ``KB``,
                ``MB``, ``GB``, or ``TB``, such as ``"5GB"``. A tensor larger
                than the limit is saved alone without being split.
            module_map: Mapping rules used to restore source tensor names.
                Defaults to the rules remembered from ``from_pretrained``.

        Returns:
            A tuple containing the paths written by this invocation, with
            configuration files first, followed by weight files and their
            index when present.
        """
        if self._lora_state_dict():
            # Adapters are tiny; the eager snapshot path is fine for them.
            return self._save_pretrained_snapshot(
                self._checkpoint_snapshot(),
                path,
                max_shard_size=max_shard_size,
                module_map=module_map,
            )
        return self._stream_save_pretrained(
            path,
            max_shard_size=max_shard_size,
            module_map=module_map,
        )

    def load_pretrained(self, path: str) -> tp.Any:
        """Load a Taktiny-native full checkpoint into this model in place.

        This is the inverse of ``save_pretrained`` for full-model checkpoints.
        Numbered checkpoint layers are reconstructed into ``SeqStack``
        parameters without applying external checkpoint name mappings or
        matrix transpositions.

        Tensors are streamed shard-by-shard: each tensor is routed and
        placed (or assembled into its SeqStack buffer) as soon as it is
        read, so peak host memory is bounded by the largest single
        parameter plus one in-flight stack instead of the whole checkpoint.

        Args:
            path: Local directory containing model Safetensors.

        Returns:
            This model instance.
        """
        path = os.fspath(path)
        quantization_path = os.path.join(
            path,
            'quantization_config.json',
        )
        quantization_metadata = None
        if os.path.isfile(quantization_path):
            with open(quantization_path) as quantization_file:
                quantization_metadata = json.load(quantization_file)
        index_path = os.path.join(
            path,
            'model.safetensors.index.json',
        )
        if os.path.isfile(index_path):
            with open(index_path) as index_file:
                index = json.load(index_file)
            weight_map = index.get('weight_map')
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(
                    'Model Safetensors index has no weight_map'
                )
            filenames = dict.fromkeys(weight_map.values())
        else:
            filenames = {'model.safetensors': None}

        parameters = self.flat_parameter_dict()
        grouped_layouts = _grouped_stack_layout(parameters)

        quantization_specs = {}
        if quantization_metadata is not None:
            if (
                quantization_metadata.get('format') != 'taktiny-qwix'
                or quantization_metadata.get('version') != 1
            ):
                raise ValueError(
                    'Unsupported Qwix checkpoint metadata format'
                )
            specs = quantization_metadata.get('parameters')
            if not isinstance(specs, dict):
                raise ValueError(
                    'Qwix checkpoint metadata has no parameter mapping'
                )
            quantization_specs = specs

        seen_names: set = set()
        unexpected = []
        loaded_names: set = set()
        stacked_entries: dict = {}
        component_buffers: dict = {}

        def decode_qarray(base: str, parts: dict) -> tp.Any:
            specification = quantization_specs[base]
            qvalue_dtype = jnp.dtype(specification['qvalue_dtype'])
            qvalue = parts['qvalue'].astype(qvalue_dtype)
            zero_point = None
            if 'zero_point' in parts:
                zero_point = parts['zero_point'].astype(qvalue_dtype)
            return qwix.QArray(
                qvalue=qvalue,
                scale=parts['scale'],
                zero_point=zero_point,
                qtype=specification.get('qtype'),
            )

        def place_value(name: str, value: tp.Any) -> None:
            parameter = parameters[name]
            if (
                isinstance(parameter.value, qwix.QArray)
                and not isinstance(value, qwix.QArray)
            ):
                raise TypeError(
                    'Loading a dense native checkpoint into an existing '
                    f'quantized parameter is unsupported: {name}'
                )
            if isinstance(value, qwix.QArray):
                target = parameter.value

                def place(
                    component: tp.Any,
                    target_component: tp.Any=None,
                ) -> tp.Any:
                    component = jnp.asarray(component)
                    sharding = getattr(target_component, 'sharding', None)
                    if sharding is not None:
                        component = jax.device_put(component, sharding)
                    return component

                if isinstance(target, qwix.QArray):
                    parameter.value = qwix.QArray(
                        qvalue=place(value.qvalue, target.qvalue),
                        scale=place(value.scale, target.scale),
                        zero_point=(
                            place(value.zero_point, target.zero_point)
                            if value.zero_point is not None
                            else None
                        ),
                        qtype=value.qtype,
                    )
                else:
                    parameter.value = jax.tree.map(place, value)
                loaded_names.add(name)
                return

            array = jnp.asarray(value, dtype=parameter.dtype)
            sharding = getattr(parameter.value, 'sharding', None)
            if sharding is not None:
                array = jax.device_put(array, sharding)
            parameter.value = array
            loaded_names.add(name)

        def finalize_stacked(stacked_name: str) -> None:
            entry = stacked_entries[stacked_name]
            parameter = parameters[stacked_name]
            expected_indices = set(range(parameter.shape[0]))
            if entry['indices'] != expected_indices:
                return
            ordered = [
                entry['values'][index]
                for index in range(parameter.shape[0])
            ]
            if isinstance(ordered[0], qwix.QArray):
                value = jax.tree.map(
                    lambda *values: jnp.stack(values),
                    *ordered,
                )
            else:
                value = np.stack(ordered)
            del entry['values']
            place_value(stacked_name, value)
            del stacked_entries[stacked_name]

        def handle_assembled(name: str, value: tp.Any) -> None:
            if name in seen_names:
                raise ValueError(
                    f'Duplicate model tensor in checkpoint: {name}'
                )
            seen_names.add(name)

            if name in parameters:
                parameter = parameters[name]
                if value.shape != parameter.shape:
                    raise ValueError(
                        f'Model tensor {name!r} has shape '
                        f'{value.shape}, expected {parameter.shape}'
                    )
                place_value(name, value)
                return

            resolved = _resolve_stacked_parameter(
                name,
                parameters,
                grouped_layouts,
            )
            if resolved is None:
                unexpected.append(name)
                return
            stacked_name, layer_index = resolved
            parameter = parameters[stacked_name]
            expected_shape = parameter.shape[1:]
            if value.shape != expected_shape:
                raise ValueError(
                    f'Model tensor {name!r} has shape '
                    f'{value.shape}, expected {expected_shape}'
                )
            entry = stacked_entries.setdefault(
                stacked_name,
                {'values': {}, 'indices': set()},
            )
            if layer_index in entry['indices']:
                raise ValueError(
                    f'Duplicate model layer tensor: {name}'
                )
            entry['values'][layer_index] = value
            entry['indices'].add(layer_index)
            finalize_stacked(stacked_name)

        def handle_tensor(name: str, value: tp.Any) -> None:
            base, separator, component = name.rpartition('.__qwix__.')
            if not separator or component not in (
                'qvalue',
                'scale',
                'zero_point',
            ) or base not in quantization_specs:
                handle_assembled(name, value)
                return

            specification = quantization_specs[base]
            required = ['qvalue', 'scale']
            if specification.get('zero_point') is not None:
                required.append('zero_point')
            parts = component_buffers.setdefault(base, {})
            if component in parts:
                raise ValueError(
                    f'Duplicate model tensor in checkpoint: {name}'
                )
            parts[component] = value
            if all(part in parts for part in required):
                del component_buffers[base]
                try:
                    decoded = decode_qarray(base, parts)
                except KeyError as error:
                    raise ValueError(
                        f'Qwix checkpoint metadata is missing dtype for '
                        f'{base!r}'
                    ) from error
                handle_assembled(base, decoded)

        for filename in filenames:
            checkpoint_path = os.path.join(path, filename)
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f'Model checkpoint file was not found: {checkpoint_path}'
                )
            with safe_open(
                checkpoint_path,
                framework='np',
                device='cpu',
            ) as checkpoint:
                for name in checkpoint.keys():
                    handle_tensor(name, checkpoint.get_tensor(name))

        if component_buffers:
            missing_group = sorted(component_buffers)[0]
            raise ValueError(
                f'Qwix parameter {missing_group!r} is missing components '
                'in the checkpoint'
            )

        if unexpected:
            preview = ', '.join(sorted(unexpected)[:8])
            raise ValueError(
                f'Model checkpoint contains unexpected tensors: {preview}'
            )

        for stacked_name, entry in stacked_entries.items():
            parameter = parameters[stacked_name]
            missing_indices = set(range(parameter.shape[0])) - entry['indices']
            if missing_indices:
                missing = ', '.join(map(str, sorted(missing_indices)))
                raise ValueError(
                    f'Model checkpoint is missing layers {missing} '
                    f'for {stacked_name!r}'
                )

        missing = sorted(set(parameters) - loaded_names)
        if missing:
            preview = ', '.join(missing[:8])
            raise ValueError(
                f'Model checkpoint is missing tensors: {preview}'
            )

        return self

    def placement_report(self) -> str:
        """Summarize parameter memory grouped by device placement.

        Useful for diagnosing out-of-memory failures on multi-device
        meshes. The per-device section reports the bytes each device
        actually holds (sharded arrays contribute their slice, replicated
        arrays their full copy), so unbalanced placement is visible after
        loading.

        Returns:
            A human-readable report with a per-device breakdown, logical
            bytes grouped by placement kind, and a logical total.
        """

        per_device: tp.Any = collections.Counter()
        by_kind: tp.Any = collections.Counter()
        for path, leaf in jax.tree_util.tree_flatten_with_path(
            self.flat_state_dict()
        )[0]:
            devices = getattr(leaf, 'devices', None)
            if not callable(devices):
                continue
            resolved = tuple(sorted(str(device) for device in devices()))
            if not resolved:
                continue

            shards = getattr(leaf, 'addressable_shards', None)
            if shards:
                for shard in shards:
                    per_device[str(shard.device)] += int(
                        getattr(shard.data, 'nbytes', 0)
                    )
            else:
                share = int(getattr(leaf, 'nbytes', 0)) // len(resolved)
                for device in resolved:
                    per_device[device] += share

            spec = getattr(getattr(leaf, 'sharding', None), 'spec', None)
            replicated = (
                spec is not None and all(part is None for part in spec)
            )
            nbytes = int(getattr(leaf, 'nbytes', 0))
            if len(resolved) == 1:
                kind = f'single {resolved[0]}'
            elif replicated:
                kind = 'replicated on: ' + ', '.join(resolved)
            else:
                kind = 'sharded across: ' + ', '.join(resolved)
            by_kind[kind] += nbytes

        total = sum(by_kind.values())
        lines = ['per device:']
        lines.extend(
            f'  {device}: {size / 2 ** 30:.2f} GiB'
            for device, size in sorted(
                per_device.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        lines.append('by placement (logical bytes):')
        lines.extend(
            f'  {kind}: {size / 2 ** 30:.2f} GiB'
            for kind, size in sorted(
                by_kind.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        lines.append(f'total: {total / 2 ** 30:.2f} GiB')
        return '\n'.join(lines)

    def push_to_hub(
        self,
        repo_id: str,
        *,
        commit_message: tp.Any=None,
        commit_description: tp.Any=None,
        private: tp.Any=None,
        token: tp.Any=None,
        revision: tp.Any=None,
        create_pr: bool=False,
        max_shard_size: str='5GB',
        module_map: tp.Any=None,
    ) -> str:
        """Save and upload this model or adapter to the Hugging Face Hub.

        The checkpoint is staged in a temporary directory and removed after
        the upload completes. Existing unrelated repository files are
        preserved, while obsolete shards belonging to the uploaded checkpoint
        family are deleted in the same commit.

        Args:
            repo_id: Hub repository identifier, optionally including an
                organization or username.
            commit_message: Optional Hub commit title.
            commit_description: Optional longer commit description.
            private: Visibility used when creating a new repository.
            token: Hugging Face authentication token or token-selection flag.
            revision: Branch or revision to receive the commit.
            create_pr: Whether to create a pull request instead of committing
                directly to the target revision.
            max_shard_size: Maximum size passed to ``save_pretrained``.
            module_map: Mapping rules used to restore source tensor names;
                defaults to the rules remembered from ``from_pretrained``.

        Returns:
            The URL of the created Hub commit or pull request.
        """
        api = HfApi(
            token=token,
            library_name='taktiny',
        )
        repo = api.create_repo(
            repo_id=repo_id,
            private=private,
            token=token,
            repo_type='model',
            exist_ok=True,
        )
        resolved_repo_id = getattr(repo, 'repo_id', repo_id)

        if revision is not None and not revision.startswith('refs/pr'):
            api.create_branch(
                repo_id=resolved_repo_id,
                branch=revision,
                token=token,
                exist_ok=True,
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            saved_paths = self.save_pretrained(
                temporary_directory,
                max_shard_size=max_shard_size,
                module_map=module_map,
            )
            filenames = {
                os.path.basename(path)
                for path in saved_paths
            }
            is_adapter = any(
                filename.startswith('adapter_model')
                for filename in filenames
            )
            stem = 'adapter_model' if is_adapter else 'model'
            delete_patterns = [
                f'{stem}.safetensors',
                f'{stem}-*-of-*.safetensors',
                f'{stem}.safetensors.index.json',
            ]
            if not is_adapter:
                delete_patterns.append('quantization_config.json')

            commit = api.upload_folder(
                repo_id=resolved_repo_id,
                folder_path=temporary_directory,
                commit_message=commit_message or 'Upload model',
                commit_description=commit_description,
                token=token,
                repo_type='model',
                revision=revision,
                create_pr=create_pr,
                delete_patterns=delete_patterns,
            )

        commit_url = getattr(commit, 'commit_url', commit)
        return str(commit_url)

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: PathLike,
        config: tp.Any,
        *,
        module_map: tp.List | None = None,
        local: bool = False,
        dtype: DType | None = None,
        quant: tp.Any = None,
        subfolder: PathLike | str | None = None,
        weights_filename: str = 'model.safetensors',
        mesh: jax.sharding.Mesh | None = None,
        sharding_rules: LogicalRules | None = None,
        allow_unmatched: bool = False,
        show_progress: bool = True,
        load_chunk_size: int | str | None = '1GB',
        **kwargs,
    ) -> tp.Any:
        """
        Loads Safetensors weights into a newly instantiated model. Supports
        both single-file and sharded checkpoints, including architecture-
        specific filenames selected through ``weights_filename``.

        Args:
            load_chunk_size: Peak host memory budget used while streaming
                checkpoint tensors, expressed as an integer byte count or a
                string such as ``"256MB"``. Tensors are decoded, name-mapped,
                and transferred in batches up to this size instead of one at
                a time, trading additional host memory for faster
                materialization. Defaults to ``"1GB"``; pass ``None`` or
                ``0`` to load one tensor at a time instead.
        """
        # Keep a pristine copy of the caller's configuration so that
        # ``save_pretrained`` can write back what the model was loaded with
        # instead of this session's library overrides.
        original_config_dict = copy.deepcopy(
            cls._config_as_dict(config)
        )

        def set_config_override(name: str, value: tp.Any) -> None:
            setattr(config, name, value)
            text_config = vars(config).get('text_config')
            if text_config is not None:
                setattr(text_config, name, value)

        uniform_quant = None
        plain_dtype_override = None
        if dtype is not None:
            dtype_name = dtype.lower() if isinstance(dtype, str) else None
            quantized_dtypes = {'fp8', 'int8', 'int4', 'nf4'}
            if dtype_name in quantized_dtypes:
                compute_dtype = (
                    getattr(config, 'torch_dtype', None)
                    or getattr(config, 'dtype', None)
                )
                if (
                    compute_dtype is None
                    or (
                        isinstance(compute_dtype, str)
                        and compute_dtype.lower() in quantized_dtypes
                    )
                ):
                    compute_dtype = 'bfloat16'

                uniform_quant = dtype_name
                set_config_override('dtype', compute_dtype)
                set_config_override('torch_dtype', compute_dtype)
            else:
                set_config_override('dtype', dtype)
                set_config_override('torch_dtype', dtype)
                plain_dtype_override = (
                    dtype
                    if isinstance(dtype, str)
                    else getattr(dtype, 'name', str(dtype))
                )
        if quant is not None and uniform_quant is not None:
            set_config_override(
                'quant',
                merge_quantization(quant, uniform_quant),
            )
        elif quant is not None:
            set_config_override('quant', quant)
        elif uniform_quant is not None:
            set_config_override('quant', uniform_quant)

        if not isinstance(weights_filename, str) or not weights_filename:
            raise ValueError('weights_filename must be a non-empty string')
        if not weights_filename.endswith('.safetensors'):
            raise ValueError('weights_filename must end with .safetensors')
        index_filename = f'{weights_filename}.index.json'
        load_chunk_bytes = (
            parse_size(load_chunk_size)
            if load_chunk_size is not None
            else 0
        )

        if sharding_rules is None:
            sharding_rules = cls._resolve_sharding_rules()

        path_or_repo_str = str(path_or_repo)
        module_map = module_map or []
        native_qwix_directory = None
        if local:
            candidate = os.path.join(
                path_or_repo_str,
                subfolder if subfolder else '',
                'quantization_config.json',
            )
            if os.path.isfile(candidate):
                native_qwix_directory = os.path.dirname(candidate)

        # 1. Determine if model is sharded or single file
        is_sharded = False
        if local:
            index_path = os.path.join(
                path_or_repo_str,
                subfolder if subfolder else '',
                index_filename,
            )
            if os.path.exists(index_path):
                is_sharded = True
        else:
            try:
                info = repo_info(repo_id=path_or_repo_str)
                files = [f.rfilename for f in info.siblings]
                target_index = (
                    f'{subfolder}/{index_filename}'
                    if subfolder
                    else index_filename
                )
                if target_index in files:
                    is_sharded = True
                    index_path = hf_hub_download(
                        repo_id=path_or_repo_str,
                        subfolder=subfolder,
                        filename=index_filename,
                    )
                target_quantization = (
                    f'{subfolder}/quantization_config.json'
                    if subfolder
                    else 'quantization_config.json'
                )
                if target_quantization in files:
                    quantization_path = hf_hub_download(
                        repo_id=path_or_repo_str,
                        subfolder=subfolder,
                        filename='quantization_config.json',
                    )
                    native_qwix_directory = os.path.dirname(
                        quantization_path
                    )
            except Exception as e:
                print(f"Failed to fetch repo info: {e}")
                is_sharded = False

        # 2. Build files_to_load mapping: file_name -> list of keys (or None for all)
        files_to_load = {}
        if is_sharded:
            with open(index_path, "r") as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            for k_str, file_name in weight_map.items():
                if file_name not in files_to_load:
                    files_to_load[file_name] = []
                files_to_load[file_name].append(k_str)
        else:
            files_to_load[weights_filename] = None

        # 3. Resolve every checkpoint file before materializing any parameters.
        resolved_files = {}
        for file_name in files_to_load:
            if local:
                resolved_files[file_name] = os.path.join(
                    path_or_repo_str,
                    subfolder if subfolder else "",
                    file_name,
                )
            else:
                resolved_files[file_name] = hf_hub_download(
                    repo_id=path_or_repo_str,
                    subfolder=subfolder,
                    filename=file_name,
                )

        # 4. Instantiate model skeleton using eval_shape (no memory allocation)
        rngs = kwargs.pop('rngs', nn.Rngs(0))
        state = jax.eval_shape(
            lambda: cls(
                config,
                rngs=rngs,
                mesh=mesh,
                sharding_rules=sharding_rules,
                **kwargs,
            )
        )
        state.original_config_dict = original_config_dict
        state.loaded_dtype_override = plain_dtype_override
        # Remember which module_map rules fire during load so that saving can
        # restore only the checkpoint spellings that were actually mapped.
        state.checkpoint_module_map = []
        checkpoint_keys_seen = set()
        if native_qwix_directory is not None:
            state.load_pretrained(native_qwix_directory)
            state.base_model_name_or_path = path_or_repo_str
            return state

        current_state_dict = state.flat_parameter_dict()
        grouped_layouts = _grouped_stack_layout(current_state_dict)
        new_state = {}
        not_found_some = False

        # 5. Load weights
        cpu_device = jax.devices('cpu')[0]
        default_device = jax.devices()[0]
        quantizers: dict[tuple[tp.Any, ...], tp.Callable[[tp.Any], tp.Any]] = {}

        def quantize_weight(
            value: tp.Any,
            parameter: tp.Any,
            rule: tp.Any,
            quantization_kind: str,
        ) -> tp.Any:
            batch_axis_count = getattr(
                parameter,
                'quantization_batch_axis_count',
                0,
            )
            input_axis_count = getattr(
                parameter,
                'input_axis_count',
                None,
            )
            parameter_dtype = jnp.dtype(parameter.dtype)
            cache_key = (
                quantization_kind,
                batch_axis_count,
                input_axis_count,
                str(parameter_dtype),
                str(rule.weight_qtype),
                rule.tile_size,
                rule.weight_calibration_method,
            )
            quantizer = quantizers.get(cache_key)
            if quantizer is None:
                metadata = SimpleNamespace(
                    dtype=parameter_dtype,
                    input_axis_count=input_axis_count,
                    quantization_batch_axis_count=batch_axis_count,
                )
                if quantization_kind == 'embedding':
                    def apply(array: tp.Any) -> tp.Any:
                        return quantize_embedding_weight(
                            array,
                            metadata,
                            rule,
                        )
                else:
                    def apply(array: tp.Any) -> tp.Any:
                        return quantize_linear_weight(
                            array,
                            metadata,
                            rule,
                        )
                quantizer = jax.jit(apply)
                quantizers[cache_key] = quantizer

            # Keep PTQ on the host so loading never needs a dense copy of the
            # source weight in accelerator memory. Reusing the jitted callable
            # fuses Qwix calibration and quantization for repeated layer shapes.
            cpu_value = jax.device_put(value, cpu_device)
            return quantizer(cpu_value)

        unsharded_warning_shown = False

        def parameter_sharding(
            parameter: tp.Any,
            axis_names: AxisNames | None=None,
            *,
            use_explicit: bool=True,
        ) -> tp.Any:
            nonlocal unsharded_warning_shown
            sharding = (
                getattr(parameter, 'sharding', None)
                if use_explicit
                else None
            )
            if (
                sharding is None
                and axis_names is not None
                and mesh is not None
            ):
                sharding = create_sharding(
                    mesh,
                    axis_names,
                    rules=sharding_rules,
                )
                spec = getattr(sharding, 'spec', None)
                if (
                    mesh.size > 1
                    and spec is not None
                    and all(part is None for part in spec)
                    and not unsharded_warning_shown
                ):
                    # Every axis resolved to replicated: the tensor will be
                    # copied whole onto every device in the mesh.
                    unsharded_warning_shown = True
                    print(
                        'Warning: parameters resolve no sharded axes under '
                        'the current mesh and will be fully replicated on '
                        'every device; check that sharding_rules cover '
                        'their logical axis names.'
                    )
            if (
                sharding is None
                and mesh is not None
                and not unsharded_warning_shown
            ):
                # Without a resolved sharding the tensor is placed whole on
                # the first device, which can silently exhaust one GPU while
                # the rest of the mesh stays idle.
                unsharded_warning_shown = True
                print(
                    'Warning: some parameters resolve no sharding under the '
                    'current mesh and will be placed whole on a single '
                    'device; give them axis_names or matching '
                    'sharding_rules to shard them across the mesh.'
                )
            if sharding is None and mesh is None:
                sharding = default_device
            return sharding

        def place_qarray(value: tp.Any, parameter: tp.Any) -> tp.Any:
            axis_names = getattr(parameter, 'axis_names', None)
            qvalue_sharding = parameter_sharding(parameter, axis_names)

            scale_axis_names = None
            if axis_names is not None:
                scale_axis_names = tuple(
                    axis_name if size != 1 else None
                    for axis_name, size in zip(
                        axis_names,
                        value.scale.shape,
                    )
                )
            scale_sharding = parameter_sharding(
                parameter,
                scale_axis_names,
                use_explicit=False,
            )

            zero_point = value.zero_point
            if zero_point is not None:
                zero_axis_names = None
                if axis_names is not None:
                    zero_axis_names = tuple(
                        axis_name if size != 1 else None
                        for axis_name, size in zip(
                            axis_names,
                            zero_point.shape,
                        )
                    )
                zero_point = jax.device_put(
                    zero_point,
                    parameter_sharding(
                        parameter,
                        zero_axis_names,
                        use_explicit=False,
                    ),
                )

            return value.replace(
                qvalue=jax.device_put(value.qvalue, qvalue_sharding),
                scale=jax.device_put(value.scale, scale_sharding),
                zero_point=zero_point,
            )

        def parameter_quantization_rule(key: tp.Any, parameter: tp.Any) -> tuple[tp.Any, ...]:
            quantization = getattr(parameter, 'quantization', None)
            quantization_kind = getattr(
                parameter,
                'quantization_kind',
                'dot_general',
            )
            rule = resolve_quantization_rule(
                quantization,
                key.rsplit('.', 1)[0],
                op_name=quantization_kind,
            )
            return rule, quantization_kind

        def stage_parameter(key: tp.Any, value: tp.Any, parameter: tp.Any) -> tp.Any:
            rule, quantization_kind = parameter_quantization_rule(
                key,
                parameter,
            )
            if rule is not None:
                parameter.trainable = False
                quantized = quantize_weight(
                    value,
                    parameter,
                    rule,
                    quantization_kind,
                )
                return place_qarray(quantized, parameter)

            target_dtype = np.dtype(parameter.dtype)
            if value.dtype != target_dtype:
                value = value.astype(target_dtype, copy=False)
            # Defer the device transfer so dense parameters decoded within
            # one chunk are placed by a single pipelined device_put call.
            pending_keys.append(key)
            pending_values.append(value)
            pending_shardings.append(
                parameter_sharding(
                    parameter,
                    getattr(parameter, 'axis_names', None),
                ),
            )
            return None

        pending_keys: list[tp.Any] = []
        pending_values: list[tp.Any] = []
        pending_shardings: list[tp.Any] = []

        def flush_pending_puts() -> None:
            if not pending_values:
                return
            placed = jax.device_put(pending_values, pending_shardings)
            for key, array in zip(pending_keys, placed):
                new_state[key] = array
            pending_keys.clear()
            pending_values.clear()
            pending_shardings.clear()

        def initialize_stacked_parameter(parameter: tp.Any) -> tp.Any:
            sharding = parameter_sharding(
                parameter,
                getattr(parameter, 'axis_names', None),
            )
            shape = tuple(parameter.shape)
            dtype = jnp.dtype(parameter.dtype)
            if isinstance(sharding, jax.sharding.Sharding):
                return jax.jit(
                    lambda: jnp.zeros(shape, dtype=dtype),
                    out_shardings=sharding,
                )()
            return jax.device_put(jnp.zeros(shape, dtype=dtype), sharding)

        def update_stacked_parameter(stacked: tp.Any, layer: tp.Any, layer_index: int) -> tp.Any:
            return jax.lax.dynamic_update_index_in_dim(
                stacked,
                layer,
                layer_index,
                axis=0,
            )

        update_stacked_parameter = jax.jit(
            update_stacked_parameter,
            donate_argnums=(0,),
        )

        stacked_states = {}

        def finalize_stacked_state(k_stacked: str) -> bool:
            stacked_state = stacked_states[k_stacked]
            target_var = current_state_dict[k_stacked]
            expected_indices = set(range(target_var.shape[0]))
            if stacked_state['indices'] != expected_indices:
                return False

            if stacked_state['kind'] == 'dense':
                axis_names = getattr(target_var, 'axis_names', None)
                value = jax.device_put(
                    stacked_state['value'],
                    parameter_sharding(target_var, axis_names),
                )
            else:
                value = place_qarray(
                    stacked_state['value'],
                    target_var,
                )

            # Complete the transfer before releasing the host assembly buffer.
            new_state[k_stacked] = jax.block_until_ready(value)
            del stacked_states[k_stacked]
            return True

        grouped_mapping = any(
            len(rule) == 3
            and isinstance(rule[0], (list, tuple))
            and len(rule[0]) > 1
            for rule in module_map
        )

        total_tensors = 0
        for file_name, keys_in_file in files_to_load.items():
            if keys_in_file is not None:
                total_tensors += len(keys_in_file)
                continue
            with safe_open(
                resolved_files[file_name],
                framework='np',
                device='cpu',
            ) as checkpoint:
                total_tensors += len(checkpoint.keys())

        progress = tqdm(
            total=total_tensors,
            desc='Loading checkpoint',
            unit='tensor',
            dynamic_ncols=True,
            disable=not show_progress or not is_jax_rank_zero(),
        )

        for file_name, keys_in_file in files_to_load.items():
            shard_path = resolved_files[file_name]
            with safe_open(shard_path, framework="np", device="cpu") as f:
                keys_to_process = keys_in_file if keys_in_file is not None else f.keys()
                checkpoint_keys_seen.update(keys_to_process)

                # Multi-source mapping rules (N-to-1) require their sibling
                # tensors to be name-mapped together, so each sibling set
                # forms an indivisible unit when the shard stream is split
                # into chunks.
                sibling_units: dict[str, tuple] = {}
                if grouped_mapping:
                    key_set = set(keys_to_process)
                    groups: list[set] = []
                    for rule in module_map:
                        if not (
                            len(rule) == 3
                            and isinstance(rule[0], (list, tuple))
                            and len(rule[0]) > 1
                        ):
                            continue
                        source_patterns = rule[0]
                        primary_source = source_patterns[0]
                        for key in keys_to_process:
                            if primary_source not in key:
                                continue
                            siblings = {
                                sibling
                                for sibling in (
                                    key.replace(primary_source, pattern)
                                    for pattern in source_patterns
                                )
                                if sibling in key_set
                            }
                            if len(siblings) <= 1:
                                continue
                            overlapping = [
                                group
                                for group in groups
                                if group & siblings
                            ]
                            for group in overlapping:
                                groups.remove(group)
                                siblings |= group
                            groups.append(siblings)
                    for group in groups:
                        unit = tuple(sorted(group))
                        for member in unit:
                            sibling_units[member] = unit

                def shard_chunks() -> tp.Iterator[dict]:
                    visited = set()
                    pending = {}
                    pending_bytes = 0
                    for key in keys_to_process:
                        if key in visited:
                            continue
                        unit = sibling_units.get(key) or (key,)
                        visited.update(unit)
                        for member in unit:
                            pending[member] = f.get_tensor(member)
                            pending_bytes += pending[member].nbytes
                            progress.update(1)
                        # A zero budget emits every unit immediately,
                        # preserving one-tensor-at-a-time loading.
                        if (
                            not load_chunk_bytes
                            or pending_bytes >= load_chunk_bytes
                        ):
                            yield pending
                            pending = {}
                            pending_bytes = 0
                    if pending:
                        yield pending

                def mapped_checkpoint_items() -> tp.Iterator[tp.Any]:
                    for chunk in shard_chunks():
                        yield from map_state_dict(chunk, module_map).items()
                        # Dense parameters staged during this chunk are now
                        # placed together through one pipelined device_put.
                        flush_pending_puts()

                mapped_items = mapped_checkpoint_items()

                for k_mapped, value in mapped_items:
                    if k_mapped in current_state_dict:
                        target_var = current_state_dict[k_mapped]

                        if value.ndim == 2:
                            if k_mapped.endswith(".weight") or ".lora_" in k_mapped:
                                value = value.T
                        elif (
                            value.ndim >= 3
                            and k_mapped.endswith('.weight')
                            and value.shape != target_var.shape
                        ):
                            convolution_shape = (
                                *value.shape[2:],
                                value.shape[1],
                                value.shape[0],
                            )
                            if convolution_shape == target_var.shape:
                                value = value.transpose(
                                    *range(2, value.ndim),
                                    1,
                                    0,
                                )
                        if value.shape != target_var.shape:
                            try:
                                value = value.reshape(target_var.shape)
                            except ValueError as error:
                                raise ValueError(
                                    f'Cannot load checkpoint tensor for '
                                    f'{k_mapped!r}: shape {value.shape} is '
                                    f'incompatible with parameter shape '
                                    f'{target_var.shape}'
                                ) from error
                        placed = stage_parameter(
                            k_mapped,
                            value,
                            target_var,
                        )
                        if placed is not None:
                            new_state[k_mapped] = placed

                    else:
                        # Check if it belongs to a SeqStack
                        resolved = _resolve_stacked_parameter(
                            k_mapped,
                            current_state_dict,
                            grouped_layouts,
                        )
                        if resolved is not None:
                            k_stacked, idx = resolved
                            if k_stacked in current_state_dict:
                                target_var = current_state_dict[k_stacked]

                                layer_shape = target_var.shape[1:]
                                if value.ndim == 2:
                                    if k_mapped.endswith(".weight") or ".lora_" in k_mapped:
                                        value = value.T
                                elif (
                                    value.ndim >= 3
                                    and k_mapped.endswith('.weight')
                                    and value.shape != layer_shape
                                ):
                                    convolution_shape = (
                                        *value.shape[2:],
                                        value.shape[1],
                                        value.shape[0],
                                    )
                                    if convolution_shape == layer_shape:
                                        value = value.transpose(
                                            *range(2, value.ndim),
                                            1,
                                            0,
                                        )
                                if value.shape != layer_shape:
                                    try:
                                        value = value.reshape(layer_shape)
                                    except ValueError as error:
                                        raise ValueError(
                                            f'Cannot load checkpoint tensor '
                                            f'for {k_mapped!r}: shape '
                                            f'{value.shape} is incompatible '
                                            f'with stacked layer shape '
                                            f'{layer_shape}'
                                        ) from error

                                stacked_state = stacked_states.get(k_stacked)
                                if stacked_state is None:
                                    if k_stacked in new_state:
                                        raise ValueError(
                                            'Checkpoint contains duplicate '
                                            f'values for {k_stacked!r}'
                                        )
                                    rule, quantization_kind = (
                                        parameter_quantization_rule(
                                            k_stacked,
                                            target_var,
                                        )
                                    )
                                    if rule is None:
                                        target_dtype = np.dtype(target_var.dtype)
                                        stacked_state = {
                                            'kind': 'dense',
                                            'value': np.zeros(
                                                target_var.shape,
                                                dtype=target_dtype,
                                            ),
                                            'indices': set(),
                                        }
                                    else:
                                        target_var.trainable = False
                                        stacked_state = {
                                            'kind': 'quantized',
                                            'value': None,
                                            'indices': set(),
                                            'rule': rule,
                                            'quantization_kind': (
                                                quantization_kind
                                            ),
                                        }
                                    stacked_states[k_stacked] = stacked_state

                                loaded_indices = stacked_state['indices']
                                if idx in loaded_indices:
                                    raise ValueError(
                                        'Checkpoint contains duplicate layer '
                                        f'{idx} for {k_stacked!r}'
                                    )

                                if stacked_state['kind'] == 'quantized':
                                    layer_parameter = SimpleNamespace(
                                        dtype=target_var.dtype,
                                        input_axis_count=getattr(
                                            target_var,
                                            'input_axis_count',
                                            None,
                                        ),
                                        quantization_batch_axis_count=max(
                                            0,
                                            getattr(
                                                target_var,
                                                'quantization_batch_axis_count',
                                                1,
                                            )
                                            - 1,
                                        ),
                                    )
                                    layer_value = quantize_weight(
                                        value,
                                        layer_parameter,
                                        stacked_state['rule'],
                                        stacked_state[
                                            'quantization_kind'
                                        ],
                                    )

                                    if stacked_state['value'] is None:
                                        def make_stacked_zero(arr: tp.Any) -> tp.Any:
                                            if arr is None:
                                                return None
                                            stacked_shape = (target_var.shape[0],) + arr.shape
                                            return np.zeros(stacked_shape, dtype=np.dtype(arr.dtype))

                                        stacked_state['value'] = jax.tree.map(make_stacked_zero, layer_value)

                                    def update_np_slice(s_arr: tp.Any, l_arr: tp.Any) -> None:
                                        if s_arr is not None and l_arr is not None:
                                            s_arr[idx] = np.asarray(l_arr)

                                    jax.tree.map(
                                        update_np_slice,
                                        stacked_state['value'],
                                        layer_value,
                                    )
                                    stacked_state['indices'].add(idx)
                                    finalize_stacked_state(k_stacked)
                                    continue
                                else:
                                    target_dtype = np.dtype(target_var.dtype)
                                    if value.dtype != target_dtype:
                                        value = value.astype(
                                            target_dtype,
                                            copy=False,
                                        )
                                    stacked_state['value'][idx] = np.asarray(value)
                                    stacked_state['indices'].add(idx)
                                    finalize_stacked_state(k_stacked)
                                continue

                        not_found_some = True
                        print(f"Warning: mapped key {k_mapped} found in checkpoint but not in model.")

        progress.close()

        # Move accumulated SeqStack weights to JAX
        for k_stacked, stacked_state in tuple(stacked_states.items()):
            target_var = current_state_dict[k_stacked]
            expected_indices = set(range(target_var.shape[0]))
            loaded_indices = stacked_state['indices']
            if loaded_indices != expected_indices:
                missing = sorted(expected_indices - loaded_indices)
                raise ValueError(
                    f'Checkpoint is missing layers {missing} for '
                    f'{k_stacked!r}'
                )
            finalize_stacked_state(k_stacked)

        if not_found_some:
            print("\nSome modules from the checkpoint were not found in this model.")
            print("You can try to map module names using module_map.")
            print("e.g. module_map = {'target_module': 'name_to_change'}")
        missing_parameters = sorted(set(current_state_dict) - set(new_state))
        if missing_parameters:
            preview = ', '.join(missing_parameters[:8])
            if len(missing_parameters) > 8:
                preview += f', ... ({len(missing_parameters)} total)'
            if not allow_unmatched:
                raise ValueError(
                    'Checkpoint did not provide values for model parameters: '
                    f'{preview}'
                )
            else:
                print(
                    'Warning: Checkpoint did not provide values for model parameters: '
                    f'{preview}'
                )

        # 6. Inject actual arrays into the PyTree skeleton
        state.checkpoint_module_map = cls._fired_rename_rules(
            checkpoint_keys_seen,
            module_map,
        )
        state.load_flat_state_dict(new_state)
        state.base_model_name_or_path = path_or_repo_str
        return state

__all__ = [
    'ModelOutput',
    'PretrainedModel',
]
