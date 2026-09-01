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
"""Construction and materialization of local pretrained checkpoints."""

from __future__ import annotations

import json
import os
import typing as tp
import warnings
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import qwix
from safetensors import safe_open
from taktiny import nn

from destiny.cosette.utils import ModuleMap
from destiny.utils.format import parse_size
from destiny.utils.logging import is_jax_rank_zero, tqdm
from destiny.utils.quantization import (
    merge_quantization,
    quantize_embedding_weight,
    quantize_linear_weight,
    resolve_quantization_rule,
)
from destiny.utils.sharding import create_sharding
from destiny.utils.typing import LogicalRules, PathLike


_SAFETENSORS_DTYPE_BYTES = {
    'BOOL': 1,
    'U8': 1,
    'I8': 1,
    'F8_E4M3': 1,
    'F8_E5M2': 1,
    'F8_E8M0': 1,
    'U16': 2,
    'I16': 2,
    'F16': 2,
    'BF16': 2,
    'U32': 4,
    'I32': 4,
    'F32': 4,
    'U64': 8,
    'I64': 8,
    'F64': 8,
}


def _checkpoint_files(
    directory: str,
    weights_filename: str,
) -> list[str]:
    if not isinstance(weights_filename, str):
        raise TypeError('weights_filename must be a string')
    if not weights_filename.endswith('.safetensors'):
        raise ValueError('weights_filename must end with .safetensors')
    index_path = os.path.join(directory, f'{weights_filename}.index.json')
    if os.path.isfile(index_path):
        with open(index_path, encoding='utf-8') as file:
            index = json.load(file)
        weight_map = index.get('weight_map')
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError('checkpoint index has no weight_map')
        filenames = dict.fromkeys(weight_map.values())
    else:
        filenames = {weights_filename: None}
    paths = [os.path.join(directory, filename) for filename in filenames]
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(missing[0])
    return paths


def _safetensors_metadata(paths: Sequence[str]) -> dict[str, str]:
    metadata = None
    for path in paths:
        with safe_open(path, framework='np', device='cpu') as checkpoint:
            shard_metadata = checkpoint.metadata() or {}
        if metadata is not None and metadata != shard_metadata:
            raise ValueError('checkpoint shards have inconsistent metadata')
        metadata = shard_metadata
    return metadata or {}


def _checkpoint_inventory(
    paths: Sequence[str],
) -> tuple[list[str], dict[str, str], dict[str, int]]:
    """Read checkpoint headers without decoding tensor payloads."""
    names = []
    locations = {}
    sizes = {}
    for path in paths:
        with safe_open(path, framework='np', device='cpu') as checkpoint:
            for name in checkpoint.keys():
                if name in locations:
                    raise ValueError(
                        f'checkpoint contains duplicate tensor {name!r}'
                    )
                tensor_slice = checkpoint.get_slice(name)
                dtype = tensor_slice.get_dtype()
                try:
                    itemsize = _SAFETENSORS_DTYPE_BYTES[dtype]
                except KeyError as error:
                    raise ValueError(
                        f'checkpoint tensor {name!r} has unsupported '
                        f'Safetensors dtype {dtype!r}'
                    ) from error
                elements = int(np.prod(tensor_slice.get_shape(), dtype=np.int64))
                names.append(name)
                locations[name] = path
                sizes[name] = elements * itemsize
    return names, locations, sizes


def _mapping_units(
    names: Sequence[str],
    rules: ModuleMap,
) -> list[tuple[str, ...]]:
    """Keep the inputs of N-to-one module-map rules in one decode unit."""
    parent = {name: name for name in names}

    def find(name: str) -> str:
        root = name
        while parent[root] != root:
            root = parent[root]
        while parent[name] != name:
            next_name = parent[name]
            parent[name] = root
            name = next_name
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    name_set = set(names)
    for rule in rules:
        source = rule[0]
        if isinstance(source, str):
            continue
        source_patterns = tuple(source)
        if len(source_patterns) < 2:
            continue
        primary = source_patterns[0]
        for name in names:
            if primary not in name:
                continue
            siblings = tuple(
                name.replace(primary, pattern)
                for pattern in source_patterns
            )
            if all(sibling in name_set for sibling in siblings):
                for sibling in siblings[1:]:
                    union(siblings[0], sibling)

    grouped: dict[str, list[str]] = {}
    for name in names:
        grouped.setdefault(find(name), []).append(name)
    return [tuple(group) for group in grouped.values()]


def _decode_checkpoint_unit(
    unit: Sequence[str],
    locations: Mapping[str, str],
) -> dict[str, np.ndarray]:
    decoded = {}
    checkpoints = {}
    with ExitStack() as stack:
        for name in unit:
            path = locations[name]
            checkpoint = checkpoints.get(path)
            if checkpoint is None:
                checkpoint = stack.enter_context(
                    safe_open(path, framework='np', device='cpu')
                )
                checkpoints[path] = checkpoint
            decoded[name] = checkpoint.get_tensor(name)
    return decoded


def _checkpoint_chunks(
    units: Sequence[Sequence[str]],
    locations: Mapping[str, str],
    sizes: Mapping[str, int],
    byte_budget: int,
) -> tp.Iterator[dict[str, np.ndarray]]:
    """Decode mapping units in batches bounded by ``byte_budget``."""
    pending = []
    pending_bytes = 0
    for unit in units:
        unit_bytes = sum(sizes[name] for name in unit)
        if pending and (
            byte_budget == 0
            or pending_bytes + unit_bytes > byte_budget
        ):
            names = tuple(
                name
                for pending_unit in pending
                for name in pending_unit
            )
            yield _decode_checkpoint_unit(names, locations)
            pending = []
            pending_bytes = 0
        pending.append(unit)
        pending_bytes += unit_bytes
        if byte_budget == 0 or pending_bytes >= byte_budget:
            names = tuple(
                name
                for pending_unit in pending
                for name in pending_unit
            )
            yield _decode_checkpoint_unit(names, locations)
            pending = []
            pending_bytes = 0
    if pending:
        names = tuple(
            name
            for pending_unit in pending
            for name in pending_unit
        )
        yield _decode_checkpoint_unit(names, locations)


def _grouped_stack_layout(
    parameters: Mapping[str, tp.Any],
) -> dict[tuple[str, ...], tuple[int, ...]]:
    sizes: dict[tuple[str, ...], dict[int, int]] = {}
    for name, parameter in parameters.items():
        parts = name.split('.')
        for position in range(len(parts) - 2):
            if (
                parts[position] != 'groups'
                or not parts[position + 1].isdigit()
                or parts[position + 2] != 'stacked'
            ):
                continue
            shape = tuple(parameter.shape)
            if not shape:
                raise ValueError(
                    f'grouped stacked parameter {name!r} has no layer axis'
                )
            root = tuple(parts[:position])
            group_index = int(parts[position + 1])
            group_sizes = sizes.setdefault(root, {})
            previous = group_sizes.setdefault(group_index, shape[0])
            if previous != shape[0]:
                raise ValueError(
                    f'grouped stack {".".join(root)!r} has inconsistent size'
                )
            break

    layouts = {}
    for root, indexed_sizes in sizes.items():
        indices = sorted(indexed_sizes)
        if indices != list(range(len(indices))):
            raise ValueError(
                f'grouped stack {".".join(root)!r} has non-contiguous groups'
            )
        layouts[root] = tuple(indexed_sizes[index] for index in indices)
    return layouts


def _resolve_stacked_parameter(
    name: str,
    parameters: Mapping[str, tp.Any],
    grouped_layouts: Mapping[tuple[str, ...], tuple[int, ...]],
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
                grouped_parts = [
                    *root,
                    'groups',
                    str(group_index),
                    'stacked',
                    *parts[position + 1:],
                ]
                grouped_name = '.'.join(grouped_parts)
                if grouped_name in parameters:
                    return grouped_name, layer_index - offset
                break
            offset += group_size
    return None


def _reshape_external_tensor(
    name: str,
    value: np.ndarray,
    target_shape: tuple[int, ...],
    *,
    external_layout: bool = True,
) -> np.ndarray:
    value = np.asarray(value)
    if external_layout and value.ndim == 2 and (
        name.endswith('.weight') or '.lora_' in name
    ):
        value = value.T
    elif (
        external_layout
        and
        value.ndim >= 3
        and name.endswith('.weight')
        and value.shape != target_shape
    ):
        convolution_shape = (
            *value.shape[2:],
            value.shape[1],
            value.shape[0],
        )
        if convolution_shape == target_shape:
            value = value.transpose(*range(2, value.ndim), 1, 0)
    if value.shape == target_shape:
        return value
    try:
        return value.reshape(target_shape)
    except ValueError as error:
        raise ValueError(
            f'checkpoint tensor {name!r} with shape {value.shape} cannot '
            f'be reshaped to {target_shape}'
        ) from error


def _module_map(
    model_type: type,
    module_map: Sequence[tuple[tp.Any, ...]] | ModuleMap | None,
) -> ModuleMap:
    defaults = getattr(model_type, '_default_module_map', ())
    rules = defaults.copy() if isinstance(defaults, ModuleMap) else ModuleMap(defaults)
    if module_map is not None:
        rules.extend(module_map)
    return rules


def _set_config_override(config: tp.Any, name: str, value: tp.Any) -> None:
    setattr(config, name, value)
    text_config = vars(config).get('text_config')
    if text_config is not None:
        setattr(text_config, name, value)


def _prepare_runtime_config(
    config: tp.Any,
    *,
    dtype: tp.Any,
    quant: tp.Any,
) -> tp.Any:
    uniform_quant = None
    if dtype is not None:
        dtype_name = dtype.lower() if isinstance(dtype, str) else None
        if dtype_name in {'fp8', 'int8', 'int4', 'nf4'}:
            compute_dtype = (
                getattr(config, 'torch_dtype', None)
                or getattr(config, 'dtype', None)
            )
            if (
                compute_dtype is None
                or (
                    isinstance(compute_dtype, str)
                    and compute_dtype.lower()
                    in {'fp8', 'int8', 'int4', 'nf4'}
                )
            ):
                compute_dtype = 'bfloat16'
            uniform_quant = dtype_name
            _set_config_override(config, 'dtype', compute_dtype)
            _set_config_override(config, 'torch_dtype', compute_dtype)
        else:
            _set_config_override(config, 'dtype', dtype)
            _set_config_override(config, 'torch_dtype', dtype)

    if quant is not None and uniform_quant is not None:
        quant = merge_quantization(quant, uniform_quant)
    elif quant is None:
        quant = uniform_quant
    if quant is not None:
        _set_config_override(config, 'quant', quant)
    return getattr(config, 'quant', None)


def _construct_abstract_model(
    model_type: type,
    config: tp.Any,
    *,
    mesh: jax.sharding.Mesh | None,
    sharding_rules: LogicalRules | None,
    stack_type: tp.Literal['stack', 'list'] | None,
    kwargs: dict[str, tp.Any],
) -> tp.Any:
    rngs = kwargs.pop('rngs', nn.Rngs(0))
    if stack_type is not None:
        config.stack_type = stack_type
    return jax.eval_shape(
        lambda: model_type(
            config,
            rngs=rngs,
            mesh=mesh,
            sharding_rules=sharding_rules,
            **kwargs,
        )
    )


def _place_loaded_model(
    model: tp.Any,
    mesh: jax.sharding.Mesh,
    sharding_rules: LogicalRules | None,
) -> None:
    """Place an already-decoded native quantized model on ``mesh``."""
    default_device = jax.devices()[0]

    def sharding(
        axis_names: tp.Any,
    ) -> tp.Any:
        if axis_names is not None:
            return create_sharding(
                mesh,
                axis_names,
                rules=sharding_rules,
            )
        return default_device

    def component_axes(axis_names: tp.Any, shape: Sequence[int]) -> tp.Any:
        if axis_names is None:
            return None
        return tuple(
            axis_name if size != 1 else None
            for axis_name, size in zip(axis_names, shape, strict=True)
        )

    for parameter in model.flat_parameter_dict().values():
        value = parameter.value
        axis_names = getattr(parameter, 'axis_names', None)
        if not isinstance(value, qwix.QArray):
            parameter.value = jax.device_put(
                value,
                sharding(axis_names),
            )
            continue
        zero_point = value.zero_point
        if zero_point is not None:
            zero_point = jax.device_put(
                zero_point,
                sharding(
                    component_axes(axis_names, zero_point.shape),
                ),
            )
        parameter.value = value.replace(
            qvalue=jax.device_put(
                value.qvalue,
                sharding(axis_names),
            ),
            scale=jax.device_put(
                value.scale,
                sharding(
                    component_axes(axis_names, value.scale.shape),
                ),
            ),
            zero_point=zero_point,
        )


def materialize_pretrained(
    model_type: type,
    directory: PathLike,
    config: tp.Any,
    *,
    source_name: str | None = None,
    module_map: Sequence[tuple[tp.Any, ...]] | ModuleMap | None = None,
    weights_filename: str = 'model.safetensors',
    dtype: tp.Any = None,
    quant: tp.Any = None,
    mesh: jax.sharding.Mesh | None = None,
    sharding_rules: LogicalRules | None = None,
    stack_type: tp.Literal['stack', 'list'] | None = None,
    allow_unmatched: bool = False,
    load_chunk_size: int | str | None = '1GB',
    show_progress: bool = False,
    **kwargs: tp.Any,
) -> tp.Any:
    """Construct ``model_type`` and stream a Safetensors checkpoint into it.

    ``load_chunk_size`` bounds decoded host batches. A single tensor mapping
    group may exceed the budget because all inputs to an N-to-one module-map
    transform must be resident together. ``None`` or ``0`` decodes one such
    group at a time.
    """
    root = os.path.abspath(os.fspath(directory))
    if not os.path.isdir(root):
        raise NotADirectoryError(root)
    if not isinstance(show_progress, bool):
        raise TypeError('show_progress must be a boolean')
    chunk_bytes = (
        parse_size(load_chunk_size)
        if load_chunk_size is not None
        else 0
    )
    effective_quant = _prepare_runtime_config(
        config,
        dtype=dtype,
        quant=quant,
    )

    paths = _checkpoint_files(root, weights_filename)
    metadata = _safetensors_metadata(paths)
    if os.path.isfile(os.path.join(root, 'adapter_config.json')):
        raise ValueError(
            'adapter checkpoints require an existing transformed base model'
        )

    if sharding_rules is None:
        sharding_rules = getattr(
            model_type,
            '_default_sharding_rules',
            None,
        )

    model = _construct_abstract_model(
        model_type,
        config,
        mesh=mesh,
        sharding_rules=sharding_rules,
        stack_type=stack_type,
        kwargs=dict(kwargs),
    )
    native_layout = metadata.get('destiny.format') == 'native'
    native_quantized = os.path.isfile(
        os.path.join(root, 'quantization_config.json')
    )
    if native_quantized or (
        native_layout
        and effective_quant is None
        and mesh is None
    ):
        model.load_pretrained(root)
        if native_quantized and mesh is not None:
            _place_loaded_model(model, mesh, sharding_rules)
        model.base_model_name_or_path = source_name or root
        return model

    rules = ModuleMap() if native_layout else _module_map(
        model_type,
        module_map,
    )
    checkpoint_names, locations, tensor_sizes = _checkpoint_inventory(paths)
    units = _mapping_units(checkpoint_names, rules)
    parameters = dict(model.flat_parameter_dict())
    grouped_layouts = _grouped_stack_layout(parameters)
    loaded = set()
    unexpected = set()
    stacked_states: dict[str, dict[str, tp.Any]] = {}
    cpu_device = jax.devices('cpu')[0]
    default_device = jax.devices()[0]
    quantizers: dict[tuple[tp.Any, ...], tp.Callable] = {}
    warned_unsharded = False

    def parameter_sharding(
        parameter: tp.Any,
        axis_names: tp.Any = None,
        *,
        use_explicit: bool = True,
    ) -> tp.Any:
        nonlocal warned_unsharded
        sharding = (
            getattr(parameter, 'sharding', None)
            if use_explicit
            else None
        )
        if sharding is None and axis_names is not None and mesh is not None:
            sharding = create_sharding(
                mesh,
                axis_names,
                rules=sharding_rules,
            )
        if sharding is None:
            if mesh is not None and not warned_unsharded:
                warned_unsharded = True
                warnings.warn(
                    'some parameters have no resolved mesh sharding and '
                    'will be placed on one device',
                    RuntimeWarning,
                    stacklevel=2,
                )
            sharding = default_device
        return sharding

    def quantization_rule(
        name: str,
        parameter: tp.Any,
    ) -> tuple[tp.Any, str]:
        kind = getattr(parameter, 'quantization_kind', 'dot_general')
        rule = resolve_quantization_rule(
            getattr(parameter, 'quantization', None),
            name.rpartition('.')[0],
            op_name=kind,
        )
        return rule, kind

    def quantize_weight(
        value: tp.Any,
        parameter: tp.Any,
        rule: tp.Any,
        kind: str,
    ) -> qwix.QArray:
        batch_axes = getattr(
            parameter,
            'quantization_batch_axis_count',
            0,
        )
        input_axes = getattr(parameter, 'input_axis_count', None)
        parameter_dtype = jnp.dtype(parameter.dtype)
        cache_key = (
            kind,
            batch_axes,
            input_axes,
            str(parameter_dtype),
            str(rule.weight_qtype),
            rule.tile_size,
            rule.weight_calibration_method,
        )
        quantizer = quantizers.get(cache_key)
        if quantizer is None:
            quantization_metadata = SimpleNamespace(
                dtype=parameter_dtype,
                input_axis_count=input_axes,
                quantization_batch_axis_count=batch_axes,
            )
            helper = (
                quantize_embedding_weight
                if kind == 'embedding'
                else quantize_linear_weight
            )

            def apply(array: tp.Any) -> qwix.QArray:
                return helper(array, quantization_metadata, rule)

            quantizer = jax.jit(apply)
            quantizers[cache_key] = quantizer
        return quantizer(jax.device_put(value, cpu_device))

    def component_axis_names(
        axis_names: tp.Any,
        shape: Sequence[int],
    ) -> tp.Any:
        if axis_names is None:
            return None
        return tuple(
            axis_name if size != 1 else None
            for axis_name, size in zip(axis_names, shape, strict=True)
        )

    def place_qarray(value: qwix.QArray, parameter: tp.Any) -> qwix.QArray:
        axis_names = getattr(parameter, 'axis_names', None)
        qvalue = jax.device_put(
            value.qvalue,
            parameter_sharding(parameter, axis_names),
        )
        scale = jax.device_put(
            value.scale,
            parameter_sharding(
                parameter,
                component_axis_names(axis_names, value.scale.shape),
                use_explicit=False,
            ),
        )
        zero_point = value.zero_point
        if zero_point is not None:
            zero_point = jax.device_put(
                zero_point,
                parameter_sharding(
                    parameter,
                    component_axis_names(axis_names, zero_point.shape),
                    use_explicit=False,
                ),
            )
        return value.replace(
            qvalue=qvalue,
            scale=scale,
            zero_point=zero_point,
        )

    pending: list[tuple[str, np.ndarray, tp.Any, tp.Any]] = []

    def flush_pending() -> None:
        if not pending:
            return
        values = [item[1] for item in pending]
        shardings = [item[3] for item in pending]
        placed = jax.device_put(values, shardings)
        for (name, _, parameter, _), value in zip(
            pending,
            placed,
            strict=True,
        ):
            parameter.value = value
            loaded.add(name)
        pending.clear()

    def stage_parameter(
        name: str,
        value: np.ndarray,
        parameter: tp.Any,
    ) -> None:
        if name in loaded or any(item[0] == name for item in pending):
            raise ValueError(
                f'checkpoint contains duplicate value for {name!r}'
            )
        rule, kind = quantization_rule(name, parameter)
        if rule is not None:
            parameter.trainable = False
            parameter.value = place_qarray(
                quantize_weight(value, parameter, rule, kind),
                parameter,
            )
            loaded.add(name)
            return
        target_dtype = np.dtype(parameter.dtype)
        if value.dtype != target_dtype:
            value = value.astype(target_dtype, copy=False)
        pending.append(
            (
                name,
                value,
                parameter,
                parameter_sharding(
                    parameter,
                    getattr(parameter, 'axis_names', None),
                ),
            )
        )

    def finalize_stacked(name: str) -> bool:
        stacked_state = stacked_states[name]
        parameter = parameters[name]
        if stacked_state['indices'] != set(range(parameter.shape[0])):
            return False
        if stacked_state['kind'] == 'quantized':
            value = place_qarray(stacked_state['value'], parameter)
        else:
            value = jax.device_put(
                stacked_state['value'],
                parameter_sharding(
                    parameter,
                    getattr(parameter, 'axis_names', None),
                ),
            )
        parameter.value = jax.block_until_ready(value)
        loaded.add(name)
        del stacked_states[name]
        return True

    def stage_stacked(
        value: np.ndarray,
        stacked_name: str,
        layer_index: int,
    ) -> None:
        parameter = parameters[stacked_name]
        stacked_state = stacked_states.get(stacked_name)
        if stacked_state is None:
            if stacked_name in loaded:
                raise ValueError(
                    f'checkpoint contains duplicate value for {stacked_name!r}'
                )
            rule, kind = quantization_rule(stacked_name, parameter)
            if rule is None:
                stacked_state = {
                    'kind': 'dense',
                    'value': np.zeros(
                        parameter.shape,
                        dtype=np.dtype(parameter.dtype),
                    ),
                    'indices': set(),
                }
            else:
                parameter.trainable = False
                stacked_state = {
                    'kind': 'quantized',
                    'value': None,
                    'indices': set(),
                    'rule': rule,
                    'quantization_kind': kind,
                }
            stacked_states[stacked_name] = stacked_state
        if layer_index in stacked_state['indices']:
            raise ValueError(
                f'checkpoint contains duplicate layer {layer_index} '
                f'for {stacked_name!r}'
            )

        if stacked_state['kind'] == 'dense':
            target_dtype = np.dtype(parameter.dtype)
            if value.dtype != target_dtype:
                value = value.astype(target_dtype, copy=False)
            stacked_state['value'][layer_index] = value
        else:
            layer_parameter = SimpleNamespace(
                dtype=parameter.dtype,
                input_axis_count=getattr(
                    parameter,
                    'input_axis_count',
                    None,
                ),
                quantization_batch_axis_count=max(
                    0,
                    getattr(
                        parameter,
                        'quantization_batch_axis_count',
                        1,
                    ) - 1,
                ),
            )
            layer_value = quantize_weight(
                value,
                layer_parameter,
                stacked_state['rule'],
                stacked_state['quantization_kind'],
            )
            if stacked_state['value'] is None:
                def stack_zeros(array: tp.Any) -> tp.Any:
                    if array is None:
                        return None
                    shape = (parameter.shape[0], *array.shape)
                    return np.zeros(shape, dtype=np.dtype(array.dtype))

                stacked_state['value'] = jax.tree.map(
                    stack_zeros,
                    layer_value,
                )

            def set_layer(stacked: tp.Any, layer: tp.Any) -> None:
                if stacked is not None and layer is not None:
                    stacked[layer_index] = np.asarray(layer)

            jax.tree.map(set_layer, stacked_state['value'], layer_value)
        stacked_state['indices'].add(layer_index)
        finalize_stacked(stacked_name)

    progress = tqdm(
        total=len(checkpoint_names),
        desc='Loading checkpoint',
        unit='tensor',
        dynamic_ncols=True,
        disable=not show_progress or not is_jax_rank_zero(),
    )
    try:
        for chunk in _checkpoint_chunks(
            units,
            locations,
            tensor_sizes,
            chunk_bytes,
        ):
            with jax.default_device(cpu_device):
                mapped = rules.apply(chunk)
            progress.update(len(chunk))
            for name, value in mapped.items():
                parameter = parameters.get(name)
                if parameter is not None:
                    reshaped = _reshape_external_tensor(
                        name,
                        value,
                        tuple(parameter.shape),
                        external_layout=not native_layout,
                    )
                    stage_parameter(name, reshaped, parameter)
                    continue
                resolved = _resolve_stacked_parameter(
                    name,
                    parameters,
                    grouped_layouts,
                )
                if resolved is None:
                    unexpected.add(name)
                    continue
                stacked_name, layer_index = resolved
                target_shape = tuple(parameters[stacked_name].shape[1:])
                reshaped = _reshape_external_tensor(
                    name,
                    value,
                    target_shape,
                    external_layout=not native_layout,
                )
                stage_stacked(
                    reshaped,
                    stacked_name,
                    layer_index,
                )
            flush_pending()
    finally:
        progress.close()

    for name, stacked_state in tuple(stacked_states.items()):
        expected_indices = set(range(parameters[name].shape[0]))
        missing_indices = sorted(expected_indices - stacked_state['indices'])
        if missing_indices:
            missing_text = ', '.join(map(str, missing_indices))
            raise ValueError(
                f'checkpoint is missing layers {missing_text} for {name!r}'
            )
        finalize_stacked(name)

    missing = sorted(set(parameters) - loaded)
    if missing and not allow_unmatched:
        preview = ', '.join(missing[:8])
        raise ValueError(
            f'checkpoint did not provide model parameters: {preview}'
        )
    model.base_model_name_or_path = source_name or root
    model.unexpected_checkpoint_keys = tuple(sorted(unexpected))
    model.missing_checkpoint_keys = tuple(missing)
    return model


__all__ = ['materialize_pretrained']
