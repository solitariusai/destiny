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
# WITHOUT WARRANTIES OR CONDITIONS OF tp.Any KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Layer utilities"""

import typing as tp

import jax
import jax.numpy as jnp
from taktiny import nn


class StackLayer(nn.Module):
    """Construct and execute repeated layers through one container-neutral API.

    ``nn.List`` executes layers with a Python loop while ``nn.SeqStack`` scans
    stacked parameters with :func:`jax.lax.scan`. ``call_stack`` presents the
    same callback contract for both containers and optionally slices a PyTree
    whose leading axis contains per-layer values, such as a KV cache.
    """

    @classmethod
    def init_stack(
        cls,
        layer_type: type[nn.Module],
        *args: tp.Any,
        num_stacks: int,
        stack_type: tp.Literal['list', 'stack'] | None = None,
        **kwargs: tp.Any,
    ) -> nn.List | nn.SeqStack:
        stack_type = stack_type or 'stack'
        if not isinstance(layer_type, type) or not issubclass(
            layer_type,
            nn.Module,
        ):
            raise TypeError('layer_type must be an nn.Module subclass')
        if not isinstance(num_stacks, int) or isinstance(num_stacks, bool):
            raise TypeError('num_stacks must be an integer')
        if num_stacks <= 0:
            raise ValueError('num_stacks must be positive')
        if stack_type not in {'list', 'stack'}:
            raise ValueError("stack_type must be 'list' or 'stack'")

        layers = []
        for layer_idx in range(num_stacks):
            layer = layer_type(
                *args,
                layer_idx=layer_idx,
                **kwargs,
            )
            layers.append(layer)

        if stack_type == 'list':
            return nn.List(layers)
        return nn.SeqStack(layers)

    @staticmethod
    def _validate_per_layer(
        per_layer: tp.Any,
        num_stacks: int,
    ) -> None:
        if per_layer is None:
            return
        for leaf in jax.tree.leaves(per_layer):
            shape = getattr(leaf, 'shape', None)
            if not shape:
                raise ValueError(
                    'every per_layer leaf must have a leading layer axis'
                )
            if shape[0] != num_stacks:
                raise ValueError(
                    'per_layer leading axes must match the number of layers; '
                    f'got {shape[0]} and {num_stacks}'
                )

    @staticmethod
    def _stack_outputs(outputs: tp.Sequence[tp.Any]) -> tp.Any:
        if not outputs or all(output is None for output in outputs):
            return None
        if any(output is None for output in outputs):
            raise ValueError(
                'all layers must either return an output or return None'
            )
        return jax.tree.map(lambda *values: jnp.stack(values), *outputs)

    @classmethod
    def call_stack(
        cls,
        layers: nn.List | nn.SeqStack,
        function: tp.Callable[
            [nn.Module, tp.Any, tp.Any],
            tuple[tp.Any, tp.Any],
        ],
        carry: tp.Any,
        *,
        per_layer: tp.Any = None,
        with_layer_index: bool = False,
    ) -> tuple[tp.Any, tp.Any]:
        if not isinstance(layers, (nn.List, nn.SeqStack)):
            raise TypeError('layers must be nn.List or nn.SeqStack')
        if not callable(function):
            raise TypeError('function must be callable')

        num_stacks = len(layers)
        cls._validate_per_layer(per_layer, num_stacks)

        if isinstance(layers, nn.List):
            outputs = []
            for layer_idx, layer in enumerate(layers):
                layer_input = None
                if per_layer is not None:
                    layer_input = jax.tree.map(
                        lambda value: value[layer_idx],
                        per_layer,
                    )
                if with_layer_index:
                    carry, output = function(
                        layer,
                        carry,
                        layer_input,
                        jnp.asarray(layer_idx, dtype=jnp.int32),
                    )
                else:
                    carry, output = function(layer, carry, layer_input)
                outputs.append(output)
            return carry, cls._stack_outputs(outputs)

        def scan_layer(
            layer: nn.Module,
            scan_carry: tuple[tp.Any, jax.Array],
        ) -> tuple[tuple[tp.Any, jax.Array], tp.Any]:
            current_carry, layer_idx = scan_carry
            layer_input = None
            if per_layer is not None:
                layer_input = jax.tree.map(
                    lambda value: jax.lax.dynamic_index_in_dim(
                        value,
                        layer_idx,
                        axis=0,
                        keepdims=False,
                    ),
                    per_layer,
                )
            if with_layer_index:
                current_carry, output = function(
                    layer,
                    current_carry,
                    layer_input,
                    layer_idx,
                )
            else:
                current_carry, output = function(
                    layer,
                    current_carry,
                    layer_input,
                )
            return (current_carry, layer_idx + 1), output

        (carry, _), outputs = layers(
            scan_layer,
            (carry, jnp.asarray(0, dtype=jnp.int32)),
        )
        return carry, outputs