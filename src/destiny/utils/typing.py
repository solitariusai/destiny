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
"""Shared type aliases and protocols used across TakTiny."""

from __future__ import annotations
import collections.abc as cab
import enum
import typing as tp
from os import PathLike as OSPathLike

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.typing import ArrayLike as JaxArrayLike
from jax.typing import DTypeLike

type Array          = jax.Array
type ArrayLike      = JaxArrayLike
type Activation     = str | cab.Callable[[Array], Array]
type DType          = DTypeLike
type Initializer    = cab.Callable[..., Array]
type PRNGKey        = jax.Array
type PyTree         = tp.Any
type Shape          = cab.Sequence[int]
type Axes           = int | cab.Sequence[int]
type MeshAxisName   = str | tuple[str, ...] | None
type LogicalRules   = cab.Sequence[tuple[str, MeshAxisName]]
type Sharding       = NamedSharding | PartitionSpec | None
type MeshLike       = Mesh | None
type PathLike       = str | OSPathLike[str]
type Batch          = cab.Mapping[str, PyTree]
type MutableBatch   = dict[str, PyTree]
type StateDict      = dict[str, PyTree]
type ParameterDict  = dict[str, tp.Any]
type ModuleFactory  = cab.Callable[..., tp.Any]
type LossFn         = cab.Callable[[tp.Any, Batch], Array]


@tp.runtime_checkable
class StatefulIterator[T](tp.Protocol):
    """cab.Iterator whose cursor can be checkpointed and restored."""

    def __iter__(self) -> cab.Iterator[T]: ...

    def __next__(self) -> T: ...

    def get_state(self) -> PyTree: ...

    def set_state(self, state: PyTree) -> None: ...


@tp.runtime_checkable
class EpochAware(tp.Protocol):
    """Data source that supports deterministic epoch selection."""
    def set_epoch(self, epoch: int) -> None: ...


class MeshAxis(enum.Enum):
    """
    Logical mesh axes.

    TP: Tensor parallel.
    DP: Data parallel.
    RP: Replicated (not sharded).
    """

    TP = 'tp'
    DP = 'dp'
    RP = None


__all__ = [
    'Activation',
    'Array',
    'ArrayLike',
    'Axes',
    'Batch',
    'DType',
    'EpochAware',
    'Initializer',
    'LogicalRules',
    'LossFn',
    'MeshAxis',
    'MeshAxisName',
    'MeshLike',
    'ModuleFactory',
    'MutableBatch',
    'PRNGKey',
    'ParameterDict',
    'PathLike',
    'PyTree',
    'Shape',
    'Sharding',
    'StateDict',
    'StatefulIterator',
]
