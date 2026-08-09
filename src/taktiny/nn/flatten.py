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
"""Shape-only flattening modules."""
from __future__ import annotations
from collections.abc import Sequence
import math
import jax
import jax.numpy as jnp
from taktiny import nn

def _canonical_dim(dim: int, ndim: int, *, name: str) -> int:
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise TypeError(f'{name} must be an integer')

    if ndim == 0:
        if dim in {-1, 0}:
            return 0
        raise ValueError(
            f'{name}={dim} is out of range for a scalar input'
        )

    canonical = dim + ndim if dim < 0 else dim
    if canonical < 0 or canonical >= ndim:
        raise ValueError(
            f'{name}={dim} is out of range for an input with {ndim} dimensions'
        )
    return canonical


class Flatten(nn.Module):
    """Collapse a contiguous range of dimensions into one dimension."""

    def __init__(
        self,
        start_dim: int = 1,
        end_dim: int = -1,
    ) -> None:
        if not isinstance(start_dim, int) or isinstance(start_dim, bool):
            raise TypeError('start_dim must be an integer')
        if not isinstance(end_dim, int) or isinstance(end_dim, bool):
            raise TypeError('end_dim must be an integer')
        self.start_dim = start_dim
        self.end_dim = end_dim

    def __call__(self, x: jax.Array) -> jax.Array:
        start_dim = _canonical_dim(
            self.start_dim,
            x.ndim,
            name='start_dim',
        )
        end_dim = _canonical_dim(
            self.end_dim,
            x.ndim,
            name='end_dim',
        )
        if start_dim > end_dim:
            raise ValueError(
                'start_dim must refer to a dimension before or equal to end_dim'
            )

        flattened_size = math.prod(x.shape[start_dim:end_dim + 1])
        shape = (
            *x.shape[:start_dim],
            flattened_size,
            *x.shape[end_dim + 1:],
        )
        return jnp.reshape(x, shape)

    def extra_repr(self) -> str:
        return f'start_dim={self.start_dim}, end_dim={self.end_dim}'


class Unflatten(nn.Module):
    """Expand one dimension into a specified sequence of dimensions."""

    def __init__(
        self,
        dim: int,
        unflattened_size: int | Sequence[int],
    ) -> None:
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError('dim must be an integer')
        if isinstance(unflattened_size, int):
            unflattened_size = (unflattened_size,)
        elif isinstance(unflattened_size, Sequence) and not isinstance(
            unflattened_size,
            (str, bytes),
        ):
            unflattened_size = tuple(unflattened_size)
        else:
            raise TypeError(
                'unflattened_size must be an integer or a sequence of integers'
            )

        if not unflattened_size:
            raise ValueError('unflattened_size must contain at least one dimension')
        if any(
            not isinstance(size, int) or isinstance(size, bool)
            for size in unflattened_size
        ):
            raise TypeError('unflattened_size values must be integers')
        if any(size < -1 for size in unflattened_size):
            raise ValueError(
                'unflattened_size values must be non-negative or -1'
            )
        if unflattened_size.count(-1) > 1:
            raise ValueError('only one unflattened dimension may be inferred')

        self.dim = dim
        self.unflattened_size = unflattened_size

    def __call__(self, x: jax.Array) -> jax.Array:
        if x.ndim == 0:
            raise ValueError('cannot unflatten a scalar input')
        dim = _canonical_dim(self.dim, x.ndim, name='dim')
        sizes = self.unflattened_size
        flattened_size = x.shape[dim]

        if -1 in sizes:
            known_size = math.prod(size for size in sizes if size != -1)
            if known_size == 0:
                raise ValueError(
                    'cannot infer an unflattened dimension when the known '
                    'dimensions have size zero'
                )
            if flattened_size % known_size:
                raise ValueError(
                    f'dimension of size {flattened_size} cannot be unflattened '
                    f'into {sizes}'
                )
            inferred_size = flattened_size // known_size
            sizes = tuple(
                inferred_size if size == -1 else size
                for size in sizes
            )
        elif math.prod(sizes) != flattened_size:
            raise ValueError(
                f'dimension of size {flattened_size} cannot be unflattened '
                f'into {sizes}'
            )

        shape = (*x.shape[:dim], *sizes, *x.shape[dim + 1:])
        return jnp.reshape(x, shape)

    def extra_repr(self) -> str:
        return f'dim={self.dim}, unflattened_size={self.unflattened_size}'


__all__ = ['Flatten', 'Unflatten']
