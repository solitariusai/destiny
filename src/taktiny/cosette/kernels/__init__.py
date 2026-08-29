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
"""Experimental kernels for TakTiny (Attention, Megablox GMM, Ragged, Gather-Reduce)."""

from taktiny.cosette.kernels import attention
from taktiny.cosette.kernels import megablox
from taktiny.cosette.kernels import ragged
from taktiny.cosette.kernels import gather_reduce_pallas
from taktiny.cosette.kernels import gather_reduce_sc
from taktiny.cosette.kernels import sort_activations

from taktiny.cosette.kernels.sort_activations import route, unroute
from taktiny.cosette.kernels.gather_reduce_sc import sc_gather_reduce

__all__ = [
    "attention",
    "megablox",
    "ragged",
    "gather_reduce_pallas",
    "gather_reduce_sc",
    "sort_activations",
    "route",
    "unroute",
    "sc_gather_reduce",
]