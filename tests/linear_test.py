import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from taktiny import nn
from taktiny.utils.typing import ShardMode


def test_linear_explicit_sharding_covers_biased_output():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('data',))
    out_sharding = NamedSharding(mesh, P())
    layer = nn.Linear(
        2,
        3,
        bias=True,
        shard_mode=ShardMode.EXPLICIT,
        rngs=nn.Rngs(0),
    )
    apply = lambda value: layer(value, out_sharding=out_sharding)
    x = jnp.ones((2, 2))

    jaxpr = jax.make_jaxpr(apply)(x).jaxpr
    output = jax.jit(apply)(x)

    assert jaxpr.eqns[-1].primitive.name == 'sharding_constraint'
    assert output.sharding.is_equivalent_to(out_sharding, output.ndim)
