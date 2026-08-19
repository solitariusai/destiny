import functools
import jax

p1 = functools.partial(jax.nn.gelu, approximate=True)
p2 = functools.partial(jax.nn.gelu, approximate=True)

print(p1 == p2)
