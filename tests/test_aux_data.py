import jax
from taktiny.layers.attention import Attention
from taktiny import nn

def test():
    class Dummy(nn.Module):
        def __init__(self, x):
            self.x = x

    d1 = Dummy(512)
    d2 = Dummy(None)
    
    print("d1:", jax.tree_util.tree_structure(d1))
    print("d2:", jax.tree_util.tree_structure(d2))
    print("Eq?", jax.tree_util.tree_structure(d1) == jax.tree_util.tree_structure(d2))
    
if __name__ == "__main__":
    test()
