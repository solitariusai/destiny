import jax
from taktiny.maestro import Maestro
from taktiny.nn.block import _stack_compatible

def test():
    model = Maestro.eval_shape("openai/gpt-oss-20b", use_list=True)
    layers = list(model.model.layers.layers)
    for layer in layers:
        layer.layer_idx = None
    
    treedef0 = jax.tree_util.tree_structure(layers[0])
    treedef1 = jax.tree_util.tree_structure(layers[1])
    
    s0 = str(treedef0)
    s1 = str(treedef1)
    
    if s0 != s1:
        for i in range(min(len(s0), len(s1))):
            if s0[i] != s1[i]:
                print(f"Diff at index {i}:")
                print("0:", s0[max(0, i-20):i+20])
                print("1:", s1[max(0, i-20):i+20])
                break

if __name__ == "__main__":
    test()
