import jax
from taktiny.maestro import Maestro

def test():
    model = Maestro.eval_shape("google/gemma-3-270m", use_list=True)
    layers = list(model.model.layers.layers)
    
    l0 = layers[0]
    l5 = layers[5]
    
    struct0 = jax.tree_util.tree_structure(l0)
    struct5 = jax.tree_util.tree_structure(l5)
    
    s0 = str(struct0)
    s5 = str(struct5)
    
    if s0 != s5:
        for i in range(min(len(s0), len(s5))):
            if s0[i] != s5[i]:
                print(f"Diff at index {i}:")
                print("0:", s0[max(0, i-20):i+20])
                print("5:", s5[max(0, i-20):i+20])
                break

if __name__ == "__main__":
    test()
