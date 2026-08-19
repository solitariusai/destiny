import jax
from taktiny.maestro import Maestro

def test():
    model = Maestro.eval_shape("google/gemma-3-270m", use_list=True)
    layers = list(model.model.layers.layers)
    
    l0 = layers[0] # sliding
    l5 = layers[5] # full
    
    print("l0 window_size:", l0.self_attn.window_size)
    print("l5 window_size:", l5.self_attn.window_size)
    
    struct0 = jax.tree_util.tree_structure(l0)
    struct5 = jax.tree_util.tree_structure(l5)
    
    print("Same structure?", struct0 == struct5)
    
if __name__ == "__main__":
    test()
