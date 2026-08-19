import jax
from taktiny.maestro import Maestro
from taktiny.nn.block import SeqStack

def test():
    model = Maestro.eval_shape("google/gemma-3-270m", use_list=True)
    layers = list(model.model.layers.layers)
    for layer in layers:
        layer.layer_idx = None
    stack = SeqStack(layers)
    print("Num stack groups:", len(stack.groups) if hasattr(stack, 'groups') else 1)
    
if __name__ == "__main__":
    test()
