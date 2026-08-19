import jax
from taktiny.maestro import Maestro
from taktiny.nn.block import SeqStack

def test():
    model = Maestro.eval_shape("google/gemma-3-270m", use_list=False)
    stack = model.model.layers
    # Since stack is SeqStack(18), let's see its 'stacked' attribute
    print("Stacked layer window_size:", getattr(stack.stacked.self_attn, 'window_size', 'NOT FOUND'))
    
if __name__ == "__main__":
    test()
