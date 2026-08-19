from taktiny.maestro import Maestro
from taktiny.nn.block import SeqStack

def test():
    model = Maestro.eval_shape("google/gemma-3-270m", use_list=False)
    stack = model.model.layers
    
    print("Num stack groups:", len(stack.groups) if hasattr(stack, 'groups') else 1)
    
if __name__ == "__main__":
    test()
