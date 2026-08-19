from taktiny.maestro import Maestro
import jax

def test():
    model = Maestro.eval_shape("google/gemma-3-270m", use_list=False)
    layers = model.model.layers
    if hasattr(layers, 'groups'):
        print("Num groups:", len(layers.groups))
    else:
        print("Num groups: 1")

if __name__ == "__main__":
    test()
