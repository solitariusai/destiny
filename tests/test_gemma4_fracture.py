import jax
from taktiny.maestro import Maestro

def test():
    model = Maestro.eval_shape("google/gemma-4-26B-A4B-it", use_list=False)
    layers = model.language_model.model.layers
    if hasattr(layers, 'groups'):
        print("Num groups:", len(layers.groups))
    else:
        print("Num groups: 1")

if __name__ == "__main__":
    test()
