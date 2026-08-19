import jax
from taktiny.maestro import Maestro

def test():
    model = Maestro.eval_shape("google/gemma-4-26B-A4B-it", use_list=False)
    print("Num groups:", len(model.language_model.model.layers.groups) if hasattr(model.language_model.model.layers, 'groups') else 1)
    # let's just print a bit of the tree for validation
    s = str(model)
    print(s[:500])
    
if __name__ == "__main__":
    test()
