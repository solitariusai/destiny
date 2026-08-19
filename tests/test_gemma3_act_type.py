from taktiny.maestro import Maestro
def test():
    model = Maestro.eval_shape("google/gemma-3-270m", use_list=False)
    act = model.model.layers.stacked.mlp.activation
    print("Activation:", act)
    print("Is lambda?", act.__name__ == '<lambda>')

if __name__ == "__main__":
    test()
