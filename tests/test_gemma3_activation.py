from taktiny.maestro import Maestro
def test():
    model = Maestro.eval_shape("google/gemma-3-270m", use_list=False)
    # Check the activation in the first layer's MLP
    print("Activation:", getattr(model.model.layers.stacked.mlp, 'activation', None))
    print("Activation eq:", model.model.layers.stacked.mlp.activation == model.model.layers.stacked.mlp.activation)

if __name__ == "__main__":
    test()
