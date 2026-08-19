from taktiny.maestro.opus.gpt import GPTOSS
from taktiny.maestro import Maestro

def test():
    model = Maestro.eval_shape("openai/gpt-oss-20b", use_list=False)
    layers = model.model.layers
    if hasattr(layers, 'groups'):
        print("Num groups:", len(layers.groups))
    else:
        print("Num groups:", 1)
        
if __name__ == "__main__":
    test()
