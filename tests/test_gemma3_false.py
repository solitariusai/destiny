from taktiny.maestro import Maestro

def test():
    model = Maestro.eval_shape("google/gemma-3-270m", use_list=False)
    print(model)

if __name__ == "__main__":
    test()
