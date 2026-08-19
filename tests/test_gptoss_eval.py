from taktiny.maestro import Maestro
def test():
    model = Maestro.eval_shape("openai/gpt-oss-20b", use_list=False)
    print("Model structure:")
    print(model)

if __name__ == "__main__":
    test()
