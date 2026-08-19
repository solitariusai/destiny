import jax
from taktiny.maestro import Maestro

def test():
    print("Trying eval_shape...")
    try:
        model = Maestro.eval_shape("openai/gpt-oss-20b")
        print("Success! Model instance:", model.__class__.__name__)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
