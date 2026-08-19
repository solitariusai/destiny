import jax
from taktiny.maestro import Maestro
import json
import os

def test():
    # create a dummy config
    config = {
        "architectures": ["GptOssForCausalLM"],
        "model_type": "gpt_oss",
        "vocab_size": 100,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "max_position_embeddings": 512,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "rms_norm_eps": 1e-6
    }
    os.makedirs("dummy_gptoss", exist_ok=True)
    with open("dummy_gptoss/config.json", "w") as f:
        json.dump(config, f)
    
    print("Trying eval_shape...")
    try:
        model = Maestro.eval_shape("dummy_gptoss", local=True)
        print("Success! Model instance:", model.__class__.__name__)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
