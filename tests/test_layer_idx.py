from taktiny.maestro.opus.gpt import GptOssDecoderLayer
from taktiny.maestro.config import ModelConfig
from taktiny import nn
import jax

def test():
    config = ModelConfig(hidden_size=32, intermediate_size=64, num_local_experts=2, num_experts_per_tok=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128, rope_theta=1000)
    layer = GptOssDecoderLayer(config, rngs=nn.Rngs(jax.random.PRNGKey(0)), layer_idx=1)
    
    print("Before:", layer.layer_idx)
    layer.layer_idx = None
    print("After:", layer.layer_idx)
    
    leaves, treedef = jax.tree_util.tree_flatten(layer)
    print("Treedef:", treedef)

if __name__ == "__main__":
    test()
