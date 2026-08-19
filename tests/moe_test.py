import jax
import jax.numpy as jnp
from taktiny import nn
from taktiny.cosettes.layers.ffn import MoeFFN

def test_moe():
    rngs = jax.random.PRNGKey(0)
    # Using 3 experts, 2 experts per token
    moe = MoeFFN(hidden_size=4, intermediate_size=8, num_experts=3, num_experts_per_tok=2, rngs=nn.Rngs(rngs))
    
    # Run forward pass
    x = jax.random.normal(rngs, (2, 3, 4)) # [batch, seq_len, hidden_size]
    print("Testing forward pass...")
    out = moe(x)
    print("Output shape:", out.shape)
    assert out.shape == (2, 3, 4)
    print("Forward pass complete and shapes are correct!")

if __name__ == "__main__":
    test_moe()
