from taktiny.cosettes.transformers.gemma import Gemma3DecoderLayer
from taktiny import nn
from taktiny.maestro.config import ModelConfig
import jax

class Gemma4DecoderLayer(Gemma3DecoderLayer):
    def __init__(self, config, rngs, layer_idx=None):
        self.config = config
        super().__init__(config=config, rngs=rngs, layer_idx=layer_idx)

    def _create_module(self, *, name: str, module_type: type[nn.Module] | nn.Module, **kwargs) -> tuple[nn.Module, str]:
        text_config = getattr(self.config, 'text_config', self.config)
        enable_moe = getattr(text_config, 'enable_moe_block', False)
        
        if name == 'mlp' and enable_moe:
            from taktiny.layers.ffn import MoeFFN
            from taktiny.nn._continuo import _resolve_activation
            module = MoeFFN(
                hidden_size=kwargs['hidden_size'],
                intermediate_size=getattr(text_config, 'moe_intermediate_size', kwargs['intermediate_size']),
                num_experts=getattr(text_config, 'num_experts', 1),
                num_experts_per_tok=getattr(text_config, 'top_k_experts', 1),
                activation=_resolve_activation(getattr(text_config, 'hidden_activation', 'gelu_pytorch_tanh')),
                bias=kwargs['mlp_bias'],
                dtype=kwargs.get('dtype', None),
                rngs=kwargs['rngs'],
            )
            return module, 'residual'
            
        return super()._create_module(name=name, module_type=module_type, **kwargs)

def test():
    config = ModelConfig(
        hidden_size=128, 
        head_dim=32,
        intermediate_size=256, 
        num_attention_heads=4, 
        num_key_value_heads=4,
        max_position_embeddings=1024,
        text_config=ModelConfig(
            enable_moe_block=True,
            moe_intermediate_size=512,
            num_experts=16,
            top_k_experts=4,
            hidden_activation='gelu'
        )
    )
    layer = Gemma4DecoderLayer(config, rngs=nn.Rngs(jax.random.PRNGKey(0)), layer_idx=0)
    print("MLP type:", type(layer.mlp))

if __name__ == "__main__":
    test()
