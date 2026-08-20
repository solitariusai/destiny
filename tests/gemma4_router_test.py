import jax
import jax.numpy as jnp
import numpy as np
import qwix
from safetensors.numpy import save_file

from taktiny import nn
from taktiny.cosettes.overture import PretrainedModel
from taktiny.cosettes.transformers.gemma import Gemma4MoeFFN, Gemma4Router
from taktiny.maestro.config import ModelConfig
from taktiny.maestro.opus.gemma import _GEMMA4_MODULE_MAP
from taktiny.utils.weights import map_state_dict


class _NestedTextLoadableModel(PretrainedModel):
    def __init__(
        self,
        config,
        rngs,
        mesh=None,
        sharding_rules=None,
    ):
        del mesh, sharding_rules
        self.config = config
        text_config = vars(config)['text_config']
        self.proj = nn.Linear(
            text_config.hidden_size,
            3,
            bias=False,
            dtype=text_config.torch_dtype,
            quant=text_config.quant,
            rngs=rngs,
        )


def test_gemma4_router_matches_normalized_scaled_top_k_routing():
    router = Gemma4Router(
        hidden_size=4,
        num_experts=3,
        num_experts_per_tok=2,
        dtype=jnp.float32,
        rngs=nn.Rngs(0),
    )
    router.proj.weight.value = jnp.asarray([
        [1.0, -1.0, 0.5],
        [0.5, 0.25, -0.5],
        [-0.25, 0.75, 1.0],
        [1.5, 0.0, -1.0],
    ])
    router.scale.value = jnp.asarray([1.0, 2.0, 0.5, 1.5])
    router.per_expert_scale.value = jnp.asarray([1.0, 0.5, 2.0])
    hidden_states = jnp.asarray([
        [1.0, 2.0, -1.0, 0.5],
        [-0.5, 1.0, 2.0, -1.0],
    ])

    probabilities, weights, indices = router(hidden_states)

    variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
    normalized = hidden_states * jax.lax.rsqrt(variance + 1e-6)
    scaled = normalized * router.scale.value * (4 ** -0.5)
    expected_probabilities = jax.nn.softmax(
        scaled @ router.proj.weight.value,
        axis=-1,
    )
    expected_weights, expected_indices = jax.lax.top_k(
        expected_probabilities,
        2,
    )
    expected_weights /= expected_weights.sum(axis=-1, keepdims=True)
    expected_weights *= router.per_expert_scale.value[expected_indices]

    assert jnp.allclose(probabilities, expected_probabilities)
    assert jnp.array_equal(indices, expected_indices)
    assert jnp.allclose(weights, expected_weights)


def test_gemma4_checkpoint_map_preserves_all_router_parameters():
    source = {
        'model.layers.0.router.proj.weight': jnp.zeros((3, 4)),
        'model.layers.0.router.scale': jnp.ones((4,)),
        'model.layers.0.router.per_expert_scale': jnp.ones((3,)),
    }

    mapped = map_state_dict(source, _GEMMA4_MODULE_MAP)

    assert set(mapped) == {
        'model.layers.0.experts.router.proj.weight',
        'model.layers.0.experts.router.scale',
        'model.layers.0.experts.router.per_expert_scale',
    }


def test_quant_override_reaches_nested_text_config(tmp_path):
    save_file(
        {'proj.weight': np.arange(12, dtype=np.float32).reshape(3, 4)},
        tmp_path / 'model.safetensors',
    )
    config = ModelConfig(
        text_config=ModelConfig(
            hidden_size=4,
            torch_dtype='bfloat16',
        ),
    )

    model = _NestedTextLoadableModel.from_pretrained(
        tmp_path,
        config,
        local=True,
        quant='int4',
    )

    assert config.quant == 'int4'
    assert config.text_config.quant == 'int4'
    assert isinstance(model.proj.weight.value, qwix.QArray)
    assert model.proj.weight.value.qtype == 'int4'


def test_quantized_gemma4_router_and_experts_run_on_reference_gmm():
    moe = Gemma4MoeFFN(
        hidden_size=8,
        intermediate_size=4,
        num_experts=3,
        num_experts_per_tok=2,
        dtype=jnp.bfloat16,
        rngs=nn.Rngs(0),
        quant='int4',
    )
    for name, parameter in moe.flat_parameter_dict().items():
        if name.endswith('weight') or name in {'w1', 'w2', 'w3'}:
            parameter.value = qwix.quantize(
                parameter.value,
                'int4',
                channelwise_axes=(parameter.value.ndim - 1,),
            )

    output = moe(jnp.ones((1, 2, 8), dtype=jnp.bfloat16))

    assert output.shape == (1, 2, 8)
    assert jnp.all(jnp.isfinite(output))
