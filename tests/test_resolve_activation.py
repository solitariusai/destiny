from taktiny.nn._continuo import _resolve_activation

act1 = _resolve_activation('gelu_pytorch_tanh')
act2 = _resolve_activation('gelu_pytorch_tanh')

print("act1:", act1)
print("act2:", act2)
print("Eq:", act1 == act2)
