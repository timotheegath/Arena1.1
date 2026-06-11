from collections import defaultdict

import torch as t
from torch import Tensor
from tqdm import tqdm
from transformer_lens import HookedTransformer

import tests
from custom_transformer import Config, DemoTransformer, TransformerSampler
from training import TransformerTrainer, TransformerTrainingArgs

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)
# Training model config
""" model_cfg = Config(
    debug=False,
    d_model=32,
    n_heads=16,
    d_head=2,
    d_mlp=32 * 4,
    n_layers=4,
    n_ctx=128,
    # d_vocab will be taken from the ref model automatically
) """
model = DemoTransformer(Config()).to(device)
model.load_pretrained_weights_from_reference()
sampler = TransformerSampler(model, model.tokenizer) # type: ignore

tests.test_apply_temperature(TransformerSampler.apply_temperature)

logits = t.tensor([1, 2]).log()

cold_logits = TransformerSampler.apply_temperature(logits, temperature=0.001)
print('A low temperature "sharpens" or "peaks" the distribution: ', cold_logits)
t.testing.assert_close(cold_logits, 1000.0 * logits)

hot_logits = TransformerSampler.apply_temperature(logits, temperature=1000.0)
print("A high temperature flattens the distribution: ", hot_logits)
t.testing.assert_close(hot_logits, 0.001 * logits)

print("Tests passed!")


