import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import datasets
import einops
import numpy as np
import torch as t
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
from transformers import GPT2TokenizerFast


device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda"
    if t.cuda.is_available()
    else "cpu"
)

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part1_transformer_from_scratch"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

# import part1_transformer_from_scratch.solutions as solutions
# import part1_transformer_from_scratch.tests as tests



def get_vocabulary(tokenizer):
    sorted_vocab = sorted(list(tokenizer.vocab.items()), key=lambda n: n[1])
    return sorted_vocab

def tokenize_sentence(tokenizer, sentence):
    tokens = tokenizer.encode(sentence)
    return tokens

def evaluate_tokens(model, input_tokens):
    input_ids = t.tensor(input_tokens).unsqueeze(0).to(device)
    with t.no_grad():
        outputs, cache = model.run_with_cache(input_ids)
    logits = outputs
    #probabilities = t.softmax(logits, dim=-1)
    return logits, cache

def plot_token_probabilities(input_tokens, logits, tokenizer, input_token_n=0):
    # Set input_token_n to the index of the input token you want to analyze the predictions for (0 for the first token, 1 for the second, etc.)
    probabilities = t.softmax(logits[0, input_token_n], dim=-1) # Column vector of probabilities for the next token, size of vocab
    top_k = 10 # Set to retrieve the top ... predictions only
    top_k_probs, top_k_indices = t.topk(probabilities, k=top_k, dim=-1) # Retrieve the top k probabilities and their corresponding token indices, vector of length top_k
    top_k_tokens = [tokenizer.decode([idx]) for idx in top_k_indices]
    y = np.arange(len(top_k_tokens))

    fig, ax = plt.subplots(figsize=(10, 6))
    top_k_probs_np = top_k_probs.cpu().numpy()
    ax.barh(y, top_k_probs_np)
    ax.set_yticks(y)
    ax.set_yticklabels(top_k_tokens)
    ax.set_xlabel("Probability")
    input_word = tokenizer.decode([input_tokens[input_token_n]])
    ax.set_title("Top 10 Predicted Tokens for the input token '{}'".format(input_word))
    ax.invert_yaxis()

    # Annotate the actual probability values on each bar
    for prob, y_pos in zip(top_k_probs_np, y):
        ax.annotate(f"{prob:.3f}", xy=(prob, y_pos), xytext=(5, 0), textcoords="offset points", va="center")
    plt.show()

def get_model_config(model):
    return model.cfg

MAIN = __name__ == "__main__"
if MAIN:
    reference_gpt2 = HookedTransformer.from_pretrained(
        "gpt2-small",
        fold_ln=False,
        center_unembed=False,
        center_writing_weights=False,  # you'll learn about these arguments later!
    )

    tokenizer = reference_gpt2.tokenizer
    if tokenizer is None:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    sorted_vocab = get_vocabulary(tokenizer)
    sentence = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
    input_tokens = tokenize_sentence(tokenizer, sentence)
    logits, cache = evaluate_tokens(reference_gpt2, input_tokens)
    print(logits.shape)
    most_likely_next_tokens = tokenizer.batch_decode(logits.argmax(dim=-1)[0])

    print(most_likely_next_tokens)
    # plot_token_probabilities(input_tokens,logits, tokenizer, input_token_n=5)
    for activation_name, activation in cache.items():
    # Only print for first layer
        if ".0." in activation_name or "blocks" not in activation_name:
            print(f"{activation_name:30} {tuple(activation.shape)}")


