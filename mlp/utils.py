from pathlib import Path

import torch
from torch import Tensor


def make_dataset(file: Path, context_window: int = 3) -> tuple[Tensor, Tensor, dict]:
    words = open(file, "r").read().split()

    chars = sorted(list(set("".join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0
    itos = {i: s for s, i in stoi.items()}

    context_window = 3
    X, Y = [], []
    for word in words:
        # print(word)
        context = [0] * context_window

        for char in word + ".":
            ix = stoi[char]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X, Y, itos
