import torch
import torch.nn as nn


class CharacterLevelMLP(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embedding: int,
        hidden_dim: int,
        context_window: int = 3,
    ):
        super().__init__()
        self.C = nn.Parameter(torch.randn((vocab_size, n_embedding)))
        self.W1 = nn.Parameter(torch.randn((n_embedding * context_window, hidden_dim)))
        self.b1 = nn.Parameter(torch.randn(hidden_dim))
        self.W2 = nn.Parameter(torch.randn((hidden_dim, vocab_size)))
        self.b2 = nn.Parameter(torch.randn(vocab_size))

    def forward(self, x):
        embeddings = self.C[x]
        hidden_dim = embeddings.view((embeddings.shape[0], -1)) @ self.W1 + self.b1
        logits = hidden_dim @ self.W2 + self.b2
        return logits
