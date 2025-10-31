import torch
import torch.nn as nn


class CharacterLevelMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.C = nn.Parameter(torch.randn((27, 2)))
        self.W1 = nn.Parameter(torch.randn((6, 100)))
        self.b1 = nn.Parameter(torch.randn(100))
        self.W2 = nn.Parameter(torch.randn((100, 27)))
        self.b2 = nn.Parameter(torch.randn(27))

    def forward(self, x):
        embeddings = self.C[x]
        hidden_dim = embeddings.view((embeddings.shape[0], 6)) @ self.W1 + self.b1
        logits = hidden_dim @ self.W2 + self.b2
        return logits
