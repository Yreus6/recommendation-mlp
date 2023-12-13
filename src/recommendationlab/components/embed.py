from typing import List, Tuple

import torch
from torch import nn


class FeaturesEmbedding(nn.Module):
    def __init__(self, fields: List[Tuple[int, int]]):
        super().__init__()
        self.embeddings = nn.ModuleList()
        for field_size, embed_size in fields:
            self.embeddings.append(nn.Embedding(field_size, embed_size))

    def forward(self, x):
        embedded = torch.zeros((x.shape[0], 0), dtype=torch.float)

        for i, embedding in enumerate(self.embeddings):
            embedded = torch.concat([embedded, embedding(x[:, i])])

        return embedded
