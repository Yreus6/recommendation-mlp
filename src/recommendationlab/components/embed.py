from typing import List

import torch
from torch import nn


class FeaturesEmbedding(nn.Module):
    def __init__(self, field_embeds: List[int], embed_sizes: List[int]):
        super().__init__()
        self.embeddings = nn.ModuleList()
        for i, (field_size, embed_size) in enumerate(zip(field_embeds, embed_sizes)):
            self.embeddings.append(nn.Embedding(field_size + 1, embed_size, padding_idx=0))

    def forward(self, x):
        embedded = torch.zeros((x.shape[0], 0), dtype=torch.float, device=x.device)

        for i, embedding in enumerate(self.embeddings):
            embedded = torch.concat([embedded, embedding(x[:, i])], dim=-1)

        return embedded
