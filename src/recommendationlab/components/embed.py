from collections import OrderedDict

import numpy as np
from torch import nn


class FeaturesEmbedding(nn.Module):
    def __init__(self, field_sizes: list, embed_size: int):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_sizes), embed_size)
        self.offsets = np.array((0, *np.cumsum(field_sizes)[:-1]), dtype=np.int64)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)

        return self.embedding(x)


class UserItemToId:
    def __init__(self, user_data, item_data):
        self.user_data = user_data
        self.item_data = item_data
        self._init_data()
        self.id2user = {self.user2id[k]: k for k in self.user2id}
        self.id2item = {self.item2id[k]: k for k in self.item2id}

    def _init_data(self):
        self.user2id, self.item2id = OrderedDict(), OrderedDict()
        for user in self.user_data:
            self.user2id[user] = len(self.user2id)
        for item in self.item_data:
            self.item2id[item] = len(self.item2id)

    @property
    def users(self):
        return self.id2user.keys()

    @property
    def items(self):
        return self.id2item.keys()
