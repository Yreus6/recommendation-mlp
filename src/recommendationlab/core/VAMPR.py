import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

from src.recommendationlab.core.embed import UserEmbedding, ItemEmbedding


class VAMPR(Dataset):
    def __init__(
        self,
        users: np.ndarray,
        items: np.ndarray,
        matrix: sp.dok_matrix,
        num_negatives: int,
        embed_size: int
    ):
        num_users = len(users)
        num_items = len(items)

        users = torch.from_numpy(users).long()
        items = torch.from_numpy(items).long()

        user_ids = users[:, 0]
        item_ids = items[:, 0]

        embed_users = UserEmbedding(num_users, embed_size)(users)
        embed_items = ItemEmbedding(num_items, embed_size)(items)

        self.users_map = {}
        self.items_map = {}

        for i, user_id in enumerate(user_ids):
            self.users_map[user_id] = embed_users[i]
        for i, item_id in enumerate(item_ids):
            self.items_map[item_id] = embed_items[i]

        user_input, item_input, labels = [], [], []
        for (u, i) in matrix.keys():
            # positive instance
            user_input.append(u)
            item_input.append(i)
            labels.append(1)
            # negative instances
            for t in range(num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in matrix:
                    j = np.random.randint(num_items)
                user_input.append(u)
                item_input.append(j)
                labels.append(0)

        self.user_ids = user_input
        self.item_ids = item_input
        self.labels = torch.from_numpy(np.array(labels, dtype=np.float32))

    def __getitem__(self, index):
        merged_vector = torch.concat(
            [self.users_map[self.user_ids[index]], self.items_map[self.item_ids[index]]],
            dim=-1
        )

        return merged_vector, self.labels[index]

    def __len__(self):
        return len(self.user_ids)
