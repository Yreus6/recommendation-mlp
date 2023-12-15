import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset


class VAMPR(Dataset):
    def __init__(self, users: np.ndarray, items: np.ndarray, matrix: sp.dok_matrix, num_negs: int = 4):
        self.users = users
        self.items = items

        num_items = matrix.shape[1]
        user_input, item_input, labels = [], [], []

        for (u, i) in matrix.keys():
            # positive instance
            user_input.append(u)
            item_input.append(i)
            labels.append(1.)
            # negative instances
            for t in range(num_negs):
                j = np.random.randint(1, num_items)
                while (u, j) in matrix.keys():
                    j = np.random.randint(1, num_items)
                    if j in item_input:
                        j = np.random.randint(1, num_items)
                user_input.append(u)
                item_input.append(j)
                labels.append(0.)

        self.user_ids = np.array(user_input)
        self.item_ids = np.array(item_input)
        self.labels = np.array(labels)
    
    def __getitem__(self, index):
        user_id = self.user_ids[index]
        item_id = self.item_ids[index]
        label = self.labels[index]
        
        return np.array(self.users[user_id], dtype=int), np.array(self.items[item_id], dtype=int), label
    
    def __len__(self):
        return len(self.user_ids)


class VAMPRPredict(Dataset):
    def __init__(self, users, items, user_ids, item_ids):
        self.users = users
        self.items = items
        self.user_ids = user_ids
        self.item_ids = item_ids
    
    def __getitem__(self, index):
        user_id = self.user_ids[index]
        item_id = self.item_ids[index]

        return np.array(self.users[user_id], dtype=int), np.array(self.items[item_id], dtype=int)
    
    def __len__(self):
        return len(self.user_ids)
