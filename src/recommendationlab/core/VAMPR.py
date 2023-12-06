import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset


class VAMPR(Dataset):
    def __init__(self, matrix: sp.dok_matrix, num_negatives: int):
        num_items = matrix.shape[1]

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

        self.user_ids = np.array(user_input)
        self.item_ids = np.array(item_input)
        self.labels = np.array(labels)

    def __getitem__(self, index):
        return self.user_ids[index], self.item_ids[index], self.labels[index]

    def __len__(self):
        return len(self.user_ids)
