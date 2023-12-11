import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset


class VAMPR(Dataset):
    def __init__(
        self,
        phase: str,
        matrix: sp.dok_matrix,
        num_items: int,
        num_negs: int,
        train_matrix: sp.dok_matrix = None,
        val_matrix: sp.dok_matrix = None,
    ):
        user_input, item_input, labels = [], [], []
        for (u, i) in matrix.keys():
            # positive instance
            user_input.append(u)
            item_input.append(i)
            labels.append(1.)
            # negative instances
            for t in range(num_negs):
                j = np.random.randint(num_items)
                if phase == 'train':
                    while (u, j) in matrix.keys():
                        j = np.random.randint(num_items)
                elif phase == 'val':
                    while (u, j) in matrix.keys() and (u, j) in train_matrix.keys():
                        j = np.random.randint(num_items)
                elif phase == 'test':
                    while (u, j) in matrix.keys() and (u, j) in train_matrix.keys() and (u, j) in val_matrix.keys():
                        j = np.random.randint(num_items)
                user_input.append(u)
                item_input.append(j)
                labels.append(0.)
        
        self.user_ids = np.array(user_input)
        self.item_ids = np.array(item_input)
        self.labels = np.array(labels)
    
    def __getitem__(self, index):
        return self.user_ids[index], self.item_ids[index], self.labels[index]
    
    def __len__(self):
        return len(self.user_ids)
