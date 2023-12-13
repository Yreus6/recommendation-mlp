import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from src.recommendationlab.components.embed import UserItemToId


def build_user_item_matrix(interactions: pd.DataFrame, user_item: UserItemToId):
    interactions = interactions[['USER_ID', 'ITEM_ID', 'EVENT_VALUE', 'TIMESTAMP']].drop_duplicates().values
    
    mat = sp.dok_matrix((len(user_item.user2id.keys()), len(user_item.item2id.keys())), dtype=np.float32)
    
    for interaction in interactions:
        user_id, item_id, event_value, _ = interaction
        if event_value == 1:
            mat[user_item.user2id[user_id], user_item.item2id[item_id]] = 1
    
    return mat


def calculate_metrics(output, top_k):
    rank = torch.sum((output >= output[0]).float()).item()
    if rank <= top_k:
        hit_rate = 1.0
        ndcg = 1 / np.log2(rank + 1)
    else:
        hit_rate = 0.0
        ndcg = 0.0
    
    return hit_rate, ndcg
