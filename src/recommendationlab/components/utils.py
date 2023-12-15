import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from src.recommendationlab.components.vocab import Vocab


def users_normalize(user_df: pd.DataFrame, user_id_vocab: Vocab, user_vocab):
    genres_vocab = user_vocab['genres']
    instruments_vocab = user_vocab['instruments']
    countries_vocab = user_vocab['countries']
    
    user_df['USER_ID'] = user_df['USER_ID'].apply(lambda x: user_id_vocab.item2id.get(x, 0))
    user_df['GENRES'] = (
        user_df['GENRES'].apply(lambda x: genres_vocab[x] if x in genres_vocab else 0))
    user_df['INSTRUMENTS'] = (
        user_df['INSTRUMENTS'].apply(lambda x: instruments_vocab[x] if x in instruments_vocab else 0))
    user_df['COUNTRY'] = (
        user_df['COUNTRY'].apply(lambda x: countries_vocab[x] if x in countries_vocab else 0))

    return user_df.set_index('USER_ID', drop=False).T.to_dict('list')


def items_normalize(item_df: pd.DataFrame, item_id_vocab: Vocab, item_vocab):
    genres_vocab = item_vocab['genres']
    genre_l2_vocab = item_vocab['genre_l2']
    genre_l3_vocab = item_vocab['genre_l3']
    
    item_df['ITEM_ID'] = item_df['ITEM_ID'].apply(lambda x: item_id_vocab.item2id.get(x, 0))
    item_df['GENRES'] = (
        item_df['GENRES'].apply(lambda x: genres_vocab[x] if x in genres_vocab else 0))
    item_df['GENRE_L2'] = (
        item_df['GENRE_L2'].apply(lambda x: genre_l2_vocab[x] if x in genre_l2_vocab else 0))
    item_df['GENRE_L3'] = (
        item_df['GENRE_L3'].apply(lambda x: genre_l3_vocab[x] if x in genre_l3_vocab else 0))

    return item_df.set_index('ITEM_ID', drop=False).T.to_dict('list')


def build_user_item_matrix(interactions: pd.DataFrame, user_id_vocab: Vocab, item_id_vocab: Vocab):
    interactions = interactions[['USER_ID', 'ITEM_ID', 'EVENT_VALUE', 'TIMESTAMP']].drop_duplicates().values
    
    mat = sp.dok_matrix((len(user_id_vocab.item2id.keys()), len(item_id_vocab.item2id.keys())), dtype=np.float32)
    
    for interaction in interactions:
        user_id, item_id, event_value, _ = interaction
        if event_value == 1:
            mat[user_id_vocab.item2id[user_id], item_id_vocab.item2id[item_id]] = 1
    
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
