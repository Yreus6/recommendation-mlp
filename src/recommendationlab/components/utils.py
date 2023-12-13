import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from src.recommendationlab.components.vocab import Vocab


def users_normalize(users, stretch_size=32):
    user_vocab = Vocab.load_vocab('user_vocab.json')

    res = {}
    for user in users:
        genres = user[1].split('|')
        instruments = user[2].split('|')
        country = user[3]
        age = user[4]

        genres_vocab = user_vocab['genres']
        instruments_vocab = user_vocab['instruments']
        countries_vocab = user_vocab['countries']

        genres = [genres_vocab[genre] for genre in genres]
        instruments = [instruments_vocab[instrument] for instrument in instruments]
        country = countries_vocab[country]

        while len(genres) < stretch_size:
            genres.append('UNK')

        while len(instruments) < stretch_size:
            instruments.append('UNK')

        user[1] = genres
        user[2] = instruments
        user[3] = country
        user[4] = age / 100

        res[user[0]] = user

    return res


def items_normalize(items, stretch_size=32):
    item_vocab = Vocab.load_vocab('item_vocab.json')

    res = {}
    for item in items:
        genres = item[1]
        genre_l2 = item[2].split('|')
        genre_l3 = item[3]
        creation_timestamp = item[4]

        genres_vocab = item_vocab['genres']
        genre_l2_vocab = item_vocab['genre_l2']
        genre_l3_vocab = item_vocab['genre_l3']

        genres = genres_vocab[genres]
        genre_l2 = [genre_l2_vocab[g] for g in genre_l2]
        genre_l3 = genre_l3_vocab[genre_l3]

        while len(genre_l2) < stretch_size:
            genre_l2.append('UNK')

        item[1] = genres
        item[2] = genre_l2
        item[3] = genre_l3
        item[4] = creation_timestamp / 5000000000

        res[item[0]] = item

    return res


def build_user_item_matrix(interactions: pd.DataFrame, user_ids: Vocab, item_ids: Vocab):
    interactions = interactions[['USER_ID', 'ITEM_ID', 'EVENT_VALUE', 'TIMESTAMP']].drop_duplicates().values

    mat = sp.dok_matrix((len(user_ids.item2id.keys()), len(item_ids.item2id.keys())), dtype=np.float32)

    for interaction in interactions:
        user_id, item_id, event_value, _ = interaction
        if event_value == 1:
            mat[user_ids.item2id[user_id], item_ids.item2id[item_id]] = 1

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
