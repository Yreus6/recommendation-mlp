import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.recommendationlab.components.embed import UserItemToId


def separate_col(df: pd.DataFrame, col: str):
    df[col] = df[col].str.split('|')
    df = df.explode(col)
    df = df.reset_index(drop=True)

    return df


def build_user_item_matrix(df: pd.DataFrame, user_item: UserItemToId):
    # users = df[['USER_ID', 'GENRES_USER', 'INSTRUMENTS', 'COUNTRY', 'AGE']].drop_duplicates()
    # users = users.groupby(['USER_ID'], as_index=False).aggregate({
    #     'GENRES_USER': 'sum',
    #     'INSTRUMENTS': 'sum',
    #     'COUNTRY': 'mean',
    #     'AGE': 'mean'
    # }).values
    # items = df[['ITEM_ID', 'GENRES_ITEM', 'GENRE_L2', 'GENRE_L3', 'CREATION_TIMESTAMP']].drop_duplicates()
    # items = items.groupby('ITEM_ID', as_index=False).aggregate({
    #     'GENRES_ITEM': 'sum',
    #     'GENRE_L2': 'sum',
    #     'GENRE_L3': 'mean',
    #     'CREATION_TIMESTAMP': 'mean'
    # }).values
    interactions = df[['USER_ID', 'ITEM_ID', 'EVENT_VALUE', 'TIMESTAMP']].drop_duplicates().values

    mat = sp.dok_matrix((len(user_item.user2id.keys()), len(user_item.item2id.keys())), dtype=np.float32)

    for interaction in interactions:
        user_id, item_id, event_value, _ = interaction
        if event_value == 1:
            mat[user_item.user2id[user_id], user_item.item2id[item_id]] = 1

    return mat
