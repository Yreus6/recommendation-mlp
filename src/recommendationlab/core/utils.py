import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def separate_col(df: pd.DataFrame, col: str):
    df[col] = df[col].str.split('|')
    df = df.explode(col)
    df = df.reset_index(drop=True)

    return df


def get_users_items_mat(df: pd.DataFrame):
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
    interactions = df[['USER_ID', 'ITEM_ID', 'EVENT_VALUE']].drop_duplicates().values

    user_ids = interactions[:, 0]
    item_ids = interactions[:, 1]
    mat = sp.dok_matrix((len(user_ids) + 1, len(item_ids) + 1), dtype=np.float32)

    for interaction in interactions:
        user_id, item_id, event_value = interaction
        if event_value == 1:
            mat[user_id, item_id] = 1

    return mat


def normalize_label_col(df: pd.DataFrame, col: str):
    le = LabelEncoder()
    le.fit(df[col].unique())
    df[col] = le.transform(df[col].values)

    return df


def normalize_min_max(df: pd.DataFrame, col: str):
    scaler = MinMaxScaler()
    df[col] = scaler.fit(df[col].unique().reshape(-1, 1))

    return df
