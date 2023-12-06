import os
from typing import Any

import numpy as np
import pytorch_lightning as L
import pandas as pd
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from src.recommendationlab.core.VAMPR import VAMPR
from src.recommendationlab.core.embed import UserEmbedding, ItemEmbedding
from src.recommendationlab.core.utils import separate_col, get_users_items_mat, normalize_label_col, normalize_min_max


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        model: str,
        data_dir: str,
        embed_size: int,
        batch_size: int = 8,
        num_workers: int = 8,
        num_negatives: int = 4,
    ):
        super().__init__()
        self.model = model
        self.data_dir = data_dir
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_negatives = num_negatives

    def setup(self, stage: str) -> None:
        user_path = os.path.join(self.data_dir, 'users_dataset.csv')
        item_path = os.path.join(self.data_dir, 'items_dataset.csv')
        interaction_path = os.path.join(self.data_dir, 'interactions_dataset.csv')

        user_df = pd.read_csv(user_path)
        item_df = pd.read_csv(item_path)
        interaction_df = pd.read_csv(interaction_path)

        # user_df = separate_col(user_df, 'GENRES')
        # user_df = separate_col(user_df, 'INSTRUMENTS')
        # user_df = normalize_label_col(user_df, 'GENRES')
        # user_df = normalize_label_col(user_df, 'INSTRUMENTS')
        # user_df = normalize_label_col(user_df, 'COUNTRY')
        #
        # item_df = separate_col(item_df, 'GENRE_L2')
        # item_df = normalize_label_col(item_df, 'GENRES')
        # item_df = normalize_label_col(item_df, 'GENRE_L2')
        # item_df = normalize_label_col(item_df, 'GENRE_L3')
        # item_df = normalize_min_max(item_df, 'CREATION_TIMESTAMP')

        # df = pd.merge(user_df, interaction_df, on='USER_ID')
        # df = pd.merge(df, item_df, on='ITEM_ID', suffixes=('_USER', '_ITEM'))
        # interaction_df = normalize_label_col(interaction_df, 'USER_ID')
        # interaction_df = normalize_label_col(interaction_df, 'ITEM_ID')
        interaction_df = interaction_df.apply(LabelEncoder().fit_transform)

        self.user_embedding = UserEmbedding(interaction_df['USER_ID'].nunique(), self.embed_size)
        self.item_embedding = ItemEmbedding(interaction_df['ITEM_ID'].nunique(), self.embed_size)

        train, val, test = np.split(
            interaction_df.sample(frac=1, random_state=42),
            [int(.6 * len(interaction_df)), int(.8 * len(interaction_df))]
        )

        if stage == 'fit':
            self.train_mat = get_users_items_mat(train)
            self.val_mat = get_users_items_mat(val)
        elif stage == 'test':
            self.test_mat = get_users_items_mat(test)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = VAMPR(self.train_mat, self.num_negatives)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        dataset = VAMPR(self.val_mat, self.num_negatives)

        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        dataset = VAMPR(self.test_mat, self.num_negatives)

        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int):
        user_ids, item_ids, labels = batch
        user_ids = self.user_embedding(user_ids).flatten(start_dim=1).float()
        item_ids = self.item_embedding(item_ids).flatten(start_dim=1).float()

        if self.model == 'gmf':
            return user_ids * item_ids, labels.float().reshape(-1, 1)
        elif self.model == 'mlp':
            return torch.concat([user_ids, item_ids], dim=-1), labels.float().reshape(-1, 1)
        else:
            gmf_vector = user_ids * item_ids
            mlp_vector = torch.concat([user_ids, item_ids], dim=-1)

            return gmf_vector, mlp_vector, labels.float().reshape(-1, 1)
