import os
import numpy as np
import pytorch_lightning as L
import pandas as pd
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader

from recommendationlab import config
from src.recommendationlab.core.VAMPR import VAMPR
from src.recommendationlab.core.utils import build_user_item_matrix


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        model: str,
        data_dir: str,
        embed_size: int,
        batch_size: int = 8,
        num_workers: int = 8,
        num_negs: int = 4,
        num_negs_val: int = 100,
        num_negs_test: int = 100,
    ):
        super().__init__()
        self.model = model
        self.data_dir = data_dir
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_negs = num_negs
        self.num_negs_val = num_negs_val
        self.num_negs_test = num_negs_test
    
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
        interaction_df.sort_values(by=['USER_ID', 'TIMESTAMP'], ascending=True, inplace=True)
        
        train, val, test = np.split(
            interaction_df,
            [int(0.8 * len(interaction_df)), int(0.9 * len(interaction_df))]
        )
        
        val = val[val['USER_ID'].isin(val['USER_ID'].unique())]
        val = val[val['ITEM_ID'].isin(val['ITEM_ID'].unique())]
        test = test[test['USER_ID'].isin(train['USER_ID'].unique())]
        test = test[test['ITEM_ID'].isin(train['ITEM_ID'].unique())]
        test = test[test['USER_ID'].isin(val['USER_ID'].unique())]
        test = test[test['ITEM_ID'].isin(val['ITEM_ID'].unique())]
        
        val = val.groupby('USER_ID').last().reset_index()
        test = test.groupby('USER_ID').last().reset_index()
        
        train.to_csv(os.path.join(config.SPLITSPATH, 'train.csv'), index=False)
        val.to_csv(os.path.join(config.SPLITSPATH, 'val.csv'), index=False)
        test.to_csv(os.path.join(config.SPLITSPATH, 'test.csv'), index=False)
        
        self.num_users = train['USER_ID'].nunique()
        self.num_items = train['ITEM_ID'].nunique()
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train = pd.read_csv(os.path.join(config.SPLITSPATH, 'train.csv'))
        train_mat = build_user_item_matrix(train)
        dataset = VAMPR('train', train_mat, self.num_items, self.num_negs)
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        train = pd.read_csv(os.path.join(config.SPLITSPATH, 'train.csv'))
        train_mat = build_user_item_matrix(train)
        val = pd.read_csv(os.path.join(config.SPLITSPATH, 'val.csv'))
        val_mat = build_user_item_matrix(val)
        dataset = VAMPR('val', val_mat, self.num_items, self.num_negs_val, train_mat)
        
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        train = pd.read_csv(os.path.join(config.SPLITSPATH, 'train.csv'))
        train_mat = build_user_item_matrix(train)
        val = pd.read_csv(os.path.join(config.SPLITSPATH, 'val.csv'))
        val_mat = build_user_item_matrix(val)
        test = pd.read_csv(os.path.join(config.SPLITSPATH, 'test.csv'))
        test_mat = build_user_item_matrix(test)
        dataset = VAMPR('test', test_mat, self.num_items, self.num_negs_test, train_mat, val_mat)
        
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)
