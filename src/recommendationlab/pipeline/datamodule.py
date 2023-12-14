import os
import pytorch_lightning as L
import pandas as pd
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader

from src.recommendationlab import config
from src.recommendationlab.components.vocab import Vocab
from src.recommendationlab.pipeline.dataset import VAMPR, VAMPRPredict
from src.recommendationlab.components.utils import build_user_item_matrix, users_normalize, items_normalize


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 8,
        num_negs: int = 4,
        num_negs_val: int = 100,
        num_negs_test: int = 100,
        predict_data: tuple = None
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_negs = num_negs
        self.num_negs_val = num_negs_val
        self.num_negs_test = num_negs_test
        self.predict_data = predict_data
    
    def setup(self, stage: str) -> None:
        user_path = os.path.join(config.SPLITSPATH, 'users_dataset.csv')
        item_path = os.path.join(config.SPLITSPATH, 'items_dataset.csv')
        
        user_df = pd.read_csv(user_path)
        item_df = pd.read_csv(item_path)
        item_df.drop(['CONTENT_OWNER'], axis=1, inplace=True)
        
        user_df.fillna({
            'GENRES': 'UNK',
            'INSTRUMENTS': 'UNK',
            'COUNTRY': 'UNK',
            'AGE': 0
        }, inplace=True)
        item_df.fillna({
            'GENRES': 'UNK',
            'GENRE_L2': 'UNK',
            'GENRE_L3': 'UNK',
            'CREATION_TIMESTAMP': 0
        }, inplace=True)
        
        user_ids = user_df['USER_ID'].unique()
        item_ids = item_df['ITEM_ID'].unique()
        
        self.user_id_vocab = Vocab(user_ids, False)
        self.item_id_vocab = Vocab(item_ids, False)
        
        self.users = users_normalize(user_df, self.user_id_vocab)
        self.items = items_normalize(item_df, self.item_id_vocab)
        
        self.users_fields = [
            len(user_ids),
            user_df['GENRES'].nunique(),
            user_df['INSTRUMENTS'].nunique(),
            user_df['COUNTRY'].nunique()
        ]
        self.items_fields = [
            len(item_ids),
            item_df['GENRES'].nunique(),
            item_df['GENRE_L2'].nunique(),
            item_df['GENRE_L3'].nunique()
        ]
        
        if stage == 'fit':
            self.train = pd.read_csv(os.path.join(config.SPLITSPATH, 'train.csv'))
            self.val = pd.read_csv(os.path.join(config.SPLITSPATH, 'val.csv'))
        if stage == 'test':
            self.test = pd.read_csv(os.path.join(config.SPLITSPATH, 'test.csv'))
        if stage == 'predict':
            user_df, item_df = self.predict_data
            users = list(users_normalize(user_df, self.user_id_vocab).values())
            items = list(items_normalize(item_df, self.item_id_vocab).values())
            self.predict = VAMPRPredict(users, items)
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_mat = build_user_item_matrix(self.train, self.user_id_vocab, self.item_id_vocab)
        dataset = VAMPR(self.users, self.items, train_mat, self.num_negs)
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_mat = build_user_item_matrix(self.val, self.user_id_vocab, self.item_id_vocab)
        dataset = VAMPR(self.users, self.items, val_mat, self.num_negs_val)
        
        return DataLoader(
            dataset,
            batch_size=self.num_negs_val + 1,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        test_mat = build_user_item_matrix(self.test, self.user_id_vocab, self.item_id_vocab)
        dataset = VAMPR(self.users, self.items, test_mat, self.num_negs_test)
        
        return DataLoader(
            dataset,
            batch_size=self.num_negs_test + 1,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False
        )
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.predict,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False
        )
