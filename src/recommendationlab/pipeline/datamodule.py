import os
import pytorch_lightning as L
import pandas as pd
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader

from src.recommendationlab import config
from src.recommendationlab.components.vocab import Vocab
from src.recommendationlab.pipeline.dataset import VAMPR, VAMPRPredict
from src.recommendationlab.components.utils import build_user_item_matrix


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
        self.train = pd.read_csv(os.path.join(config.SPLITSPATH, 'train.csv'))
        user_ids = self.train['USER_ID'].unique()
        item_ids = self.train['ITEM_ID'].unique()
        self.num_users = len(user_ids)
        self.num_items = len(item_ids)
        self.user_ids = Vocab(user_ids)
        self.item_ids = Vocab(item_ids)

        if stage == 'fit':
            self.val = pd.read_csv(os.path.join(config.SPLITSPATH, 'val.csv'))
        if stage == 'test':
            self.test = pd.read_csv(os.path.join(config.SPLITSPATH, 'test.csv'))
        if stage == 'predict':
            users, items = self.predict_data
            self.predict = VAMPRPredict(users.values, items.values)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_mat = build_user_item_matrix(self.train, self.user_ids, self.item_ids)
        dataset = VAMPR(train_mat, self.num_negs)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_mat = build_user_item_matrix(self.val, self.user_ids, self.item_ids)
        dataset = VAMPR(val_mat, self.num_negs_val)

        return DataLoader(
            dataset,
            batch_size=self.num_negs_val + 1,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        test_mat = build_user_item_matrix(self.test, self.user_ids, self.item_ids)
        dataset = VAMPR(test_mat, self.num_negs_test)

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
