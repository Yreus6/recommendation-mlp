import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F  # noqa: F401
import torchmetrics  # noqa: F401
from torch import optim, nn  # noqa: F401


class GMF(pl.LightningModule):
    """
    GMF Model
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_size: int,
        optimizer: str = 'Adam',
        lr: float = 1e-3,
        top_k: int = 10,
    ):
        super().__init__()
        self.optimizer = getattr(optim, optimizer)
        self.predict_layer = nn.Linear(embed_size, 1)
        self.lr = lr
        self.top_k = top_k
        self.user_embedding = nn.Embedding(num_users, embed_size)
        self.item_embedding = nn.Embedding(num_items, embed_size)
        self.reset_parameters()
        self.save_hyperparameters()

    def reset_parameters(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, x):
        x = self.predict_layer(x)
        x = F.sigmoid(x)

        return x

    def training_step(self, batch):
        return self._common_step(batch, 'training')

    def test_step(self, batch, *args):
        self._common_step(batch, 'test')

    def validation_step(self, batch, *args):
        self._common_step(batch, 'val')

    def predict_step(self, batch, *args):
        user_ids, item_ids = batch
        user_ids = self.user_embedding(user_ids).float()
        item_ids = self.item_embedding(item_ids).float()
        x = user_ids * item_ids
        y_hat = self(x).reshape(-1)

        return y_hat

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, mode='max')

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val-HR'
        }

    def load_weights(self, path: str):
        self.load_state_dict(state_dict=torch.load(path))

    def _common_step(self, batch, stage):
        user_ids, item_ids, labels = batch
        user_ids = self.user_embedding(user_ids).float()
        item_ids = self.item_embedding(item_ids).float()
        x, y = user_ids * item_ids, labels.float()
        y_hat = self(x).reshape(-1)

        loss = F.binary_cross_entropy(y_hat, y)
        self.log(f'{stage}-loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        if stage == 'training':
            return loss
        if stage in ['val', 'test']:
            rank = torch.sum((y_hat >= y_hat[0]).float()).item()
            if rank <= self.top_k:
                hit_rate = 1.0
                ndcg = 1 / np.log2(rank + 1)
            else:
                hit_rate = 0.0
                ndcg = 0.0
            self.log(f'{stage}-HR', hit_rate, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log(f'{stage}-NDCG', ndcg, prog_bar=True, logger=True, on_step=True, on_epoch=True)
