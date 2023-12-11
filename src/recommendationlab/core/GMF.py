import pytorch_lightning as pl
import torch
import torch.nn.functional as F  # noqa: F401
import torchmetrics  # noqa: F401
from torch import optim, nn  # noqa: F401

from recommendationlab.core.embed import UserEmbedding, ItemEmbedding


class GMF(pl.LightningModule):
    """
    GMF Model
    """

    def __init__(
        self,
        embed_size: int,
        num_users: int,
        num_items: int,
        optimizer: str = 'Adam',
        lr: float = 1e-3,
        top_k: int = 10,
    ):
        super().__init__()
        self.optimizer = getattr(optim, optimizer)
        self.linear = nn.Linear(embed_size, 1)
        self.lr = lr
        self.top_k = top_k
        self.user_embedding = UserEmbedding(num_users, embed_size)
        self.item_embedding = ItemEmbedding(num_items, embed_size)

    def forward(self, x):
        x = self.linear(x)
        x = F.sigmoid(x)

        return x

    def training_step(self, batch):
        return self._common_step(batch, 'training')

    def test_step(self, batch, *args):
        self._common_step(batch, 'test')

    def validation_step(self, batch, *args):
        self._common_step(batch, 'val')

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)

        return optimizer

    def load_weights(self, path: str):
        self.load_state_dict(state_dict=torch.load(path))

    def _common_step(self, batch, stage):
        user_ids, item_ids, labels = batch
        user_ids = self.user_embedding(user_ids).flatten(start_dim=1).float()
        item_ids = self.item_embedding(item_ids).flatten(start_dim=1).float()
        x, y = user_ids * item_ids, labels.reshape(-1, 1)
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)

        if stage == 'training':
            self.log(f'{stage}-loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

            return loss
        if stage in ['val', 'test']:
            hit_rate = torchmetrics.functional.retrieval_hit_rate(y_hat, y, self.top_k)
            ndcg = torchmetrics.functional.retrieval_normalized_dcg(y_hat, y, self.top_k)
            self.log(f'{stage}-HR', hit_rate, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log(f'{stage}-NDCG', ndcg, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log(f'{stage}-loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
