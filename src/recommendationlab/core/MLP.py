from typing import Optional, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F  # noqa: F401
import torchmetrics  # noqa: F401
from torch import optim, nn  # noqa: F401


class MLP(pl.LightningModule):
    """
    MLP Model
    """

    def __init__(
        self,
        layers: Optional[List[int]] = [20, 10],
        optimizer: str = 'Adam',
        lr: float = 1e-3,
        top_k: int = 10
    ):
        super().__init__()
        num_layers = len(layers)
        self.layers = nn.ModuleList()
        for i in range(num_layers - 1):
            linear = nn.Linear(layers[i], layers[i + 1])
            self.layers.append(linear)

        self.optimizer = getattr(optim, optimizer)
        self.lr = lr
        self.top_k = top_k

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = F.relu(x)

        x = F.sigmoid(x)

        return x

    def training_step(self, batch):
        return self._common_step(batch, 'training')

    def test_step(self, batch, *args):
        self._common_step(batch, 'test')

    def validation_step(self, batch, *args):
        self._common_step(batch, 'val')

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch

        return self(x)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)

        return optimizer

    def load_weights(self, path: str):
        self.load_state_dict(state_dict=torch.load(path))

    def _common_step(self, batch, stage):
        """consolidates common code for train, test, and validation steps"""
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)

        if stage == 'training':
            self.log(f'{stage}-loss', loss)

            return loss
        if stage in ['val', 'test']:
            hit_rate = torchmetrics.functional.retrieval_hit_rate(y_hat, y)
            ndcg = torchmetrics.functional.retrieval_normalized_dcg(y_hat, y)
            self.log(f'{stage}-HR', hit_rate)
            self.log(f'{stage}-NDCG', ndcg)
            self.log(f'{stage}-loss', loss)
