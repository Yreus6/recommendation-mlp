from typing import Optional, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F  # noqa: F401
import torchmetrics  # noqa: F401
from torch import optim, nn  # noqa: F401

from src.recommendationlab.core.GMF import GMF
from src.recommendationlab.core.MLP import MLP


class NeuMF(pl.LightningModule):
    def __init__(
        self,
        layers: Optional[List[int]] = [20, 10],
        gmf_pretrain: str = '',
        mlp_pretrain: str = '',
        alpha: float = 0.5
    ):
        super().__init__()
        self.gmf_model = GMF()
        self.mlp_model = MLP(layers)
        self.alpha = alpha

        if gmf_pretrain != '' and mlp_pretrain != '':
            self.gmf_model.load_weights(gmf_pretrain)
            self.mlp_model.load_weights(mlp_pretrain)
            print(f'Loaded pretrained model GMF: {gmf_pretrain}, MLP: {mlp_pretrain}')

    def forward(self, x):
        x_gmf, x_mlp = torch.detach(x).clone(), torch.detach(x).clone()
        x_gmf = self.gmf_model(x_gmf)
        x_mlp = self.mlp_model(x_mlp)

        x_gmf = self.alpha * x_gmf
        x_mlp = (1 - self.alpha) * x_mlp
        out = torch.concat([x_gmf, x_mlp], dim=-1)

        return out

    def training_step(self, batch):
        return self._common_step(batch, 'training')

    def test_step(self, batch, *args):
        self._common_step(batch, 'test')

    def validation_step(self, batch, *args):
        self._common_step(batch, 'val')

    def _common_step(self, batch, stage):
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
