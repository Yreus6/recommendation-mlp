from typing import Optional, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F  # noqa: F401
import torchmetrics  # noqa: F401
from torch import optim, nn  # noqa: F401

from recommendationlab.core.embed import UserEmbedding, ItemEmbedding


class MLP(pl.LightningModule):
    """
    MLP Model
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
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
        self.layers.append(nn.Linear(layers[-1], 1))
        self.optimizer = getattr(optim, optimizer)
        self.lr = lr
        self.top_k = top_k
        self.user_embedding = UserEmbedding(num_users, int(layers[0] / 2))
        self.item_embedding = ItemEmbedding(num_items, int(layers[0] / 2))
    
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
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        
        return optimizer
    
    def load_weights(self, path: str):
        self.load_state_dict(state_dict=torch.load(path))
    
    def _common_step(self, batch, stage):
        user_ids, item_ids, labels = batch
        user_ids = self.user_embedding(user_ids).flatten(start_dim=1).float()
        item_ids = self.item_embedding(item_ids).flatten(start_dim=1).float()
        x, y = torch.concat([user_ids, item_ids], dim=-1), labels.float().reshape(-1, 1)
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        
        if stage == 'training':
            self.log(f'{stage}-loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            
            return loss
        if stage in ['val', 'test']:
            hit_rate = torchmetrics.functional.retrieval_hit_rate(y_hat, y, top_k=self.top_k)
            ndcg = torchmetrics.functional.retrieval_normalized_dcg(y_hat, y, top_k=self.top_k)
            self.log(f'{stage}-HR', hit_rate, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log(f'{stage}-NDCG', ndcg, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log(f'{stage}-loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
