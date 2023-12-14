from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F  # noqa: F401
import torchmetrics  # noqa: F401
from torch import optim, nn  # noqa: F401

from src.recommendationlab.components.embed import FeaturesEmbedding
from src.recommendationlab.components.utils import calculate_metrics


class GMF(pl.LightningModule):
    """
    GMF Model
    """
    
    def __init__(
        self,
        users_fields: List[int],
        items_fields: List[int],
        user_embed_sizes: List[int],
        item_embed_sizes: List[int],
        optimizer: str = 'Adam',
        lr: float = 1e-3,
        top_k: int = 10,
    ):
        super().__init__()
        self.lr = lr
        self.top_k = top_k
        self.optimizer = getattr(optim, optimizer)
        assert sum(user_embed_sizes) == sum(item_embed_sizes)
        embed_size = sum(user_embed_sizes) + 1
        self.predict_layer = nn.Linear(embed_size, 1)
        
        self.user_embedding = FeaturesEmbedding(users_fields, user_embed_sizes)
        self.item_embedding = FeaturesEmbedding(items_fields, item_embed_sizes)
        
        self.reset_parameters()
        self.save_hyperparameters()
    
    def reset_parameters(self):
        for embedding in self.user_embedding.embeddings:
            nn.init.normal_(embedding.weight, std=0.01)
        for embedding in self.item_embedding.embeddings:
            nn.init.normal_(embedding.weight, std=0.01)
    
    def forward(self, x):
        users, items = x
        users = torch.concat([
            self.user_embedding(users[:, :-1]).float(),
            users[:, -1:].float() / 100
        ], dim=-1)
        items = torch.concat([
            self.item_embedding(items[:, :-1]).float(),
            items[:, -1:].float() / 5000000000
        ], dim=-1)
        x = users * items
        
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
        y_hat = self(batch).reshape(-1)
        
        return y_hat
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, mode='max')
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val-HR'
        }
    
    def _common_step(self, batch, stage):
        users, items, labels = batch
        x = (users, items)
        y = labels.float()
        y_hat = self(x).reshape(-1)
        
        loss = F.binary_cross_entropy(y_hat, y)
        self.log(f'{stage}-loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        if stage == 'training':
            return loss
        if stage in ['val', 'test']:
            hit_rate, ndcg = calculate_metrics(y_hat, self.top_k)
            self.log(f'{stage}-HR', hit_rate, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log(f'{stage}-NDCG', ndcg, prog_bar=True, logger=True, on_step=True, on_epoch=True)
