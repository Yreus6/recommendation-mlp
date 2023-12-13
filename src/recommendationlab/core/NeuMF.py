from typing import Optional, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F  # noqa: F401
import torchmetrics  # noqa: F401
from torch import optim, nn  # noqa: F401

from src.recommendationlab.components.utils import calculate_metrics
from src.recommendationlab.core.GMF import GMF
from src.recommendationlab.core.MLP import MLP


class NeuMF(pl.LightningModule):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        layers: Optional[List[int]] = [20, 10],
        gmf_factor: int = 10,
        gmf_pretrain: str = '',
        mlp_pretrain: str = '',
        alpha: float = 0.5,
        optimizer: str = 'Adam',
        lr: float = 1e-3,
        top_k: int = 10,
        dropout: float = 0.5
    ):
        super().__init__()
        self.alpha = alpha
        self.optimizer = getattr(optim, optimizer)
        self.lr = lr
        self.top_k = top_k
        self.dropout = nn.Dropout(dropout)
        self.gmf_user_embedding = nn.Embedding(num_users, gmf_factor)
        self.gmf_item_embedding = nn.Embedding(num_items, gmf_factor)
        self.mlp_user_embedding = nn.Embedding(num_users, int(layers[0] / 2))
        self.mlp_item_embedding = nn.Embedding(num_items, int(layers[0] / 2))
        self.gmf_pretrain = gmf_pretrain
        self.mlp_pretrain = mlp_pretrain
        
        num_layers = len(layers)
        self.layers = nn.ModuleList()
        for i in range(num_layers - 1):
            linear = nn.Linear(layers[i], layers[i + 1])
            self.layers.append(linear)
        self.predict_layer = nn.Linear(layers[-1] + gmf_factor, 1)
        
        if gmf_pretrain != '' and mlp_pretrain != '':
            self.gmf_model = GMF.load_from_checkpoint(gmf_pretrain)
            self.mlp_model = MLP.load_from_checkpoint(mlp_pretrain)
            print(f'Loaded pretrained model GMF: {gmf_pretrain}, MLP: {mlp_pretrain}')
        
        self.reset_parameters()
        self.save_hyperparameters()
    
    def reset_parameters(self):
        if self.gmf_pretrain != '' and self.mlp_pretrain != '':
            self.gmf_user_embedding.weight.copy_(self.gmf_model.user_embedding.weight)
            self.gmf_item_embedding.weight.copy_(self.gmf_model.item_embedding.weight)
            self.mlp_user_embedding.weight.copy_(self.mlp_model.user_embedding.weight)
            self.mlp_item_embedding.weight.copy_(self.mlp_model.item_embedding.weight)
            for (m1, m2) in zip(self.layers, self.mlp_model.layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.copy_(m2.weight)
                    m1.bias.copy_(m2.bias)
            
            predict_weight = torch.cat([
                self.gmf_model.predict_layer.weight,
                self.mlp_model.predict_layer.weight
            ], dim=1)
            predict_bias = self.gmf_model.predict_layer.bias + self.mlp_model.predict_layer.bias
            self.predict_layer.weight.copy_(0.5 * predict_weight)
            self.predict_layer.bias.copy_(0.5 * predict_bias)
        else:
            nn.init.normal_(self.gmf_user_embedding.weight, std=0.01)
            nn.init.normal_(self.gmf_item_embedding.weight, std=0.01)
            nn.init.normal_(self.mlp_user_embedding.weight, std=0.01)
            nn.init.normal_(self.mlp_item_embedding.weight, std=0.01)
    
    def forward(self, x):
        users, items = x
        gmf_users = self.gmf_user_embedding(users).float()
        gmf_items = self.gmf_item_embedding(items).float()
        mlp_users = self.mlp_user_embedding(users).float()
        mlp_items = self.mlp_item_embedding(items).float()
        x_gmf = gmf_users * gmf_items
        x_mlp = torch.concat([mlp_users, mlp_items], dim=-1)
        
        for i, layer in enumerate(self.layers):
            x_mlp = self.dropout(x_mlp)
            x_mlp = layer(x_mlp)
            x_mlp = F.relu(x_mlp)
        
        x_gmf = self.alpha * x_gmf
        x_mlp = (1 - self.alpha) * x_mlp
        out = torch.concat([x_gmf, x_mlp], dim=-1)
        out = self.predict_layer(out)
        out = F.sigmoid(out)
        
        return out
    
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
