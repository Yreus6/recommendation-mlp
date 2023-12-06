from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import Logger, TensorBoardLogger
from pytorch_lightning.profilers import Profiler, PyTorchProfiler

from src.recommendationlab import config


class LabTrainer(pl.Trainer):
    def __init__(
        self,
        logger: Optional[Logger] = None,
        profiler: Optional[Profiler] = None,
        callbacks: Optional[List] = [],
        plugins: Optional[List] = [],
        set_seed: bool = True,
        **trainer_init_kwargs: Dict[str, Any]
    ) -> None:
        # SET SEED
        if set_seed:
            seed_everything(config.GLOBALSEED, workers=True)
        super().__init__(
            logger=logger or TensorBoardLogger(config.LOGSPATH, name='tensorboard'),
            profiler=profiler or PyTorchProfiler(dirpath=config.TORCHPROFILERPATH, filename='profiler'),
            callbacks=callbacks + [ModelCheckpoint(dirpath=config.CHKPTSPATH, filename='model')],
            plugins=plugins,
            **trainer_init_kwargs
        )

    def persist_predictions(self, predictions_dir: Optional[Union[str, Path]] = config.PREDSPATH) -> None:
        self.test(ckpt_path="best", datamodule=self.datamodule)
        predictions = self.predict(self.model, self.datamodule.val_dataloader())
        torch.save(predictions, predictions_dir)
