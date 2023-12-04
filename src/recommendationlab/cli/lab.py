import os
from pathlib import Path
from typing import List

import typer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from typing_extensions import Annotated

from src.recommendationlab.core import LabTrainer
from src.recommendationlab.core.GMF import GMF
from src.recommendationlab.core.MLP import MLP
from src.recommendationlab.core.NeuMF import NeuMF
from src.recommendationlab.core.data_module import DataModule
import src.recommendationlab.config as Config

FILEPATH = Path(__file__)
PROJECTPATH = FILEPATH.parents[2]
PKGPATH = FILEPATH.parents[1]

app = typer.Typer()
docs_app = typer.Typer()
train_app = typer.Typer()
app.add_typer(docs_app, name="docs")


@app.callback()
def callback() -> None:
    pass


@docs_app.command("build")
def build_docs() -> None:
    import shutil

    os.system("mkdocs build")
    shutil.copyfile(src="README.md", dst="docs/index.md")


@docs_app.command("serve")
def serve_docs() -> None:
    os.system("mkdocs serve")


@train_app.command('Train model')
def train(
    data_dir: Annotated[str, typer.Option(help='Data directory')],
    model: Annotated[str, typer.Option(help='Model to use. One of (`mlp`, `gmf`, `neumf`)')],
    batch_size: Annotated[int, typer.Option(help='Batch size')] = 8,
    num_workers: Annotated[int, typer.Option(help='Number of workers')] = 4,
    num_neg: Annotated[int, typer.Option(help='Number of negative instances to pair with a positive instance.')] = 4,
    layers: Annotated[List[int], typer.Option(help='MLP layers')] = [20, 10],
    optimizer: Annotated[str, typer.Option(help='Specify an optimizer')] = 'Adam',
    lr: Annotated[float, typer.Option(help='Learning rate')] = 1e-3,
    top_k: Annotated[int, typer.Option(help='Specify top K for metrics')] = 10,
    mlp_pretrain: Annotated[str, typer.Option(help='MLP pretrained model path')] = '',
    gmf_pretrain: Annotated[str, typer.Option(help='GMF pretrained model path')] = '',
    alpha: Annotated[float, typer.Option(help='Alpha parameters used in NeuMF')] = 0.5,
    max_epochs: Annotated[int, typer.Option(help='Max number of epochs to train')] = 100,
    logger: Annotated[str, typer.Option(help="logger to use. one of (`wandb`, `tensorboard`)")] = 'tensorboard',
):
    data_module = DataModule(data_dir, int(layers[0] / 2), batch_size, num_workers, num_neg)
    data_module.setup(stage='fit')

    if model == 'mlp':
        model = MLP(layers, optimizer, lr, top_k)
    elif model == 'gmf':
        model = GMF(optimizer, lr, top_k)
    elif model == 'neumf':
        model = NeuMF(layers, gmf_pretrain, mlp_pretrain, alpha)
    else:
        raise NotImplementedError()

    if logger == 'wandb':
        logger = WandbLogger(name=Config.PROJECTNAME, save_dir=Config.WANDBPATH, project=Config.PROJECTNAME)
    else:
        logger = TensorBoardLogger(save_dir=Config.TENSORBOARDLOGGERPATH, name=Config.PROJECTNAME)

    trainer = LabTrainer(
        devices='auto',
        accelerator='auto',
        strategy='auto',
        num_nodes=1,
        precision="32-true",
        enable_checkpointing=True,
        max_epochs=max_epochs,
        callbacks=EarlyStopping(monitor='val-loss', mode='min'),
        logger=logger,
        profiler=PyTorchProfiler(dirpath='logs/torch_profiler'),
    )
    trainer.fit(model, data_module)
