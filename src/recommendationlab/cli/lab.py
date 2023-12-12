import os
from pathlib import Path

import torch
import typer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from typing_extensions import Annotated

from src.recommendationlab.core import LabTrainer
from src.recommendationlab.core.GMF import GMF
from src.recommendationlab.core.MLP import MLP
from src.recommendationlab.core.NeuMF import NeuMF
from src.recommendationlab.pipeline.datamodule import DataModule
from src.recommendationlab.pipeline.preprocess import Preprocessor

FILEPATH = Path(__file__)
PROJECTPATH = FILEPATH.parents[2]
PKGPATH = FILEPATH.parents[1]

app = typer.Typer()
docs_app = typer.Typer()
run_app = typer.Typer()
app.add_typer(docs_app, name="docs")


@app.callback()
def callback() -> None:
    pass


@docs_app.command('build')
def build_docs() -> None:
    import shutil

    os.system('mkdocs build')
    shutil.copyfile(src='README.md', dst='docs/index.md')


@docs_app.command('serve')
def serve_docs() -> None:
    os.system('mkdocs serve')


@run_app.command('preprocess')
def preprocess(data_dir: Annotated[str, typer.Option(help='Data directory')]):
    preprocessor = Preprocessor(data_dir)
    preprocessor.process_data()


@run_app.command('train')
def train(
    model_name: Annotated[str, typer.Option(help='Model to use. One of (`mlp`, `gmf`, `neumf`)')],
    batch_size: Annotated[int, typer.Option(help='Batch size')] = 8,
    num_workers: Annotated[int, typer.Option(help='Number of workers')] = 8,
    num_neg: Annotated[int, typer.Option(help='Number of negative instances to pair with a positive instance.')] = 4,
    layers: Annotated[str, typer.Option(help='MLP layers')] = '20,10',
    gmf_factor: Annotated[int, typer.Option(help='GMF factor')] = 8,
    optimizer: Annotated[str, typer.Option(help='Specify an optimizer')] = 'Adam',
    lr: Annotated[float, typer.Option(help='Learning rate')] = 1e-3,
    top_k: Annotated[int, typer.Option(help='Specify top K for metrics')] = 10,
    mlp_pretrain: Annotated[str, typer.Option(help='MLP pretrained model path')] = '',
    gmf_pretrain: Annotated[str, typer.Option(help='GMF pretrained model path')] = '',
    alpha: Annotated[float, typer.Option(help='Alpha parameters used in NeuMF')] = 0.5,
    max_epochs: Annotated[int, typer.Option(help='Max number of epochs to train')] = 100,
    dropout: Annotated[float, typer.Option(help='Dropout value')] = 0.1,
    fast_dev_run: Annotated[bool, typer.Option(help='Run dev run')] = False,
):
    if model_name == 'mlp':
        layers = [int(i) for i in layers.split(',')]
        data_module = DataModule(int(layers[0] / 2), batch_size, num_workers, num_neg)
        data_module.setup('fit')
        model = MLP(data_module.num_users, data_module.num_items, layers, optimizer, lr, top_k, dropout)
    elif model_name == 'gmf':
        data_module = DataModule(gmf_factor, batch_size, num_workers, num_neg)
        data_module.setup('fit')
        model = GMF(data_module.num_users, data_module.num_items, gmf_factor, optimizer, lr, top_k)
    elif model_name == 'neumf':
        layers = [int(i) for i in layers.split(',')]
        data_module = DataModule(int(layers[0] / 2), batch_size, num_workers, num_neg)
        data_module.setup('fit')
        model = NeuMF(
            data_module.num_users,
            data_module.num_items,
            layers,
            gmf_factor,
            gmf_pretrain,
            mlp_pretrain,
            alpha,
            optimizer,
            lr,
            top_k,
            dropout
        )
    else:
        raise NotImplementedError()

    if fast_dev_run:
        accelerator = 'cpu'
    else:
        accelerator = 'auto'

    trainer = LabTrainer(
        model_name=model_name,
        devices='auto',
        accelerator=accelerator,
        strategy='auto',
        num_nodes=1,
        precision='32-true',
        enable_checkpointing=True,
        max_epochs=max_epochs,
        fast_dev_run=fast_dev_run,
        callbacks=[
            EarlyStopping(monitor='val-HR', mode='max', patience=5, verbose=True),
            LearningRateMonitor(logging_interval='epoch')
        ],
        gradient_clip_algorithm='norm',
        gradient_clip_val=5.0
    )

    trainer.fit(model, data_module)


@run_app.command('evaluate')
def evaluate(
    model_name: Annotated[str, typer.Option(help='Model to use. One of (`mlp`, `gmf`, `neumf`)')],
    model_path: Annotated[str, typer.Option(help='Model path to use')]
):
    hparams = torch.load(model_path, map_location=lambda storage, loc: storage)
    if model_name == 'gmf':
        model = GMF.load_from_checkpoint(model_path)
    elif model_name == 'mlp':
        model = MLP.load_from_checkpoint(model_path)
    elif model_name == 'neumf':
        model = NeuMF.load_from_checkpoint(model_path)

    hyp = hparams['hyper_parameters']
    if hyp['embed_size']:
        embed_size = hyp['embed_size']
    else:
        embed_size = hyp['layers'][0] / 2

    data_module = DataModule(embed_size)
    data_module.setup('test')
    trainer = LabTrainer(model_name=model)
    trainer.test(model, data_module)


@run_app.command('predict')
def predict(
    model_name: Annotated[str, typer.Option(help='Model')],
    model_path: Annotated[str, typer.Option(help='Model path to use')],
    user_inputs: Annotated[str, typer.Option(help='User inputs')],
    item_inputs: Annotated[str, typer.Option(help='Item inputs')],
):
    hparams = torch.load(model_path, map_location=lambda storage, loc: storage)
    if model_name == 'gmf':
        model = GMF.load_from_checkpoint(model_path)
    elif model_name == 'mlp':
        model = MLP.load_from_checkpoint(model_path)
    elif model_name == 'neumf':
        model = NeuMF.load_from_checkpoint(model_path)

    user_inputs = [i for i in user_inputs.split(',')]
    item_inputs = [int(i) for i in item_inputs.split(',')]
    hyp = hparams['hyper_parameters']
    if hyp['embed_size']:
        embed_size = hyp['embed_size']
    else:
        embed_size = hyp['layers'][0] / 2
    data_module = DataModule(embed_size, predict_data=(user_inputs, item_inputs))
    data_module.setup('predict')
    trainer = LabTrainer(model_name=model)
    preds = trainer.predict(model, data_module)
    print('Predictions: {}'.format(preds))
