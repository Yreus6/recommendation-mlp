import os
from pathlib import Path

import pandas as pd
import typer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from typing_extensions import Annotated

from src.recommendationlab import config
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
app.add_typer(docs_app, name='docs')
app.add_typer(run_app, name='run')


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


@run_app.command('build-vocab')
def build_vocab(data_dir: Annotated[str, typer.Option(help='Data directory')]):
    preprocessor = Preprocessor(data_dir)
    preprocessor.build_vocab()


@run_app.command('train')
def train(
    model_name: Annotated[str, typer.Option(help='Model to use. One of (`mlp`, `gmf`, `neumf`)')],
    batch_size: Annotated[int, typer.Option(help='Batch size')] = 8,
    num_workers: Annotated[int, typer.Option(help='Number of workers')] = 8,
    num_neg: Annotated[int, typer.Option(help='Number of negative instances to pair with a positive instance.')] = 4,
    gmf_user_embed_sizes: Annotated[str, typer.Option(help='GMF user embed size')] = '20,10,10',
    gmf_item_embed_sizes: Annotated[str, typer.Option(help='GMF item embed size')] = '20,10,10',
    mlp_user_embed_sizes: Annotated[str, typer.Option(help='MLP user embed size')] = '20,10,10',
    mlp_item_embed_sizes: Annotated[str, typer.Option(help='MLP item embed size')] = '20,10,10',
    layer_size: Annotated[int, typer.Option(help='MLP layer size')] = 3,
    optimizer: Annotated[str, typer.Option(help='Specify an optimizer')] = 'Adam',
    lr: Annotated[float, typer.Option(help='Learning rate')] = 1e-3,
    top_k: Annotated[int, typer.Option(help='Specify top K for metrics')] = 10,
    mlp_pretrain: Annotated[str, typer.Option(help='MLP pretrained model path')] = '',
    gmf_pretrain: Annotated[str, typer.Option(help='GMF pretrained model path')] = '',
    alpha: Annotated[float, typer.Option(help='Alpha parameters used in NeuMF')] = 0.5,
    max_epochs: Annotated[int, typer.Option(help='Max number of epochs to train')] = 100,
    dropout: Annotated[float, typer.Option(help='Dropout value')] = 0.1,
    patience: Annotated[int, typer.Option(help='Early stop patience')] = 5,
    gradient_clip_algorithm: Annotated[str, typer.Option(help='Gradient clip algorithm')] = 'norm',
    gradient_clip_val: Annotated[float, typer.Option(help='Gradient clip value')] = 5.0,
    resume: Annotated[str, typer.Option(help='Resume ckpt training')] = 'last',
    fast_dev_run: Annotated[bool, typer.Option(help='Run dev run')] = False,
):
    if model_name == 'mlp':
        mlp_user_embed_sizes = [int(i) for i in mlp_user_embed_sizes.split(',')]
        mlp_item_embed_sizes = [int(i) for i in mlp_item_embed_sizes.split(',')]
        data_module = DataModule(batch_size, num_workers, num_neg)
        data_module.setup('fit')
        model = MLP(
            data_module.users_fields,
            data_module.items_fields,
            mlp_user_embed_sizes,
            mlp_item_embed_sizes,
            layer_size,
            optimizer,
            lr,
            top_k,
            dropout
        )
    elif model_name == 'gmf':
        gmf_user_embed_sizes = [int(i) for i in gmf_user_embed_sizes.split(',')]
        gmf_item_embed_sizes = [int(i) for i in gmf_item_embed_sizes.split(',')]
        data_module = DataModule(batch_size, num_workers, num_neg)
        data_module.setup('fit')
        model = GMF(
            data_module.users_fields,
            data_module.items_fields,
            gmf_user_embed_sizes,
            gmf_item_embed_sizes,
            optimizer,
            lr,
            top_k
        )
    elif model_name == 'neumf':
        gmf_user_embed_sizes = [int(i) for i in gmf_user_embed_sizes.split(',')]
        gmf_item_embed_sizes = [int(i) for i in gmf_item_embed_sizes.split(',')]
        mlp_user_embed_sizes = [int(i) for i in mlp_user_embed_sizes.split(',')]
        mlp_item_embed_sizes = [int(i) for i in mlp_item_embed_sizes.split(',')]
        data_module = DataModule(batch_size, num_workers, num_neg)
        data_module.setup('fit')
        model = NeuMF(
            data_module.users_fields,
            data_module.items_fields,
            gmf_user_embed_sizes,
            gmf_item_embed_sizes,
            mlp_user_embed_sizes,
            mlp_item_embed_sizes,
            layer_size,
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
    
    best_model_callbacks = ModelCheckpoint(
        dirpath=os.path.join(config.CHKPTSPATH, model_name),
        filename='best',
        monitor='val-HR',
        mode='max'
    )
    best_model_callbacks.FILE_EXTENSION = '.pth'
    
    trainer = LabTrainer(
        devices='auto',
        accelerator=accelerator,
        strategy='auto',
        num_nodes=1,
        precision='32-true',
        enable_checkpointing=True,
        max_epochs=max_epochs,
        gradient_clip_algorithm=gradient_clip_algorithm,
        gradient_clip_val=gradient_clip_val,
        fast_dev_run=fast_dev_run,
        callbacks=[
            EarlyStopping(monitor='val-HR', mode='max', patience=patience, verbose=True),
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(
                dirpath=os.path.join(config.CHKPTSPATH, model_name),
                filename='model-{epoch:02d}',
            ),
            best_model_callbacks
        ]
    )
    
    trainer.fit(model, data_module, ckpt_path=resume)


@run_app.command('evaluate')
def evaluate(
    model_name: Annotated[str, typer.Option(help='Model to use. One of (`mlp`, `gmf`, `neumf`)')],
    model_path: Annotated[str, typer.Option(help='Model path to use')]
):
    if model_name == 'gmf':
        model = GMF.load_from_checkpoint(model_path)
    elif model_name == 'mlp':
        model = MLP.load_from_checkpoint(model_path)
    elif model_name == 'neumf':
        model = NeuMF.load_from_checkpoint(model_path)
    else:
        raise NotImplementedError()
    
    data_module = DataModule()
    data_module.setup('test')
    trainer = LabTrainer()
    trainer.test(model, data_module)


@run_app.command('predict')
def predict(
    model_name: Annotated[str, typer.Option(help='Model')],
    model_path: Annotated[str, typer.Option(help='Model path to use')],
    users_file: Annotated[str, typer.Option(help='Users csv file')],
    items_file: Annotated[str, typer.Option(help='Items csv file')],
):
    if model_name == 'gmf':
        model = GMF.load_from_checkpoint(model_path)
    elif model_name == 'mlp':
        model = MLP.load_from_checkpoint(model_path)
    elif model_name == 'neumf':
        model = NeuMF.load_from_checkpoint(model_path)
    else:
        raise NotImplementedError()
    
    users = pd.read_csv(users_file)
    items = pd.read_csv(items_file)
    assert len(users) == len(items), f'Users and items must have the same length'
    
    data_module = DataModule(predict_data=(users, items))
    data_module.setup('predict')
    trainer = LabTrainer()
    preds = trainer.predict(model, data_module)
    
    predictions = pd.DataFrame(preds, columns=['prediction'])
    result = pd.concat([users, items, predictions], ignore_index=True)
    result.to_csv(config.PREDSPATH, 'predictions.csv', index=False)
