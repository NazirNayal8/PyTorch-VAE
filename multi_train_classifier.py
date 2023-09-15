import argparse
import os
import wandb
import torch

from easydict import EasyDict as edict
from utils import read_config, update_from_wandb
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from classification import Classifier
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from datamodules import DATAMODULES
from functools import partial

def main(args):
    
    # prepare config dict
    config = read_config(args.filename)
    config = edict(config)

     # create logger
    if config.WANDB.ACTIVATE and not args.dev:
        logger = WandbLogger(
            name=config.WANDB.RUN_NAME,
            project=config.WANDB.PROJECT, 
            config=config
        )
    else:
        logger = None

    seed_everything(config.RANDOM_SEED)

    config = update_from_wandb(config, edict(logger.experiment.config))
    
    vq_vae_ckpt = torch.load(config.DATA.VQ_VAE_PATH)
    vq_vae_hparams = edict(vq_vae_ckpt["hyper_parameters"])
    
    config.MODEL.CODEBOOK_SIZE = vq_vae_hparams.MODEL.NUM_EMBEDDINGS
    config.MODEL.EMBEDDING_DIM = vq_vae_hparams.MODEL.EMBEDDING_DIM

    config.WANDB.RUN_NAME = vq_vae_hparams.WANDB.RUN_NAME
    logger.experiment.name = config.WANDB.RUN_NAME
    
    classifier = Classifier(config)

    # define the DataModule
    datamodule = DATAMODULES[config.DATA.NAME](config)

    # define callbacks
    callbacks = [
        LearningRateMonitor() if not args.dev else ModelSummary(),
        ModelCheckpoint(
            save_top_k=2,
            dirpath=os.path.join(config.CKPT.DIR_PATH, config.DATA.NAME, config.WANDB.RUN_NAME),
            monitor="val_acc_epoch",
            mode="max",
            filename='{epoch}-{val_acc_epoch:.2f}',
            save_last=True
        ),
    ]

    # define trainer object
    trainer = Trainer(
        fast_dev_run=args.dev,
        accelerator="gpu",
        devices=-1,
        log_every_n_steps=100,
        logger=logger,
        callbacks=callbacks,
        max_epochs=config.SOLVER.MAX_EPOCHS,
    )

    # start training
    trainer.fit(classifier, datamodule=datamodule)

if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description='Classifier Trainer')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/classifier/mlp_vectors.yaml')
    parser.add_argument(
        "--sweep_config", default="configs/classifier/sweep.yaml", help="wandb sweep config")
    parser.add_argument("--dev", action="store_true", help="Runs in Dev Mode")

    args = parser.parse_args()

    sweep_config = read_config(args.sweep_config)
    sweep_id = wandb.sweep(sweep_config, project=sweep_config['project'])
    wandb.agent(sweep_id=sweep_id, function=partial(main, args))