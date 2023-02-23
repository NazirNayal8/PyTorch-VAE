import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from datamodules import DATAMODULES
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
from pytorch_lightning.plugins import DDPPlugin
from easydict import EasyDict as edict
from torchinfo import summary


def read_config(filename):
    with open(filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    return config


def main(args):

    # prepare config dict
    config = read_config(args.filename)
    config = edict(config)

    # create logger
    if config.WANDB.ACTIVATE:
        logger = WandbLogger(
            name=config.WANDB.RUN_NAME,
            project=config.WANDB.PROJECT, 
            config=config
        )
    else:
        logger = None

    # for reproducibility
    seed_everything(config.RANDOM_SEED)

    # create lightning module
    experiment = VAEXperiment(config)
    print(summary(experiment.model, input_size=(1, 3, 32, 32), depth=10))
    
    # define the DataModule
    datamodule = DATAMODULES[config.DATA.NAME](config)

    # define callbacks
    callbacks = [
        LearningRateMonitor(),
        ModelCheckpoint(
            save_top_k=2,
            dirpath=os.path.join(config.CKPT.DIR_PATH, config.DATA.NAME, config.WANDB.RUN_NAME),
            monitor="val_loss_epoch",
            mode="max",
            save_last=True
        ),
    ]
    
    # define trainer object
    trainer = Trainer(
        fast_dev_run=args.dev,
        accelerator="gpu",
        devices=-1,
        # deterministic=True,
        log_every_n_steps=1,
        max_epochs=config.SOLVER.MAX_EPOCHS,
        max_steps=config.SOLVER.MAX_STEPS,
        logger=logger,
        callbacks=callbacks
    )

    # create necessary folders
    Path(f"{config.WANDB.LOG_DIR}/Samples/{config.WANDB.RUN_NAME}").mkdir(exist_ok=True, parents=True)
    Path(f"{config.WANDB.LOG_DIR}/Reconstructions/{config.WANDB.RUN_NAME}").mkdir(exist_ok=True, parents=True)

    trainer.fit(experiment, datamodule=datamodule)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/vae.yaml')
    
    parser.add_argument("--dev", action="store_true", help="Runs in Dev Mode")

    args = parser.parse_args()

    main(args)
