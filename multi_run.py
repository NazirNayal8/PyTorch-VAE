import os
import yaml
import argparse
import wandb
from pathlib import Path
from models import *
from datamodules import DATAMODULES
from experiment import VAEXperiment
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from easydict import EasyDict as edict
from torchinfo import summary
from functools import partial

def read_config(filename):
    with open(filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    return config

def update_from_wandb(config, wandb_config):

    for k, v in wandb_config.items():
        if k not in config:
            raise ValueError(f"Wandb Config has sth that you don't")
        if isinstance(v, dict):
            config[k] = update_from_wandb(config[k], wandb_config[k])
        else:
            config[k] = v

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

    config = update_from_wandb(config, edict(logger.experiment.config))
    
    config.WANDB.RUN_NAME += F"_{config.MODEL.NUM_EMBEDDINGS}_{config.MODEL.EMBEDDING_DIM}"
    logger.experiment.name = config.WANDB.RUN_NAME
    print("Config", config)

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
            mode="min",
            save_last=True
        ),
    ]

    # define trainer object
    trainer = Trainer(
        fast_dev_run=args.dev,
        accelerator="gpu",
        devices=-1,
        log_every_n_steps=50,
        max_epochs=config.SOLVER.MAX_EPOCHS,
        max_steps=config.SOLVER.MAX_STEPS,
        logger=logger,
        callbacks=callbacks
    )

    # create necessary folders
    Path(f"{config.WANDB.LOG_DIR}/Samples/{config.WANDB.RUN_NAME}").mkdir(exist_ok=True, parents=True)
    Path(f"{config.WANDB.LOG_DIR}/Reconstructions/{config.WANDB.RUN_NAME}").mkdir(
        exist_ok=True, parents=True)

    # start training
    trainer.fit(experiment, datamodule=datamodule)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/vq_vae/vq_vae_v2.yaml')

    parser.add_argument("--sweep_config", default="configs/cifar10/sweep.yaml", help="wandb sweep config")
    parser.add_argument("--dev", action="store_true", help="Runs in Dev Mode")


    args = parser.parse_args()

    sweep_config = read_config(args.sweep_config)
    
    sweep_id = wandb.sweep(sweep_config, project=sweep_config['project'])
    wandb.agent(sweep_id=sweep_id, function=partial(main, args))
    # main(args)
