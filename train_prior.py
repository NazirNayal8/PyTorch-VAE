 
import os
import argparse

from pathlib import Path
from datamodules import DATAMODULES
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import seed_everything
from easydict import EasyDict as edict
from utils import read_config
from prior import VQVAEPrior
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning import Trainer


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

    # for reproducibility
    seed_everything(config.RANDOM_SEED)

    # create lightning module
    experiment = VQVAEPrior(config)

    # define the DataModule
    datamodule = DATAMODULES[config.DATA.NAME](config)

    # define callbacks
    callbacks = [
        LearningRateMonitor() if not args.dev else ModelSummary(),
        ModelCheckpoint(
            save_top_k=2,
            dirpath=os.path.join(config.CKPT.DIR_PATH, config.DATA.NAME, config.WANDB.RUN_NAME),
            monitor="val_loss_epoch",
            mode="min",
            filename='{epoch}-{val_loss_epoch:.2f}',
            save_last=True
        ),
    ]

    # define trainer object
    trainer = Trainer(
        fast_dev_run=args.dev,
        accelerator="gpu",
        devices=-1,
        log_every_n_steps=50,
        logger=logger,
        callbacks=callbacks,
        max_epochs=config.SOLVER.MAX_EPOCHS,
    )

    # create necessary folders
    Path(f"{config.WANDB.LOG_DIR}/samples/{config.WANDB.RUN_NAME}").mkdir(exist_ok=True, parents=True)
    
    # start training
    trainer.fit(experiment, datamodule=datamodule)

if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description='Prior Trainer')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/prior/pixel_cnn/default.yaml')
    
    parser.add_argument("--dev", action="store_true", help="Runs in Dev Mode")

    args = parser.parse_args()

    main(args)

