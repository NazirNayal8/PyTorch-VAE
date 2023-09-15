import argparse
import os

from easydict import EasyDict as edict
from utils import read_config
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from classification import Classifier
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from datamodules import DATAMODULES

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

    classifier = Classifier(config)

    # define the DataModule
    datamodule = DATAMODULES[config.DATA.NAME](config)

    # define callbacks
    callbacks = [
        LearningRateMonitor() if not args.dev else ModelSummary(),
        ModelCheckpoint(
            save_top_k=6,
            dirpath=os.path.join(config.CKPT.DIR_PATH, config.DATA.NAME, config.WANDB.RUN_NAME),
            monitor="val_acc_epoch",
            mode="max",
            filename='{epoch}-{val_loss_epoch:.2f}',
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
                        default='configs/classifier/mlp.yaml')
    
    parser.add_argument("--dev", action="store_true", help="Runs in Dev Mode")

    args = parser.parse_args()

    main(args)