import os
import wandb
import math
import torch
import inspect
from torch import optim
from models import BaseVAE
from models.vae.types_ import * 
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from models import *
from easydict import EasyDict as edict

def create_model(config):

    MODEL = vae_models[config.NAME]

    params = edict()
    # gets the names of variables expected as input
    expected_params = inspect.getfullargspec(MODEL).args

    for k, v in config.items():
        # params in config are expected to be exact uppercase versions of param name
        p = k.lower()
        if p in expected_params:
            params[p] = v

    return MODEL(**params)


class VAEXperiment(pl.LightningModule):

    def __init__(
        self,
        config
    ) -> None:
        super(VAEXperiment, self).__init__()

        self.save_hyperparameters(config)

        self.model = create_model(self.hparams.MODEL)

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):

        x, y = batch

        outputs = self.forward(x, y=y)
        
        train_loss = self.model.loss_function(
            recons=outputs.recon,
            x=x,
            vq_loss=outputs.vq_loss,
            M_N=self.hparams.SOLVER.KLD_WEIGHT,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx
        )

        self.log("train/perplexity", outputs.perplexity.item(), on_epoch=True, on_step=True)
    
        for k, v in train_loss.items():
            self.log(f"train/{k}", v.item(), on_epoch=True,
                     on_step=True, sync_dist=True, prog_bar=True)

        return train_loss.loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):

        x, y = batch

        outputs = self.forward(x, y=y)

        val_loss = self.model.loss_function(
            recons=outputs.recon,
            x=x,
            vq_loss=outputs.vq_loss,
            M_N=self.hparams.SOLVER.KLD_WEIGHT,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx
        )

        self.log("val_perplexity", outputs.perplexity.item(), on_epoch=True, on_step=True)

        for k, v in val_loss.items():
            self.log(f"val_{k}", v.item(), on_epoch=True,
                     on_step=True, sync_dist=True)

    def on_validation_end(self) -> None:

        if self.hparams.WANDB.LOG_RECONSTRUCTIONS:
            self.sample_images()

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(
            iter(self.trainer.datamodule.test_dataloader()))

        # number of samples to be logged is determined by config
        # just making sure that number does not exceed batch size
        # NOTE: it better be a power of 2 and with an even power
        num_samples = min(test_input.shape[0], self.hparams.WANDB.NUM_LOG_IMGS)

        test_input = test_input[:num_samples]
        test_label = test_label[:num_samples]

        test_input = test_input.to(self.device)
        test_label = test_label.to(self.device)

        recons = self.model.generate(test_input, y=test_label)

        vutils.save_image(
            recons.data,
            os.path.join(
                self.hparams.WANDB.LOG_DIR,
                "Reconstructions", f"{self.hparams.WANDB.RUN_NAME}/recons_epoch_{self.current_epoch}.png"),
            normalize=True,
            nrow=int(np.sqrt(num_samples))
        )

        # also plot to Wandb if activated (you better activate it!)
        if self.hparams.WANDB.ACTIVATE:
            grid = vutils.make_grid(recons, nrow=int(
                np.sqrt(num_samples)), normalize=True, scale_each=True)
            wandb_img = wandb.Image(grid)
            self.logger.experiment.log({
                "reconstructions": wandb_img,
                "epoch": self.current_epoch
            })

        # if sampling is implemented, then sample some images and log them
        try:
            samples = self.model.sample(
                self.hparams.WANDB.NUM_LOG_IMGS, self.device, y=test_label)
            vutils.save_image(
                samples.cpu().data,
                os.path.join(
                    self.hparams.WANDB.LOG_DIR,
                    "Samples",
                    f"{self.hparams.WANDB.RUN_NAME}/epoch_{self.current_epoch}.png"),
                normalize=True,
                nrow=int(np.sqrt(num_samples)),
            )
        except Warning:
            pass

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(
            self.parameters(), 
            lr=self.hparams.SOLVER.LR,
            weight_decay=self.hparams.SOLVER.WEIGHT_DECAY,
            amsgrad=False
            )
    
        return optimizer
