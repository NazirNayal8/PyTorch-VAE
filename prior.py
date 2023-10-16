import os
import numpy as np
import pytorch_lightning as pl
import inspect
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import wandb

from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from pytorch_lightning.callbacks import BaseFinetuning
from typing import List, Union
from torch import Tensor
from models import PRIORS
from easydict import EasyDict as edict
from torch import optim
from experiment import VAEXperiment

def create_model(config):

    MODEL = PRIORS[config.NAME]

    params = edict()
    # gets the names of variables expected as input
    expected_params = inspect.getfullargspec(MODEL).args

    for k, v in config.items():
        # params in config are expected to be exact uppercase versions of param name
        p = k.lower()
        if p in expected_params:
            params[p] = v

    return MODEL(**params)


class VQVAEPrior(pl.LightningModule):

    def __init__(self, config) -> None:
        super(VQVAEPrior, self).__init__()

        vq_vae_ckpt = torch.load(config.VQ_VAE.CKPT_PATH)
        self.vq_vae_hparams = edict(vq_vae_ckpt["hyper_parameters"])
        self.vq_vae = VAEXperiment(self.vq_vae_hparams)
        config.MODEL.INPUT_DIM = self.vq_vae_hparams.MODEL.NUM_EMBEDDINGS

        self.save_hyperparameters(config)

        self.model = create_model(self.hparams.MODEL)
        self.vq_vae.load_state_dict(vq_vae_ckpt["state_dict"])
        self.vq_vae = self.vq_vae.model
        BaseFinetuning.freeze(self.vq_vae)
        self.vq_vae.eval()


    def forward(self, input: Tensor, label: Tensor, **kwargs):
        return self.model(input, label=label, **kwargs)

    def configure_optimizers(self):

        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.SOLVER.LR,
            weight_decay=self.hparams.SOLVER.WEIGHT_DECAY,
        )

        return optimizer

    def training_step(self, batch, batch_idx):

        x, y = batch
        B = x.shape[0]

        codebook_outputs = self.vq_vae.vq(
            self.vq_vae.pre_vq_conv(self.vq_vae.encoder(x)))
        H, W = codebook_outputs.quantized.shape[2:]

        encoding_indices = codebook_outputs.encoding_indices[:, 0].long()
        encoding_indices = encoding_indices.view(B, H, W)

        # get the prior
        prior = self.model(encoding_indices, label=y)  # shape: B, C, H, W
        prior = prior.permute(0, 2, 3, 1).contiguous()  # shape: B, H, W, C

        loss = F.cross_entropy(
            prior.view(-1, self.hparams.MODEL.INPUT_DIM), encoding_indices.view(-1))

        self.log("train_loss", loss.item(), on_epoch=True,
                 on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        B = x.shape[0]

        codebook_outputs = self.vq_vae.vq(
            self.vq_vae.pre_vq_conv(self.vq_vae.encoder(x)))
        H, W = codebook_outputs.quantized.shape[2:]

        encoding_indices = codebook_outputs.encoding_indices[:, 0].long()
        encoding_indices = encoding_indices.view(B, H, W)

        # get the prior
        prior = self.model(encoding_indices, label=y)
        prior = prior.permute(0, 2, 3, 1).contiguous()

        loss = F.cross_entropy(
            prior.view(-1, self.hparams.MODEL.INPUT_DIM), encoding_indices.view(-1))

        self.log("val_loss", loss.item(), on_epoch=True,
                 on_step=True, prog_bar=True)

        return loss

    def on_validation_end(self) -> None:
        
        if not self.hparams.WANDB.GENERATE_SAMPLES:
            return

        samples = self.generate_samples(batch_size=40)

        num_samples = min(samples.shape[0], self.hparams.WANDB.NUM_LOG_IMGS)

        vutils.save_image(
            samples.cpu().data,
            os.path.join(
                self.hparams.WANDB.LOG_DIR,
                "samples",
                f"{self.hparams.WANDB.RUN_NAME}/epoch_{self.current_epoch}.png"),
            normalize=True,
            nrow=int(np.sqrt(num_samples)),
        )

        # also plot to Wandb if activated (you better activate it!)
        if self.hparams.WANDB.ACTIVATE:
            grid = vutils.make_grid(samples.cpu().data, nrow=int(
                np.sqrt(num_samples)), normalize=True, scale_each=True)
            wandb_img = wandb.Image(grid)
            self.logger.experiment.log({
                "samples": wandb_img,
                "epoch": self.current_epoch
            })

    @torch.no_grad()
    def generate_samples(self, shape=(8, 8), batch_size=1):

        label = torch.arange(10).expand(4, 10).contiguous().view(-1)  # shape: (40,)
        label = label.long().cuda()

        x_q = self.model.generate(label=label, shape=shape, batch_size=batch_size)
        x_q = x_q.view(-1, 1)

        encodings = torch.zeros(
            x_q.shape[0], self.hparams.MODEL.INPUT_DIM, device=self.device)
        encodings.scatter_(1, x_q, 1)

        embed_weight = self.vq_vae.vq._embedding.weight
        quantized = torch.matmul(encodings, embed_weight).view(
            batch_size, *shape, -1)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        x_recon = self.vq_vae.decoder(quantized)

        return x_recon
