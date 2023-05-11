import torch.nn.functional as F
import pytorch_lightning as pl
import inspect

from models import CLASSIFIERS
from easydict import EasyDict as edict
from torch import optim


def create_classifier(config):

    MODEL = CLASSIFIERS[config.NAME]

    params = edict()
    # gets the names of variables expected as input
    expected_params = inspect.getfullargspec(MODEL).args

    for k, v in config.items():
        # params in config are expected to be exact uppercase versions of param name
        p = k.lower()
        if p in expected_params:
            params[p] = v

    return MODEL(**params)

# write a pytorch lightning module use to train an any classifier

class Classifier(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters(config)

        self.model = create_classifier

    def forward(self, x):
        return self.model(x)
    
    def criterion(self, logits, y):

        return F.cross_entropy(logits, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.SOLVER.LR,
            weight_decay=self.hparams.SOLVER.WEIGHT_DECAY,
        )

        return optimizer