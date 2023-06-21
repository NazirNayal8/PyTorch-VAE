import torch.nn.functional as F
import pytorch_lightning as pl
import inspect
import copy

from models import CLASSIFIERS
from easydict import EasyDict as edict
from torch import optim
from torchmetrics import Accuracy

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

        config = self.preprocess_config(config)
        
        self.save_hyperparameters(config)

        self.model = create_classifier(self.hparams.MODEL)
        self.accuracy = Accuracy(task='multiclass', num_classes=self.hparams.MODEL.NUM_CLASSES)
        self.accuracy_per_class = Accuracy(task='multiclass', num_classes=self.hparams.MODEL.NUM_CLASSES, average=None)

    def preprocess_config(self, config):

        config.MODEL.INPUT_SIZE = config.MODEL.NUM_TOKENS

        if config.DATA.MODE == "one_hot":
            config.MODEL.INPUT_SIZE = config.MODEL.CODEBOOK_SIZE * config.MODEL.NUM_TOKENS
        elif config.DATA.MODE == "vectors":
            config.MODEL.INPUT_SIZE = config.MODEL.EMBEDDING_DIM * config.MODEL.NUM_TOKENS

        return config

    def forward(self, x):
        return self.model(x)
    
    def criterion(self, logits, y):

        return F.cross_entropy(logits, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss.item(), on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss.item(), on_epoch=True, on_step=True, prog_bar=True)

        self.accuracy(logits, y)
        self.accuracy_per_class(logits, y)
        self.log('val_acc', self.accuracy, on_epoch=True, on_step=True, prog_bar=True)

        return loss
    
    def validation_epoch_end(self, outputs):

        accuracy_per_class = self.accuracy_per_class.compute()
        for c, acc in enumerate(accuracy_per_class):
            self.log_dict({
                f'val_acc_{c}': acc.item(),
                'epoch': self.current_epoch
            })


    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log('test_loss', loss.item(), on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.SOLVER.LR,
            weight_decay=self.hparams.SOLVER.WEIGHT_DECAY,
        )

        return optimizer