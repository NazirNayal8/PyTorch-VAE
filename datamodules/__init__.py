from .cifar import CIFAR10DataModule
from .svhn import SVHNDataModule
from easydict import EasyDict as edict


DATAMODULES = edict(
    CIFAR10=CIFAR10DataModule,
    SVHN=SVHNDataModule
)