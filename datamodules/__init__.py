
from .cifar import CIFAR10DataModule
from .svhn import SVHNDataModule
from .vq_vae_codebook import CodebookDataModule
from easydict import EasyDict as edict
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, CelebA, Omniglot
from .datasets import get_datasets


DATAMODULES = edict(
    CIFAR10=CIFAR10DataModule,
    SVHN=SVHNDataModule,
    Codebook=CodebookDataModule,
)

