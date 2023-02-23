import pytorch_lightning as pl
import numpy as np
from torchvision import transforms as T
from torchvision.datasets import CIFAR10, SVHN, MNIST
from torch.utils.data import DataLoader, Subset

class MultiOODDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.root = config.DATA.ROOT

        self.cifar_mean = [0.4914, 0.4822, 0.4465]
        self.cifar_std = [0.2471, 0.2435, 0.2616]

        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        
        if config.DATA.NORMALIZATION == 'cifar':
            mean = self.cifar_mean
            std = self.cifar_std
        elif config.DATA.NORMALIZATION == 'imagenet':
            mean = self.imagenet_mean
            std = self.imagenet_std


        self.transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])

        self.val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std)
        ])

        self.num_classes = self.config.DATA.NUM_CLASSES
        self.batch_size = self.config.SOLVER.BATCH_SIZE
        self.num_workers = self.config.SOLVER.NUM_WORKERS

    def prepare_data(self):
        # download data
        CIFAR10(root=self.root, train=True, download=True)
        CIFAR10(root=self.root, train=False, download=True)
        SVHN(root=self.root, split='train', download=True)
        SVHN(root=self.root, split='test', download=True)
        MNIST(self.root, train=True, download=True)
        MNIST(self.root, train=False, download=True)

    
    def get_train_dataset(self):

        # if self.DATA.TRAIN == 'cifar':
        #     return CIFAR10(root=self.root, train=True, transform=self.transform)
        # elif self.DATA.TRAIN == 'mnist':
        #     return MNIST(root=self.root, train=True, transform=self.transform)
        # elif self.DATA.TRAIN == 'svhn':
        #     return SVHN(root=self.svhn, split='train', transform=self.transform)
        # else:
        #     raise ValueError(f"undefined train dataset: {self.DATA.TRAIN}")
        return CIFAR10(root=self.root, train=True, transform=self.transform)

    def get_val_dataset(self):
        
        val_dataset = CIFAR10(root=self.root, train=False, transform=self.transform)

        if self.config.DATA.VAL == 'mnist':
            
            d = MNIST(root=self.root, train=False, transform=self.transform)
            
            val_dataset.data = np.concatenate([val_dataset.data, d.data], axis=0)
            val_dataset.targets = val_dataset.targets + (d.targets + self.num_classes).tolist()

        elif self.config.DATA.VAL == 'svhn':
            
            d = SVHN(root=self.root, split='test', transform=self.transform)
            val_dataset.data = np.concatenate([val_dataset.data, d.data.transpose(0, 2, 3, 1)], axis=0)
            val_dataset.targets = val_dataset.targets + (d.labels + self.num_classes).tolist()
            
        else:
            raise ValueError(f"undefined val dataset: {self.config.DATA.VAL}")

        return val_dataset

    def setup(self, stage=None):

        if stage == "fit" or stage is None:

            self.train_dataset = self.get_train_dataset()
            self.val_dataset = self.get_val_dataset()

        
        if stage == "test" or stage is None:
            self.test_dataset = self.get_val_dataset()

    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self) :
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
            