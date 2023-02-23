import pytorch_lightning as pl

from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset


class CIFAR10DataModule(pl.LightningDataModule):

    CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR_STD = [0.2471, 0.2435, 0.2616]

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.root = config.DATA.ROOT
        
        if config.DATA.NORMALIZATION == 'cifar':
            mean = self.CIFAR_MEAN
            std = self.CIFAR_STD
        elif config.DATA.NORMALIZATION == 'imagenet':
            mean = self.IMAGENET_MEAN
            std = self.IMAGENET_STD
        elif config.DATA.NORMALIZATION == 'custom':
            mean = config.DATA.MEAN
            std = config.DATA.STD

        self.transform = T.Compose([
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

    def setup(self, stage=None):

        if stage == "fit" or stage is None:
            self.train_dataset = CIFAR10(
                root=self.root,
                train=True,
                transform=self.transform
            )
            self.val_dataset = CIFAR10(
                root=self.root,
                train=False,
                transform=self.val_transform
            )
            # allow the options of training on certain classes of CIFAR
            if self.config.DATA.CLS_SUBSET is not None:
                self.train_indices = []
                for i in range(len(self.train_dataset)):
                    y = self.train_dataset.targets[i]
                    if y in self.config.DATA.CLS_SUBSET:
                        self.train_indices.append(i)
            else:
                self.train_indices = list(range(len(self.train_dataset)))

        
        self.test_dataset = CIFAR10(
            root=self.root,
            train=False,
            transform=self.val_transform
        )

    def train_dataloader(self):
        return DataLoader(
            Subset(self.train_dataset, self.train_indices),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )



