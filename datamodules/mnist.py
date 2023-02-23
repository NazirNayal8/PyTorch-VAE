import pytorch_lightning as pl

from torchvision import transforms
from torchvision.datasets import MNIST

from torch.utils.data import DataLoader, Subset


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = config.DATA.ROOT
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.dims = (1, 28, 28)
        self.num_classes = config.DATA.NUM_CLASSES
        self.batch_size = config.SOLVER.BATCH_SIZE
        self.num_workers = config.SOLVER.NUM_WORKERS

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.dataset = MNIST(self.data_dir, train=True,
                                 transform=self.transform)
            self.id_indices = []
            self.ood_indices = []
            self.class_indices = [[] for _ in range(self.num_classes)]
            for i in range(len(self.dataset)):
                y = self.dataset.targets[i]
                if y in self.config.DATA.ID_CLASSES:
                    self.id_indices.append(i)
                elif y in self.config.DATA.OOD_CLASSES:
                    self.ood_indices.append(i)
                elif y not in self.config.DATA.IGNORE_CLASSES:
                    raise ValueError(
                        f"There exists an instance of class id {y} that does not belong to any group.")
                self.class_indices[y].append(i)

            if self.config.DATA.SPLIT_MODE  == 'strict':
                self.train_indices = self.id_indices
                self.val_indices = self.ood_indices
            elif self.config.DATA.SPLIT_MODE == 'partial':
                pass

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(Subset(self.dataset, self.id_indices), batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(Subset(self.dataset, self.ood_indices), batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)
