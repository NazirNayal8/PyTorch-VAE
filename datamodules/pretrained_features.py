from .svhn import SVHNDataModule
from .cifar import CIFAR10DataModule
from .datasets import get_datasets
from experiment import VAEXperiment
import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
sys.path.append("../")
sys.path.append("../../")


DATAMODULES = edict(
    CIFAR10=CIFAR10DataModule,
    SVHN=SVHNDataModule
)


class FeatureDataset(Dataset):

    def __init__(self, root):
        super().__init__()

        self.root = root

        data = os.listdir(self.root)
        self.data = [os.path.join(self.root, d) for d in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        path = self.data[index]
        x = np.load(path)
        y = 0

        x = torch.from_numpy(x).float().contiguous()
        y = torch.tensor(y).long()

        return x, y


class DINOV2FeaturesDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.root = config.DATA.ROOT

        self.batch_size = self.config.SOLVER.BATCH_SIZE
        self.num_workers = self.config.SOLVER.NUM_WORKERS

    def get_features(self, model, x):
        with torch.no_grad():
            output = model.forward_features(x)
        return output[self.config.MODEL.FEATURES_TYPE].squeeze(0)

    def generate_and_save_features(self, model, dataset, path, device):

        if os.path.exists(path):
            contents = os.listdir(path)
            if len(contents) == len(dataset) or len(contents) == len(dataset) + 1:
                print(
                    f"Data already exists at {path}. Skipping data generation.")
                return
        # generate data
        print(f"Generating data at {path}...")
        os.makedirs(path, exist_ok=True)

        for i in tqdm(range(len(dataset))):
            x, y = dataset[i]
            
            features = self.get_features(model, x.unsqueeze(0).to(device))

            N, C = features.shape
            if self.config.MODEL.FEATURES_TYPE == "x_prenorm":
                N = N - 1
            N_sqrt = int(np.sqrt(N))
            assert N_sqrt * N_sqrt == N, f"N is not a perfect square: {N}"

            if self.config.MODEL.FEATURES_TYPE == "x_prenorm":
                features = features[1:].permute(1, 0).view(C, N_sqrt, N_sqrt)
            elif self.config.MODEL.FEATURES_TYPE == "x_norm_patchtokens":
                features = features.permute(1, 0).view(C, N_sqrt, N_sqrt)

            np.save("{}/{}.npy".format(path, i), features.cpu().numpy())

    def prepare_data(self):

        # create the DINOV2 Mode
        
        dinov2_model = torch.hub.load(
            'facebookresearch/dinov2', self.config.MODEL.VARIATION, pretrained=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dinov2_model.to(device)
        dinov2_model.eval()

        self.data_path_train = os.path.join(
            "./data/dinov2", self.config.WANDB.RUN_NAME, self.config.DATA.SOURCE, 'train')
        self.data_path_test = os.path.join(
            "./data/dinov2", self.config.WANDB.RUN_NAME, self.config.DATA.SOURCE, 'test')
        datamodule = DATAMODULES[self.config.DATA.SOURCE](self.config)
        datamodule.setup()
        train_dataset = datamodule.train_dataset
        test_dataset = datamodule.test_dataset

        # get the indices of the codebook
        self.generate_and_save_features(
            dinov2_model, train_dataset, self.data_path_train, device)
        self.generate_and_save_features(
            dinov2_model, test_dataset, self.data_path_test, device)

    def setup(self, stage=None):

        self.train_dataset = FeatureDataset(
            root=self.data_path_train,
        )

        self.val_dataset = FeatureDataset(
            root=self.data_path_test,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return self.val_dataloader()
