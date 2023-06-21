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
from experiment import VAEXperiment
from .datasets import get_datasets

from .cifar import CIFAR10DataModule
from .svhn import SVHNDataModule

DATAMODULES = edict(
    CIFAR10=CIFAR10DataModule,
    SVHN=SVHNDataModule,
)

class CodebookDataset(Dataset):

    def __init__(self, root, ood_mode, ood_prob, codebook_size, num_tokens, one_hot, corruption_params):
        super().__init__()

        self.root = root
        self.ood_mode = ood_mode
        self.ood_prob = ood_prob
        self.codebook_size = codebook_size
        self.num_tokens = num_tokens
        self.one_hot = one_hot
        self.corruption_params = corruption_params

        if not one_hot:
            # read codebook.npy from root
            codebook_root = '/'.join(root.split('/')[:-1])
            self.codebook = np.load(os.path.join(codebook_root, "codebook.npy"))

        data = os.listdir(self.root)
        self.data = [os.path.join(self.root, d) for d in data]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        path = self.data[index]
        x = np.load(path)
        y = 0

        if np.random.rand() < self.ood_prob:
            if self.ood_mode == "random":
                x = np.random.randint(low=0, high=self.codebook_size, size=self.num_tokens)
                y = 1
            elif self.ood_mode == "corruption":
                if self.corruption_params.MODE == "shuffle":
                    start = np.random.choice(np.arange(self.num_tokens - self.corruption_params.SHUFFLE_LEN))
                    end = start + self.corruption_params.SHUFFLE_LEN   
                    np.random.shuffle(x[start:end])
                    y = 1
                elif self.corruption_params.MODE == "insertion":
                    chosen_indices = np.random.choice(np.arange(self.num_tokens), size=self.corruption_params.INSERTION_LEN)
                    impure_tokens = np.random.randint(low=0, high=self.codebook_size, size=self.corruption_params.INSERTION_LEN)
            
                    if self.one_hot:
                        x[chosen_indices] = impure_tokens
                    else:
                        x[chosen_indices] = self.codebook[impure_tokens]
                    y = 1
                else:
                    raise NotImplementedError
        
        x = torch.from_numpy(x).float().contiguous()
    
        if self.one_hot:
            # turn x into a one-hot vector
            x = F.one_hot(x.long(), num_classes=self.codebook_size)

        x = x.view(-1).float()
        y = torch.tensor(y).long()
   
        return x, y
    

class CodebookTestDataset(Dataset):

    def __init__(self, id_root, ood_root, max_len, num_tokens, codebook_size, one_hot):
        super().__init__()

        self.id_root = id_root
        self.ood_root = ood_root
        self.max_len = max_len
        self.num_tokens = num_tokens
        self.codebook_size = codebook_size
        self.one_hot = one_hot

        self.data = []

        for i, p in enumerate(os.listdir(id_root)):
            self.data.append((os.path.join(id_root, p), 0))
            if i >= max_len:
                break

        for i, p in enumerate(os.listdir(ood_root)):
            self.data.append((os.path.join(ood_root, p), 1))
            if i >= max_len:
                break

    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, index):
            
            path, label = self.data[index]
            x = np.load(path)
            x = torch.from_numpy(x).float().contiguous()
            
            # turn x into a one-hot vector
            if self.one_hot:
                x = F.one_hot(x.long(), num_classes=self.codebook_size)
            x = x.view(-1).float()
            y = torch.tensor(label).long()
    
            return x, y


class CodebookDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.root = config.DATA.ROOT
        self.mode = config.DATA.MODE    

        self.batch_size = self.config.SOLVER.BATCH_SIZE
        self.num_workers = self.config.SOLVER.NUM_WORKERS

    def generate_and_save_indices(self, vq_vae, dataset, path, device):

        if os.path.exists(path):
            contents = os.listdir(path)
            if len(contents) == len(dataset) or len(contents) == len(dataset) + 1:
                print(f"Data already exists at {path}. Skipping data generation.")
                return
        # generate data
        print(f"Generating data at {path}...")
        os.makedirs(path, exist_ok=True)

        for i in tqdm(range(len(dataset))):
            x, y = dataset[i]
            with torch.no_grad():
                codebook_outputs = vq_vae.vq(vq_vae.pre_vq_conv(vq_vae.encoder(x.unsqueeze(0).to(device))))
            if self.config.DATA.MODE == "one_hot":
            
                encoding_indices = codebook_outputs.encoding_indices.squeeze().cpu().numpy()
                np.save("{}/{}.npy".format(path, i), encoding_indices)
            elif self.config.DATA.MODE == "vectors":
                quantized = codebook_outputs.quantized.squeeze()
                C, H, W = quantized.shape
                quantized = quantized.view(C, H * W).permute(1, 0).cpu().numpy()
                np.save("{}/{}.npy".format(path, i), quantized)
                    
            else:
                raise ValueError(f"Invalid data mode {self.config.DATA.MODE}")

    def prepare_data(self):
        
        # create the vq_vae model
        vq_vae_ckpt = torch.load(self.config.DATA.VQ_VAE_PATH)
        vq_vae_hparams = edict(vq_vae_ckpt["hyper_parameters"])
        vq_vae = VAEXperiment(vq_vae_hparams)
        vq_vae.load_state_dict(vq_vae_ckpt["state_dict"])
        vq_vae = vq_vae.model
        vq_vae.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vq_vae.to(device)

        codebook_path = os.path.join("./data", self.mode, vq_vae_hparams.WANDB.RUN_NAME)
        os.makedirs(codebook_path, exist_ok=True)
        np.save("{}/codebook.npy".format(codebook_path), vq_vae.vq._embedding.weight.detach().cpu().numpy())
        # path name to save data will be data/codebook/{mode}/{vq_vae_hparams.WANDB.NAME}/{dataset_name}

        # e.g. data/codebook/indices/codebook_512_32/CIFAR10 
        # if such a path exists and the number of files it contains is equal to the training size then
        # no need to generate the data
    
        self.data_path = os.path.join("./data", self.mode, vq_vae_hparams.WANDB.RUN_NAME, vq_vae_hparams.DATA.NAME)
        datamodule = DATAMODULES[vq_vae_hparams.DATA.NAME](vq_vae_hparams)
        datamodule.setup()
        train_dataset = datamodule.train_dataset

        # get the indices of the codebook
        self.generate_and_save_indices(vq_vae, train_dataset, self.data_path, device)
        
        DATASETS = get_datasets(self.config.DATA.ROOT)
        # SVHN
        svhn_dataset = DATASETS.SVHN
        svhn_path = os.path.join("./data", self.mode, vq_vae_hparams.WANDB.RUN_NAME, "SVHN")
        self.generate_and_save_indices(vq_vae, svhn_dataset, svhn_path, device)

        # CelebA
        celeba_dataset = DATASETS.CELEBA
        celeba_path = os.path.join("./data", self.mode, vq_vae_hparams.WANDB.RUN_NAME, "CelebA")
        self.generate_and_save_indices(vq_vae, celeba_dataset, celeba_path, device)

        # Omniglot
        omniglot_dataset = DATASETS.OMNIGLOT
        omniglot_path = os.path.join("./data", self.mode, vq_vae_hparams.WANDB.RUN_NAME, "Omniglot")
        self.generate_and_save_indices(vq_vae, omniglot_dataset, omniglot_path, device)

        # CIFAR10
        cifar10_dataset = DATASETS.CIFAR10
        cifar10_path = os.path.join("./data", self.mode, vq_vae_hparams.WANDB.RUN_NAME, "CIFAR10_test")
        self.generate_and_save_indices(vq_vae, cifar10_dataset, cifar10_path, device)

        self.id_val_path = cifar10_path
        self.ood_val_path = celeba_path
        self.codebook_size = vq_vae_hparams.MODEL.NUM_EMBEDDINGS

        assert self.codebook_size == self.config.MODEL.CODEBOOK_SIZE, "Codebook size mismatch between VQ-VAE and Config File"


        
    def setup(self, stage=None):
        
        self.train_dataset = CodebookDataset(
            root=self.data_path,
            ood_mode=self.config.DATA.OOD_MODE,
            ood_prob=self.config.DATA.OOD_PROB,
            codebook_size=self.codebook_size,
            num_tokens=self.config.MODEL.NUM_TOKENS,
            one_hot=self.config.DATA.MODE=="one_hot",
            corruption_params=self.config.DATA.CORRUPTION_PARAMS,
        )

        self.val_dataset = CodebookTestDataset(
            id_root=self.id_val_path,
            ood_root=self.ood_val_path,
            max_len=self.config.DATA.MAX_TEST_LEN,
            num_tokens=self.config.MODEL.NUM_TOKENS,
            one_hot=self.config.DATA.MODE=="one_hot",
            codebook_size=self.codebook_size
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