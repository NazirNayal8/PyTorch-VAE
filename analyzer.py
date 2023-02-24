import os
import argparse
import warnings

from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from easydict import EasyDict as edict

# Extractable Plots
# - Reconstruction Error Density
# - KMeans + TSNE
# - Utilization of Codebooks
# - Density of Latent <-> kth closest latent code






def main(args):
    
    assert os.path.exists(args.models_folder), "Models folder not here y u liar"
    model_names = os.listdir(args.models_folder)
    DATASETS = edict(
        CIFAR10=CIFAR10()
    )

    






if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Yo it's Analyzer")

    parser.add_argument('--models_folder', type=str, 
                        help="path to folder containing at least 1 folder. Inside each is a lightning ckpt")
    parser.add_argument('--ckpt_name', type=str, default='last',
                        help="name of lightning ckpt file")
    parser.add_argument("--datasets_root", type=str, default="/home/nazir/datasets",
                        help="path to folder containing datasets")