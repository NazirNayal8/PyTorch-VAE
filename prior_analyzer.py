import os
import argparse
import warnings
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, Omniglot, CelebA
from easydict import EasyDict as edict
from experiment import VAEXperiment
from prior import VQVAEPrior
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from likelihood import compute_dataset_log_likelihood

# Extractable Plots
# - Reconstruction Error Density
# - KMeans + TSNE
# - Utilization of Codebooks
# - Density of Latent <-> kth closest latent code

parser = argparse.ArgumentParser(description="Yo it's Prior Analyzer")

parser.add_argument('--models_folder', type=str,
                    help="path to folder containing at least 1 folder. Inside each is a lightning ckpt")
parser.add_argument('--ckpt_name', type=str, default='last',
                    help="name of lightning ckpt file")

parser.add_argument('--id_dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'SVHN'],
                    help="name of ID dataset")
parser.add_argument('--ood_datasets', nargs="*", type=str, default=["OMNIGLOT", "SVHN", "CELEBA"],
                    help="name of datasets to be considered as OOD")

parser.add_argument("--output_folder", type=str, default="visualizations")

parser.add_argument("--datasets_root", type=str, default="/home/nazir/datasets",
                    help="path to folder containing datasets")

parser.add_argument("--data_norm", type=str, default='VQ_VAE', choices=["VQ_VAE"],
                    help="""
                    choose the mean and std values used to normalize the datasets currently
                    available options are:
                    VQ_VAE: mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0]
                    """)
parser.add_argument("--max_samples", type=int, default=10000,
                    help="An upper limit on the number of samples to be considered from each dataset")
parser.add_argument("--device", type=str, default="cuda",
                    help="either cpu or cuda")

args = parser.parse_args()


# some colors for plotting
global COLOR_IDX
COLOR_IDX = 0
COLORS = ["red", "blue", "orange", "green", "purple", "yellow"]


def get_color():
    global COLOR_IDX
    c = COLORS[COLOR_IDX]
    COLOR_IDX = (COLOR_IDX + 1) % len(COLORS)
    return c


# Dataset Transformations
DATA_NORM_METRICS = edict(
    VQ_VAE=([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
)

DATA_MEAN, DATA_STD = DATA_NORM_METRICS[args.data_norm]
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=DATA_MEAN, std=DATA_STD)
])

transform_omniglot = T.Compose([
    T.ToTensor(),
    T.Lambda(lambda x: x.repeat(3, 1, 1)),
    T.Resize((32, 32)),
    T.Normalize(mean=DATA_MEAN, std=DATA_STD)
])

transform_celeba = T.Compose([
    T.ToTensor(),
    T.Resize((32, 32)),
    T.Normalize(mean=DATA_MEAN, std=DATA_STD)
])

# Creating Datasets
DATASETS = edict(
    CIFAR10=CIFAR10(root=args.datasets_root, train=False,
                    download=True, transform=transform),
    CIFAR100=CIFAR100(root=args.datasets_root, train=False,
                      download=True, transform=transform),
    SVHN=SVHN(root=args.datasets_root, split='test',
              download=True, transform=transform),
    OMNIGLOT=Omniglot(root=args.datasets_root, background=False, download=True,
                      transform=transform_omniglot),
    CELEBA=CelebA(root=args.datasets_root, split='valid', target_type='identity', download=True,
                  transform=transform_celeba)
)

# setting device
if args.device == 'cuda' and (not torch.cuda.is_available()):
    warnings.warn(
        "cuda is not available. cpu will be used. Get a GPU you poor peasant")
    args.device = 'cpu'

DEVICE = torch.device(args.device)


def apply_kmeans(data, n_clusters, max_iter=300):

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=max_iter)
    clusters = kmeans.fit_predict(data)

    return clusters


def get_tsne(features, n_components=2):

    tsne = TSNE(n_components=n_components, random_state=0)
    tsne_features = tsne.fit_transform(features)

    return tsne_features


def process_list_to_dict(lst):
    """
    Transform a List[Dict[str -> Tensors]] to Dict[str -> ConcatenatedTensors]

    - Assuming that Dict keys are the same across all the elements of List.
    - If tensors are stored in gpu they are moved to CPU
    """

    result_tmp = edict()
    for k in lst[0].keys():
        result_tmp[k] = []

    for x in lst:
        for k, v in x.items():
            tensor = v
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.cpu().numpy()
            result_tmp[k].extend([v.unsqueeze(0)])

    result = edict()
    for k, v in result_tmp.items():
        result[k] = torch.cat(v, dim=0)#[:args.max_samples]

    return result


def extract_outputs(model, dataset):
    """
    Extract the outputs of a dataset using a model in the form of Dict[str -> Tensors(N,...)]
    dataset is assumed to be an edict with 2 entries:
    - name: name of dataset
    - data: containing the Dataset Instance
    """

    all_outputs = []
    for i in tqdm(range(min(len(dataset.data), args.max_samples)), desc=f"Processing ID Dataset {dataset.name}"):

        x, y = dataset.data[i]

        with torch.no_grad():
            outputs = model(x.unsqueeze(0).to(DEVICE))
        outputs_cpu = edict()
        for k, v in outputs.items():
            outputs_cpu[k] = v.cpu()
        all_outputs.extend([outputs_cpu])

    all_outputs = process_list_to_dict(all_outputs)

    return all_outputs

def plot_log_likelihood_densities(prior, id_dataset, ood_datasets, output_folder):

    id_likelihoods = compute_dataset_log_likelihood(prior, id_dataset.data, max_samples=args.max_samples)
    ood_likelihoods = [
        compute_dataset_log_likelihood(
            prior, 
            ood_dataset.data, 
            max_samples=args.max_samples,
            arbitrary_cls=0
        ) for ood_dataset in ood_datasets
    ]

    # plot densities
    plt.figure(figsize=(10, 5))
    plt.hist(id_likelihoods, bins=100, alpha=0.5, label=id_dataset.name)
    for i in range(len(ood_datasets)):
        plt.hist(ood_likelihoods[i], bins=100, alpha=0.5, label=ood_datasets[i].name)
    plt.xlabel('log-likelihood')
    plt.ylabel('Density')
    plt.legend()
    plt.title("Log-likelihood densities")
    plt.savefig(os.path.join(output_folder, "log_likelihood_densities.png"))

def main():

    assert os.path.exists(
        args.models_folder), "Models folder not here y u liar"
    model_names = os.listdir(args.models_folder)

    # setup datasets
    id_dataset = edict(
        name=args.id_dataset,
        data=DATASETS[args.id_dataset]
    )
    ood_datasets = [edict(name=name, data=DATASETS[name])
                    for name in args.ood_datasets]

    for model_name in model_names:

        print(f"Processing {model_name} ...")

        model_path = os.path.join(
            args.models_folder, model_name, f'{args.ckpt_name}.ckpt')

        if not os.path.exists(model_path):
            warnings.warn(
                f"ckpt path is fake, are you tricking me! Skipping this model (sigh): {model_path}")
            continue

        # load za model

        prior_ckpt = torch.load(model_path, map_location="cpu")
        prior_hparams = edict(prior_ckpt["hyper_parameters"])
        prior = VQVAEPrior(prior_hparams)
        prior.load_state_dict(prior_ckpt["state_dict"])

        _ = prior.eval()
        _ = prior.to(DEVICE)

        model = prior.vq_vae
        # create paths to store visualizations
        output_folder = os.path.join(args.output_folder, model_name)
        Path(output_folder).mkdir(exist_ok=True, parents=True)  

        # # extract outputs from ID dataset:
        # id_dataset.outputs = extract_outputs(model, id_dataset)

        # # extract outputs from OoD Datasets
        # for ood_dataset in ood_datasets:
        #     ood_dataset.outputs = extract_outputs(model, ood_dataset)

        # plot log-likelihood densities
        plot_log_likelihood_densities(prior, id_dataset, ood_datasets, output_folder)

        

if __name__ == "__main__":
    main()
