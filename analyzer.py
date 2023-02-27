import os
import argparse
import warnings
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from easydict import EasyDict as edict
from experiment import VAEXperiment
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# Extractable Plots
# - Reconstruction Error Density
# - KMeans + TSNE
# - Utilization of Codebooks
# - Density of Latent <-> kth closest latent code

parser = argparse.ArgumentParser(description="Yo it's Analyzer")

parser.add_argument('--models_folder', type=str,
                    help="path to folder containing at least 1 folder. Inside each is a lightning ckpt")
parser.add_argument('--ckpt_name', type=str, default='last',
                    help="name of lightning ckpt file")

parser.add_argument('--id_dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'SVHN'],
                    help="name of ID dataset")
parser.add_argument('--ood_datasets', nargs="*", type=str, default=["CIFAR100", "SVHN"],
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
parser.add_argument("--codebook_k", nargs="*", type=int, default=[1],
                    help="the k values used when extracting the distances to codebooks")
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

# Creating Datasets
DATASETS = edict(
    CIFAR10=CIFAR10(root=args.datasets_root, train=False,
                    download=True, transform=transform),
    CIFAR100=CIFAR100(root=args.datasets_root, train=False,
                      download=True, transform=transform),
    SVHN=SVHN(root=args.datasets_root, split='test',
              download=True, transform=transform)
)

# setting device
if args.device == 'cuda' and (not torch.cuda.is_available()):
    warnings.warn(
        "cuda is not available. cpu will be used. Get a GPU you poor person")
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
        result[k] = torch.cat(v, dim=0)[:args.max_samples]

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


def calculate_recon_error(dataset):

    x = dataset.outputs.x
    
    recon = dataset.outputs.recon
    num_samples = x.shape[0]
    id_recon_error = F.mse_loss(
        x.reshape(num_samples, -1),
        recon.reshape(num_samples, -1),
        reduction='none'
    ).mean(dim=1)  # -> (num_samples,)

    return id_recon_error


def extract_reconstruction_error_density(id_dataset, ood_datasets, output_folder):
    """
    Extracts reconstruction errors from a single ID dataset and multiple OoD Datasets
    and plot the density using KDE, the plots are saved in $output_folder

    Input Assumptions:
    - id_dataset: an edict with entries $name, $dataset, $outputs

    The $outputs entry of dataset will be mainly used and will need to have a $recon and $x entry
    """

    id_recon_error = calculate_recon_error(id_dataset)
    ood_recon_errors = [calculate_recon_error(
        dataset) for dataset in ood_datasets]

    df = pd.DataFrame()
    df["error"] = torch.cat([id_recon_error] + ood_recon_errors, dim=0)

    labels = [f"ID Recon Error ({id_dataset.name})"] * id_recon_error.shape[0]
    for i in range(len(ood_datasets)):
        labels += [f"OoD Recon Loss ({ood_datasets[i].name})"] * \
            ood_recon_errors[i].shape[0]

    df["label"] = labels
    plt.figure(figsize=(8, 5))
    ax = sns.kdeplot(
        data=df,
        x="error",
        hue="label",
        fill=True,
        common_norm=False,
        bw_adjust=1
    )
    plt.savefig(os.path.join(output_folder, "recon_density.png"))


def extract_tsne_with_kmeans(id_dataset, ood_datasets, output_folder):
    """
    Applies k-means where k is equal to the total number of datasets used (ID + OoD)
    and creates two plots:
    1- showing K-Means Assignments on 2D-TSNE
    2- showing dataset membership assignment on 2D-TSNE
    """

    quantized_id = id_dataset.outputs.quantized.squeeze()
    N, C, H, W = quantized_id.shape
    quantized_id = quantized_id.view(N, C, H * W).mean(dim=2)
    quantized_ood = [dataset.outputs.quantized.view(N, C, H * W).mean(dim=2) for dataset in ood_datasets]
    
    all_vectors = torch.cat([quantized_id] + quantized_ood, dim=0)

    clusters = apply_kmeans(all_vectors, n_clusters=len(ood_datasets) + 1)
    tsne_rep = get_tsne(all_vectors, n_components=2)

    # NOTE: I can make this an args param but seems overkill to me don't know y
    alpha = 0.5 
    fig = plt.figure(constrained_layout=True, figsize=(12, 4))
    
    ax = fig.subplot_mosaic([["clusters", "dataset"]])

    for i in range(len(ood_datasets) + 1):
        ax["clusters"].scatter(tsne_rep[clusters == i, 0], tsne_rep[clusters == i, 1], color=get_color(), label=f'C{i+1}', alpha=alpha)
    ax["clusters"].set_title("k-means clustering")
    ax["clusters"].legend()

    last = 0
    for i, d in enumerate([id_dataset] + ood_datasets): 

        current_idx = last + len(d.data)
        label = 'ID ' if i == 0 else "OoD "
        label += f"({d.name})"
        ax["dataset"].scatter(tsne_rep[last:current_idx, 0], tsne_rep[last:current_idx, 1], color=get_color(), label=label, alpha=alpha) 
        
        last = current_idx

    ax["dataset"].set_title("Dataset Assignment")
    ax["dataset"].legend()

    plt.savefig(os.path.join(output_folder, "tsne_kmeans.png"))


def get_kth_codebook_distances(dataset, k):
    
    kth_distance = dataset.outputs.distances.kthvalue(dim=2, k=k).values.mean(dim=1)

    return kth_distance


def extract_codebook_distances(k, id_dataset, ood_datasets, output_folder):
    """
    After encoding the input image into a latent representation (N, C, H, W),
    each (H x W) embedding is matched with a discrete codebook which is closest
    to it after computing the distance to all of the codebooks. 
    
    This function compute the density of distances between latents and the kth 
    closest latent codebook between different datasets. Distances are averaged
    for the spatial dimensions (H, W) so that each image is represented with
    a single scalar.

    Expected Special Inputs
    - k: int or List, denotes the order of cloesest codebook for which distance
        densities are to be extracted.
    """

    if isinstance(k, int):
        k = [k]
    elif not isinstance(k, list):
        raise ValueError(f"K should be a number of a list. Focus!!, given was {type(k)}")
    
    for kth in k:
        
        if kth > id_dataset.outputs.distances.shape[2]:
            continue

        id_distances = get_kth_codebook_distances(id_dataset, kth)
        ood_distances = [get_kth_codebook_distances(dataset, kth) for dataset in ood_datasets]

        df = pd.DataFrame()
        df["value"] = torch.cat([id_distances] + ood_distances, dim=0)
        label = [f"ID ({id_dataset.name})"] * id_distances.shape[0]
        for i, dataset in enumerate(ood_datasets):
            label += [f"OoD ({dataset.name})"] * ood_distances[i].shape[0]
        
        df["label"] = label
        plt.figure(figsize=(8, 5))
        ax = sns.kdeplot(
            data=df,
            x="value",
            hue="label",
            fill=True,
            common_norm=False,
            bw_adjust=1
        )
        
        plt.savefig(os.path.join(output_folder, f"codebook_distance_to_{kth}.png"))
    


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
        ckpt = torch.load(model_path)
        model = VAEXperiment(edict(ckpt["hyper_parameters"]))
        model.load_state_dict(ckpt['state_dict'])
        _ = model.eval()
        _ = model.to(DEVICE)

        # create paths to store visualizations
        output_folder = os.path.join(args.output_folder, model_name)
        Path(output_folder).mkdir(exist_ok=True, parents=True)
        
        # extract outputs from ID dataset:
        id_dataset.outputs = extract_outputs(model, id_dataset)
        

        # extract outputs from OoD Datasets
        for ood_dataset in ood_datasets:
            ood_dataset.outputs = extract_outputs(model, ood_dataset)

        # extract_reconstruction_error_density(
        #     id_dataset=id_dataset,
        #     ood_datasets=ood_datasets,
        #     output_folder=output_folder
        # )

        # extract_tsne_with_kmeans(
        #     id_dataset=id_dataset,
        #     ood_datasets=ood_datasets,
        #     output_folder=output_folder
        # )
        extract_codebook_distances(
            k=args.codebook_k,
            id_dataset=id_dataset,
            ood_datasets=ood_datasets,
            output_folder=output_folder
        )


if __name__ == "__main__":
    main()
