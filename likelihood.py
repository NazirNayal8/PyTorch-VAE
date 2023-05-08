import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader, Subset


def process_input(x, y, device):
    x = x.to(device)

    if x.ndim == 3:
        x = x.unsqueeze(0)
    y = torch.tensor(y).cuda().unsqueeze(0)

    return x, y


@torch.no_grad()
def compute_log_likelihood(prior, x, y):
    # write a description for this function
    """
    Args:   
        prior: PixelCNN prior
        x: input image
        y: label

    Returns:
        likelihood: log likelihood of the input image
    """

    x, y = process_input(x, y, prior.device)

    vq_vae = prior.vq_vae

    codebook_outputs = vq_vae.vq(
        vq_vae.pre_vq_conv(vq_vae.encoder(x)))

    B, C, H, W = codebook_outputs.quantized.shape

    encoding_indices = codebook_outputs.encoding_indices[:, 0].long()
    encoding_indices = encoding_indices.view(B, H, W)

    # get the prior
    logits = prior(encoding_indices, y)  # shape: B, C, H, W
    logits = logits.permute(0, 2, 3, 1).contiguous()  # shape: B, H, W, C
    log_probs = logits.log_softmax(dim=-1)
    B, H, W, C = log_probs.shape

    log_probs = log_probs.reshape(B * H * W, C)

    likelihood = log_probs.gather(
        dim=1, index=encoding_indices.view(-1).unsqueeze(1)).sum(dim=0)

    return -likelihood


@torch.no_grad()
def compute_log_likelihood_v2(prior, x, y):
    # write a description for this function
    """
    Args:   
        prior: PixelCNN prior
        x: input image
        y: label

    Returns:
        likelihood: log likelihood of the input image
    """

    x, y = process_input(x, y, prior.device)

    vq_vae = prior.vq_vae
    codebook_outputs = vq_vae.vq(
        vq_vae.pre_vq_conv(vq_vae.encoder(x)))

    B, C, H, W = codebook_outputs.quantized.shape

    encoding_indices = codebook_outputs.encoding_indices[:, 0].long()
    encoding_indices = encoding_indices.view(B, H, W)

    # get the prior
    logits = prior(encoding_indices, y)  # shape: B, C, H, W
    logits = logits.permute(0, 2, 3, 1).contiguous()  # shape: B, H, W, C

    likelihood = F.cross_entropy(
        logits.view(-1, prior.hparams.MODEL.INPUT_DIM), encoding_indices.view(-1))

    return likelihood


def compute_dataset_log_likelihood(prior, dataset, max_samples=10000, arbitrary_cls=None):

    num_samples = min(len(dataset), max_samples)
    likelihoods = np.zeros(num_samples)
    for i in tqdm(range(num_samples)):

        x, y = dataset[i]

        # this condition is used for when we want to compute the likelihood of an
        # OoD sample but with choosing an explicit class to condition on.
        if arbitrary_cls is not None:
            assert isinstance(
                arbitrary_cls, int), "arbitrary_cls must be an integer"
            y = arbitrary_cls
        likelihoods[i] = compute_log_likelihood(prior, x, y).cpu().item()

    return likelihoods


def compute_dataset_log_likelihood_batched(prior, dataset, batch_size=1, num_workers=4, max_samples=10000, arbitrary_cls=None):


    num_samples = min(len(dataset), max_samples)
    dataloader = DataLoader(Subset(dataset, np.arange(num_samples)), batch_size=batch_size,
                            shuffle=False, num_workers=4)
    likelihoods = []
    for i, (x, y) in enumerate(tqdm(dataloader)):
 
        # this condition is used for when we want to compute the likelihood of an
        # OoD sample but with choosing an explicit class to condition on.
        if arbitrary_cls is not None:
            assert isinstance(
                arbitrary_cls, int), "arbitrary_cls must be an integer"
            y = arbitrary_cls
        likelihoods.extend([compute_log_likelihood(prior, x, y).view(-1).cpu().numpy()])
    likelihoods = np.concatenate(likelihoods)
    return likelihoods


@torch.no_grad()
def get_perplexity(prior, x, y):

    x, y = process_input(x, y, prior.device)

    vq_vae = prior.vq_vae
    codebook_outputs = vq_vae(x)

    return codebook_outputs.perplexity    
