import torchvision.transforms as T
from easydict import EasyDict as edict 
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, CelebA, Omniglot
from copy import deepcopy


# Dataset Transformations
DATA_NORM_METRICS = edict(
    VQ_VAE=([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
    IMAGENET=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
)

DATA_MEAN, DATA_STD = DATA_NORM_METRICS["VQ_VAE"]

TRANSFORMS = edict(
    CIFAR10=T.Compose([
        T.ToTensor(),
        T.Normalize(mean=DATA_MEAN, std=DATA_STD)
    ]),
    OMNIGLOT=T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1)),
        T.Resize((32, 32)),
        T.Normalize(mean=DATA_MEAN, std=DATA_STD)
    ]),
    CELEBA=T.Compose([
        T.ToTensor(),
        T.Resize((32, 32)),
        T.Normalize(mean=DATA_MEAN, std=DATA_STD)
    ])
)

# Creating Datasets

def get_datasets(root, transforms=None):

    T = deepcopy(TRANSFORMS)
    
    if transforms is not None:
        T.update(transforms)

    DATASETS = edict(
        CIFAR10_TRAIN=CIFAR10(root=root, train=True, 
                        download=True, transform=T.CIFAR10),
        CIFAR10=CIFAR10(root=root, train=False,
                        download=True, transform=T.CIFAR10),
        CIFAR100=CIFAR100(root=root, train=False,
                        download=True, transform=T.CIFAR10),
        SVHN=SVHN(root=root, split='test',
                download=True, transform=T.CIFAR10),
        OMNIGLOT=Omniglot(root=root, background=False, download=True,
                        transform=T.OMNIGLOT),
        CELEBA=CelebA(root=root, split='valid', target_type='identity', download=True,
                    transform=T.CELEBA)
    )

    return DATASETS