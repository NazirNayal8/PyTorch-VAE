from .vae.base import *
from .vae.vanilla_vae import *
from .vae.gamma_vae import *
from .vae.beta_vae import *
from .vae.wae_mmd import *
from .vae.cvae import *
from .vae.hvae import *
from .vae.vampvae import *
from .vae.iwae import *
from .vae.dfcvae import *
from .vae.mssim_vae import MSSIMVAE
from .vae.fvae import *
from .vae.cat_vae import *
from .vae.joint_vae import *
from .vae.info_vae import *
# from .twostage_vae import *
from .vae.lvae import LVAE
from .vae.logcosh_vae import *
from .vae.swae import *
from .vae.miwae import *
from .vq_vae import *
from .vq_vae_v2 import *
from .vae.betatc_vae import *
from .vae.dip_vae import *

from .pixel_cnn import GatedPixelCNN


# Aliases
VAE = VanillaVAE
GaussianVAE = VanillaVAE
CVAE = ConditionalVAE
GumbelVAE = CategoricalVAE

vae_models = {
    'HVAE': HVAE,
    'LVAE': LVAE,
    'IWAE': IWAE,
    'SWAE': SWAE,
    'MIWAE': MIWAE,
    'VQVAE': VQVAE,
    'VQVAE_V2': VQVAE_V2,
    'DFCVAE': DFCVAE,
    'DIPVAE': DIPVAE,
    'BetaVAE': BetaVAE,
    'InfoVAE': InfoVAE,
    'WAE_MMD': WAE_MMD,
    'VampVAE': VampVAE,
    'GammaVAE': GammaVAE,
    'MSSIMVAE': MSSIMVAE,
    'JointVAE': JointVAE,
    'BetaTCVAE': BetaTCVAE,
    'FactorVAE': FactorVAE,
    'LogCoshVAE': LogCoshVAE,
    'VanillaVAE': VanillaVAE,
    'ConditionalVAE': ConditionalVAE,
    'CategoricalVAE': CategoricalVAE
}

PRIORS = edict(
    PixelCNN=GatedPixelCNN
)
