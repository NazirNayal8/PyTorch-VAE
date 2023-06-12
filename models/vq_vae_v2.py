import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .vae.types_ import *
from .vq_vae import VectorQuantizer
from easydict import EasyDict as edict


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(
            self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._beta = beta

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(
            num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)  # [BHW, C]

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)  # [BHW, 1]
                     # [K,] K: num of codebooks
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))  # [BHW, K]

        # Encoding
        encoding_indices = torch.argmin(
            distances, dim=1).unsqueeze(1)  # [BHW, 1]

        # A one hot representation of indices
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device)  # [BHW, K]
        encodings.scatter_(1, encoding_indices, 1)
        # print(encodings.shape)

        # Quantize and unflatten
        quantized = torch.matmul(
            encodings, self._embedding.weight).view(input_shape)  # [BHW, K] x [K, C] -> [BHW, C] -> [B, H, W, C]

        # Use EMA to update the embedding vectors
        if self.training:
            # for each codebook entry, count how many of the BHW entries it was closest to
            # learn this count in a moving average manner
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                (1 - self._decay) * torch.sum(encodings, 0)  # [K]

            # Laplace smoothing of the cluster size
            # This is used to avoid 0 cluster sizes.
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)  # [K]

            # [K, BHW] x [BHW, C] -> [K, C]
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._beta * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()  # shape of quantized is [B, H, W, C]
        avg_probs = torch.mean(encodings, dim=0) # shape of the avg_probs tensor is [K]
        
        # shape of the perplexity tensor is []
        perplexity = torch.exp(-torch.sum(avg_probs *
                               torch.log(avg_probs + 1e-10)))  

        # convert quantized from BHWC -> BCHW
        return edict(
            loss=loss,
            quantized=quantized.permute(0, 3, 1, 2).contiguous(),
            encodings=encodings,
            encoding_indices=encoding_indices,
            perplexity=perplexity,
            distances=distances
        )


class ResidualLayer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_hiddens: int
    ):
        super(ResidualLayer, self).__init__()
        # NOTE: In reference code a ReLU is added before first CONV
        self.resblock = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels, num_hiddens,
                      kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(num_hiddens, out_channels,
                      kernel_size=1, bias=False)
        )

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_residual_layers, residual_hidden_dim):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([
            ResidualLayer(in_channels, hidden_dim, residual_hidden_dim)
            for _ in range(self._num_residual_layers)
        ])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class EncoderFlex(nn.Module):
    """
    Encoder for VQ-VAE with flexible number of layers

    Args:
        in_channels (int): Number of input channels
        hidden_dim (list): List of hidden dimensions
        num_residual_layers (int): Number of residual layers
        residual_hidden_dim (int): Number of hidden dimensions for residual layers
    """

    def __init__(self, in_channels, hidden_dim, num_residual_layers, residual_hidden_dim):
        super(EncoderFlex, self).__init__()

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_dim[0],
            kernel_size=4,
            stride=2, padding=1
        )
        self._conv_2 = nn.Conv2d(
            in_channels=hidden_dim[0],
            out_channels=hidden_dim[1],
            kernel_size=4,
            stride=2, padding=1
        )
        self._conv_3 = nn.Conv2d(
            in_channels=hidden_dim[1],
            out_channels=hidden_dim[2],
            kernel_size=4,
            stride=2,
            padding=1
        )
        self._residual_stack = ResidualStack(
            in_channels=hidden_dim[2],
            hidden_dim=hidden_dim[2],
            num_residual_layers=num_residual_layers,
            residual_hidden_dim=residual_hidden_dim
        )

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)


class Encoder(nn.Module):
    """
    Encoder network for VQ-VAE.

    Args:
        in_channels: Number of channels in the input tensor.
        hidden_dim: Number of channels in the hidden activations.
        num_residual_layers: Number of residual layers in the encoder.
        residual_hidden_dim: Number of channels in the residual hidden
            activations.

    Returns:
        A torch.Tensor of shape [batch_size, hidden_dim, height // 2,
            width // 2].
    """

    def __init__(self, in_channels, hidden_dim, num_residual_layers, residual_hidden_dim):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_dim//2,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self._conv_2 = nn.Conv2d(
            in_channels=hidden_dim//2,
            out_channels=hidden_dim,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self._conv_3 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self._residual_stack = ResidualStack(
            in_channels=hidden_dim,
            hidden_dim=hidden_dim,
            num_residual_layers=num_residual_layers,
            residual_hidden_dim=residual_hidden_dim
        )

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)


class DecoderFlex(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_residual_layers, residual_hidden_dim, attention=False, attention_params=None):
        super(DecoderFlex, self).__init__()

        self._conv_1 =nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_dim[2],
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        self._residual_stack = ResidualStack(
            in_channels=hidden_dim[2],
            hidden_dim=hidden_dim[2],
            num_residual_layers=num_residual_layers,
            residual_hidden_dim=residual_hidden_dim
        )

        self._conv_trans_0 = nn.Sequential(
             nn.ConvTranspose2d(
                in_channels=hidden_dim[2],
                out_channels=hidden_dim[1],
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),
        )
        
        self._conv_trans_1 = nn.ConvTranspose2d(
            in_channels=hidden_dim[1],
            out_channels=hidden_dim[0],
            kernel_size=4,
            stride=2,
            padding=1
        )

        self._conv_trans_2 = nn.ConvTranspose2d(
            in_channels=hidden_dim[0],
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.attention = attention

        if attention:
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=hidden_dim[2],
                num_heads=attention_params.NUM_HEADS,
                dropout=attention_params.DROPOUT,
                batch_first=True
            )

    def forward(self, inputs):
        
        
        x = self._conv_1(inputs)

        # shape of x is: [batch_size, hidden_dim, height, width]

        if self.attention:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).view(B, H*W, C)
            x = self.attention_layer(x, x, x)[0]
            x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        x = self._residual_stack(x)

        x = self._conv_trans_0(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_residual_layers, residual_hidden_dim, attention=False, attention_params=None):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self._residual_stack = ResidualStack(
            in_channels=hidden_dim,
            hidden_dim=hidden_dim,
            num_residual_layers=num_residual_layers,
            residual_hidden_dim=residual_hidden_dim
        )
        self._conv_trans_1 = nn.ConvTranspose2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim//2,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self._conv_trans_2 = nn.ConvTranspose2d(
            in_channels=hidden_dim//2,
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.attention = attention

        if attention:
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=attention_params.NUM_HEADS,
                dropout=attention_params.DROPOUT,
                batch_first=True
            )

    def forward(self, inputs):
        x = self._conv_1(inputs)

        if self.attention:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).view(B, H*W, C)
            x = self.attention_layer(x, x, x)[0]
            x = x.view(B, H, W, C).permute(0, 3, 1, 2)
            x = F.relu(x)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)


class EncoderPretrained(nn.Module):


    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        if hparams.NAME == "dino-v2-vit-s":
            self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
        else:
            raise ValueError(f"Unknown model name: {hparams.NAME}")
        
        self.avgpool = nn.AvgPool1d(kernel_size=hparams.KERNEL_SIZE, stride=hparams.STRIDE)

    def forward(self, inputs):
        """
        Args:
            inputs: Tensor of shape [batch_size, channels, height, width]

        Returns:
            Tensor of shape []
        """

        x = self.encoder(inputs)  # shape: [batch_size, channels, height, width]
        x = self.avgpool(x.permute(2, 1))
        x = x.permute(2, 1)
        return x


class VQVAE_V2(BaseVAE):

    def __init__(
        self,
        in_channels: int,
        embedding_dim: int,
        num_embeddings: int,
        hidden_dim: int,
        beta: float,
        num_residual_layers: int,
        residual_hidden_dim: int,
        decay: float,
        data_variance: float,
        attention: bool = False,
        attention_params: dict = None,
        use_pretrained_encoder: bool = False,
        pretrained_encoder_params: dict = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.num_residual_layers = num_residual_layers
        self.residual_hidden_dim = residual_hidden_dim
        self.data_variance = data_variance
        self.attention = attention
        self.attention_params = attention_params
        self.use_pretrained_encoder = use_pretrained_encoder
        self.pretrained_encoder_params = pretrained_encoder_params

        if isinstance(hidden_dim, int):
            EncoderM = Encoder
            DecoderM = Decoder
        elif isinstance(hidden_dim, list):
            EncoderM = EncoderFlex
            DecoderM = DecoderFlex

        if use_pretrained_encoder:
            EncoderM = EncoderPretrained(pretrained_encoder_params)

        self.encoder = EncoderM(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_residual_layers=num_residual_layers,
            residual_hidden_dim=residual_hidden_dim
        )

        self.pre_vq_conv = nn.Conv2d(
            in_channels=hidden_dim if isinstance(hidden_dim, int) else hidden_dim[-1],
            out_channels=embedding_dim,
            kernel_size=1,
            stride=1
        )

        if decay > 0.0:
            self.vq = VectorQuantizerEMA(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                beta=beta,
                decay=decay
            )
        else:
            self.vq = VectorQuantizer(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                beta=beta
            )

        self.decoder = DecoderM(
            in_channels=embedding_dim,
            hidden_dim=hidden_dim,
            num_residual_layers=num_residual_layers,
            residual_hidden_dim=residual_hidden_dim,
            attention=attention,
            attention_params=attention_params
        )

    def encode(self, input: Tensor) -> List[Tensor]:

        result = self.encoder(input)
        return [result]

    def decode(self, z: Tensor) -> Tensor:

        result = self.decoder(z)
        return result

    def forward(self, x: Tensor, **kwargs) -> List[Tensor]:

        encoding = self.encode(x)[0]

        encoding = self.pre_vq_conv(encoding)

        vq_outputs = self.vq(encoding)

        recon = self.decode(vq_outputs.quantized)

        return edict(
            x=x,
            recon=recon,
            vq_loss=vq_outputs.loss,
            quantized=vq_outputs.quantized,
            perplexity=vq_outputs.perplexity,
            encodings=vq_outputs.encodings,
            encoding_indices=vq_outputs.encoding_indices,
            distances=vq_outputs.distances
        )

    def loss_function(
        self,
        recons,
        x,
        vq_loss,
        *args,
        **kwargs,
    ):

        recon_loss = F.mse_loss(recons, x) / self.data_variance

        loss = recon_loss + vq_loss
        return edict(
            loss=loss,
            reconstruction_loss=recon_loss,
            vq_loss=vq_loss
        )

    def sample(
        self,
        num_samples: int,
        current_device: Union[int, str],
        **kwargs
    ) -> Tensor:
        raise Warning('VQVAE sampler is not implemented.')

    def generate(self, x: Tensor, **kwargs) -> Tensor:

        return self.forward(x).recon
