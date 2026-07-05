"""
VQ-VAE model for sprite discrete latent encoding.
Encoder -> Vector Quantization -> Decoder.
Input: 32x32 RGBA sprites -> Output: 8x8 discrete token grid.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.block(x))


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 4, hidden_dim: int = 128, latent_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1)
        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)
        self.conv_out = nn.Conv2d(hidden_dim, latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.conv_out(x)
        return x  # (B, latent_dim, H/4, W/4)


class Decoder(nn.Module):
    def __init__(self, out_channels: int = 4, hidden_dim: int = 128, latent_dim: int = 64):
        super().__init__()
        self.conv_in = nn.Conv2d(latent_dim, hidden_dim, 1)
        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)
        self.deconv1 = nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, out_channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.res1(x)
        x = self.res2(x)
        x = F.relu(self.deconv1(x))
        x = self.deconv2(x)
        return x  # (B, 4, H, W) raw output


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int = 256, embedding_dim: int = 64, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor) -> tuple:
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)

        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        quantized = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized.view_as(z)

        vq_loss = F.mse_loss(quantized, z.detach()) + self.commitment_cost * F.mse_loss(quantized.detach(), z)

        quantized = z + (quantized - z).detach()

        return quantized, vq_loss, encoding_indices

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(indices)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_embeddings: int = 256,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dim, latent_dim)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = Decoder(in_channels, hidden_dim, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> tuple:
        z = self.encoder(x)
        quantized, vq_loss, indices = self.quantizer(z)
        recon = self.decoder(quantized)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + vq_loss
        return {
            "recon": recon,
            "loss": loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "indices": indices.view(x.size(0), -1),
            "quantized": quantized,
        }

    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        _, _, indices = self.quantizer(z)
        return indices.view(x.size(0), -1)

    def decode_from_indices(self, indices: torch.Tensor, latent_shape: tuple) -> torch.Tensor:
        z = self.quantizer.get_codebook_entry(indices)
        z = z.view(-1, *latent_shape)
        return self.decoder(z)
