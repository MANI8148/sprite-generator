import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _groupnorm(ch, max_groups=32):
    num_groups = min(max_groups, ch // 4) if ch >= 8 else 1
    num_groups = max(1, num_groups)
    while ch % num_groups != 0:
        num_groups -= 1
    if num_groups < 1:
        num_groups = 1
    return nn.GroupNorm(num_groups, ch)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            _groupnorm(ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            _groupnorm(ch),
        )

    def forward(self, x):
        return F.relu(x + self.block(x))


class SelfAttention2d(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.flatten(2).transpose(-2, -1).float()
        k = k.flatten(2).transpose(-2, -1).float()
        v = v.flatten(2).transpose(-2, -1).float()
        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(-2, -1).reshape(B, C, H, W)
        return self.proj(out.to(x.dtype))


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 4, hidden_dim: int = 256, latent_dim: int = 96):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim // 2, 4, stride=2, padding=1)
        self.norm1 = _groupnorm(hidden_dim // 2)
        self.conv2 = nn.Conv2d(hidden_dim // 2, hidden_dim, 4, stride=2, padding=1)
        self.norm2 = _groupnorm(hidden_dim)
        self.res1 = ResBlock(hidden_dim)
        self.res2 = ResBlock(hidden_dim)
        self.attn = SelfAttention2d(hidden_dim)
        self.conv_out = nn.Conv2d(hidden_dim, latent_dim, 1)

    def forward(self, x):
        s1 = self.norm1(F.relu(self.conv1(x)))
        s2 = self.norm2(F.relu(self.conv2(s1)))
        x = self.res1(s2)
        x = self.res2(x)
        x = self.attn(x)
        z = self.conv_out(x)
        return z, (s1, s2, x)


class Decoder(nn.Module):
    def __init__(self, out_channels: int = 4, hidden_dim: int = 256, latent_dim: int = 96):
        super().__init__()
        self.conv_in = nn.Conv2d(latent_dim, hidden_dim, 1)
        self.norm_in = _groupnorm(hidden_dim)
        self.res1 = ResBlock(hidden_dim)
        self.res2 = ResBlock(hidden_dim)
        self.skip_conv1 = nn.Conv2d(hidden_dim + hidden_dim, hidden_dim, 1)
        self.skip_conv2 = nn.Conv2d(hidden_dim + hidden_dim // 2, hidden_dim // 2, 1)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            _groupnorm(hidden_dim),
            nn.ReLU(),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            _groupnorm(hidden_dim // 2),
            nn.ReLU(),
        )
        self.conv_out = nn.Conv2d(hidden_dim // 2, out_channels, 1)

    def forward(self, quantized, skip1=None, skip2=None, skip3=None):
        x = self.norm_in(F.relu(self.conv_in(quantized)))
        x = self.res1(x)
        if skip3 is not None and x.shape == skip3.shape:
            x = self.skip_conv1(torch.cat([x, skip3], dim=1))
        x = self.res2(x)
        x = self.up1(x)
        if skip2 is not None and x.shape == skip2.shape:
            x = self.skip_conv2(torch.cat([x, skip2], dim=1))
        x = self.up2(x)
        if skip1 is not None and x.shape == skip1.shape:
            x = x + skip1
        return self.conv_out(x)


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 96,
                 commitment_cost: float = 0.25, decay: float = 0.99, epsilon: float = 1e-3):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embedding", embed)
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_embedding", embed.clone())
        self.embedding: torch.Tensor

    def forward(self, z):
        # Run quantizer in fp32 for numerical stability (avoids fp16 norm underflow, matmul overflow, div-by-tiny)
        orig_dtype = z.dtype
        z_fp32 = z.float()
        z_flat = z_fp32.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        z_norm = z_flat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        z_flat = z_flat / z_norm
        emb = F.normalize(self.embedding.float(), dim=-1).clamp(min=-1e3, max=1e3)

        dist = (
            z_flat.pow(2).sum(1, keepdim=True)
            + emb.pow(2).sum(1)
            - 2 * z_flat @ emb.t()
        )
        indices = dist.argmin(dim=1)
        one_hot = F.one_hot(indices, self.num_embeddings).float()
        quantized = one_hot @ self.embedding.float()
        quantized = quantized.view_as(z_fp32)

        if self.training:
            self.ema_cluster_size.data = self.ema_cluster_size * self.decay + \
                (1 - self.decay) * one_hot.sum(0)
            n = self.ema_cluster_size.sum()
            cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
            cluster_size = cluster_size.clamp(min=1e-3)
            dw = one_hot.t() @ z_flat
            self.ema_embedding.data = self.ema_embedding * self.decay + (1 - self.decay) * dw
            self.embedding.data = self.ema_embedding / cluster_size.unsqueeze(1)

        vq_loss = self.commitment_cost * F.mse_loss(quantized.detach(), z_fp32)
        quantized = z_fp32 + (quantized - z_fp32).detach()
        return quantized.to(orig_dtype), vq_loss.to(orig_dtype), indices

    def get_codebook_entry(self, indices):
        return F.embedding(indices, self.embedding)

    def perplexity(self, indices):
        idx_flat = indices.view(-1)
        usage = torch.bincount(idx_flat, minlength=self.num_embeddings).float()
        prob = usage / usage.sum()
        ppl = torch.exp(-(prob * torch.log(prob + 1e-10)).sum())
        return ppl

    def ema_update(self, z, indices):
        if not self.training:
            return
        one_hot = F.one_hot(indices, self.num_embeddings).float()
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        self.ema_cluster_size.data = self.ema_cluster_size * self.decay + \
            (1 - self.decay) * one_hot.sum(0)
        n = self.ema_cluster_size.sum()
        cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
        dw = one_hot.t() @ z_flat
        self.ema_embedding.data = self.ema_embedding * self.decay + (1 - self.decay) * dw
        self.embedding.data = self.ema_embedding / cluster_size.unsqueeze(1)

    def reset_dead_codes(self, z, indices=None, threshold=None):
        with torch.no_grad():
            if indices is not None:
                usage = torch.bincount(indices.view(-1), minlength=self.num_embeddings).float()
                dead_mask = usage == 0
            else:
                dead_mask = self.ema_cluster_size < (threshold if threshold is not None else 1)
            n_dead = dead_mask.sum().item()
            if n_dead > 0:
                z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
                n_avail = z_flat.size(0)
                n_replace = min(n_dead, n_avail)
                perm = torch.randperm(n_avail)[:n_replace]
                new_emb = z_flat[perm]
                dead_indices = torch.where(dead_mask)[0][:n_replace]
                self.embedding.data[dead_indices] = new_emb
                self.ema_embedding.data[dead_indices] = new_emb
                self.ema_cluster_size.data[dead_indices] = 10.0
            return n_dead


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 4, ch: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for i in range(n_layers):
            mult = 2 ** i
            out_ch = ch * mult * 2
            layers += [
                nn.Conv2d(ch * mult, out_ch, 4, stride=2 if i < n_layers - 1 else 1, padding=1),
                _groupnorm(out_ch, max_groups=16),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        layers += [nn.Conv2d(ch * 2 ** n_layers, 1, 4, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers: list = None):
        super().__init__()
        if layers is None:
            layers = [3, 8, 15, 22]
        self.layers = layers
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
            for p in vgg.parameters():
                p.requires_grad = False
            self.vgg = vgg
        except Exception:
            self.vgg = None

    def forward(self, x, y):
        if self.vgg is None:
            return F.mse_loss(x, y)
        loss = 0.0
        x_feat, y_feat = x, y
        if x.size(1) == 4:
            x_feat = x[:, :3]
            y_feat = y[:, :3]
        for i, layer in enumerate(self.vgg):
            x_feat = layer(x_feat)
            y_feat = layer(y_feat)
            if i in self.layers:
                loss += F.l1_loss(x_feat, y_feat)
        return loss


def focal_frequency_loss(x, y, alpha=1.0):
    x_freq = torch.fft.fftn(x.float(), dim=(-2, -1))
    y_freq = torch.fft.fftn(y.float(), dim=(-2, -1))
    x_amp, x_phase = torch.abs(x_freq), torch.angle(x_freq)
    y_amp, y_phase = torch.abs(y_freq), torch.angle(y_freq)
    weights = 1.0 / (1.0 + torch.exp(-torch.abs(x_amp) * 0.1 + 5))
    amp_loss = F.l1_loss(x_amp * weights, y_amp * weights)
    phase_loss = F.l1_loss(torch.sin(x_phase), torch.sin(y_phase)) + \
                 F.l1_loss(torch.cos(x_phase), torch.cos(y_phase))
    return alpha * (amp_loss + 0.5 * phase_loss)


def sobel_edge_loss(x, y):
    x_f = x.float()
    y_f = y.float()
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    loss = 0.0
    for c in range(min(x_f.size(1), 3)):
        x_edge_x = F.conv2d(x_f[:, c:c+1], sobel_x, padding=1)
        x_edge_y = F.conv2d(x_f[:, c:c+1], sobel_y, padding=1)
        x_edge = torch.sqrt(x_edge_x ** 2 + x_edge_y ** 2 + 1e-8)
        y_edge_x = F.conv2d(y_f[:, c:c+1], sobel_x, padding=1)
        y_edge_y = F.conv2d(y_f[:, c:c+1], sobel_y, padding=1)
        y_edge = torch.sqrt(y_edge_x ** 2 + y_edge_y ** 2 + 1e-8)
        loss += F.l1_loss(x_edge, y_edge)
    return loss / min(x_f.size(1), 3)


def palette_histogram_loss(x, y, palette, alpha=1.0):
    B, C, H, W = x.shape
    x_rgb = x[:, :3].permute(0, 2, 3, 1).reshape(-1, 3).float()
    y_rgb = y[:, :3].permute(0, 2, 3, 1).reshape(-1, 3).float()
    pal = torch.tensor(palette, dtype=torch.float32, device=x.device) / 255.0
    x_dists = torch.cdist(x_rgb, pal)
    y_dists = torch.cdist(y_rgb, pal)
    x_soft = F.softmax(-x_dists * 10, dim=-1)
    y_soft = F.softmax(-y_dists * 10, dim=-1)
    x_hist = x_soft.mean(0)
    y_hist = y_soft.mean(0)
    return alpha * (F.l1_loss(x_hist, y_hist) + (1 - x_hist.max()) * 0.1)


class ImprovedVQVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 256,
        latent_dim: int = 96,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dim, latent_dim)
        self.quantizer = VectorQuantizerEMA(num_embeddings, latent_dim, commitment_cost, decay)
        self.decoder = Decoder(in_channels, hidden_dim, latent_dim)
        self.latent_dim = latent_dim

        self.discriminator = PatchDiscriminator(in_channels)
        self.perceptual_loss = VGGPerceptualLoss()

        self.ema_decoder = None
        self.ema_encoder = None

    def forward(self, x, return_skips=False):
        z, skips = self.encoder(x)
        quantized, vq_loss, indices = self.quantizer(z)
        recon = self.decoder(quantized, *skips)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + vq_loss
        result = {
            "recon": recon,
            "loss": loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "indices": indices.view(x.size(0), -1),
            "quantized": quantized,
            "z": z,
        }
        if return_skips:
            result["skips"] = skips
        return result

    def compute_full_loss(self, x, recon, indices, palette=None, lambda_perc=0.5,
                          lambda_ffl=0.1, lambda_edge=0.05, lambda_palette=0.1):
        perc = self.perceptual_loss(x, recon) * lambda_perc
        ffl = focal_frequency_loss(x, recon) * lambda_ffl
        edge = sobel_edge_loss(x, recon) * lambda_edge
        pal_loss = 0.0
        if palette is not None and len(palette) > 0:
            pal_loss = palette_histogram_loss(x, recon, palette) * lambda_palette
        return perc + ffl + edge + pal_loss

    def encode_to_indices(self, x):
        z, _ = self.encoder(x)
        _, _, indices = self.quantizer(z)
        return indices.view(x.size(0), -1)

    def decode_from_indices(self, indices, latent_shape):
        z = self.quantizer.get_codebook_entry(indices)
        z = z.view(-1, *latent_shape)
        decoder = self.ema_decoder if self.ema_decoder is not None and not self.training else self.decoder
        return decoder(z, None, None, None)

    def decode_from_quantized(self, quantized):
        return self.decoder(quantized, None, None, None)

    def init_ema(self, decay=0.999):
        self.ema_encoder = [p.data.clone() for p in self.encoder.parameters()]
        self.ema_decoder_weights = [p.data.clone() for p in self.decoder.parameters()]

    def update_ema(self, decay=0.999):
        if self.ema_encoder is not None:
            for ema_p, p in zip(self.ema_encoder, self.encoder.parameters()):
                ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)
        if self.ema_decoder_weights is not None:
            for ema_p, p in zip(self.ema_decoder_weights, self.decoder.parameters()):
                ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)

    def ema_update(self, x):
        z, _ = self.encoder(x)
        _, _, indices = self.quantizer(z)
        self.quantizer.ema_update(z, indices)

    def perplexity(self, x):
        _, _, indices = self.quantizer(self.encoder(x)[0])
        return self.quantizer.perplexity(indices)

    def reset_dead_codes(self, x, threshold=None):
        z, _ = self.encoder(x)
        _, _, indices = self.quantizer(z)
        return self.quantizer.reset_dead_codes(z, indices, threshold)


VQVAE = ImprovedVQVAE
VectorQuantizer = VectorQuantizerEMA
