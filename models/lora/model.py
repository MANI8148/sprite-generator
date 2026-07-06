import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_a = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (x @ self.lora_a @ self.lora_b) * self.scaling

    def merge_weights(self):
        self.linear.weight.data += (self.lora_a @ self.lora_b).t() * self.scaling
        self.lora_a.data.zero_()
        self.lora_b.data.zero_()


class LoRAConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 rank: int = 4, alpha: float = 1.0, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.conv.weight.requires_grad = False
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        fan_in = in_channels * kernel_size * kernel_size
        self.lora_a = nn.Parameter(torch.randn(fan_in, rank) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(rank, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        ba = (self.lora_a @ self.lora_b) * self.scaling
        lora_weight = ba.t().view(self.conv.out_channels, self.conv.in_channels,
                                  self.conv.kernel_size[0], self.conv.kernel_size[1])
        lora_update = F.conv2d(x, lora_weight, stride=self.conv.stride, padding=self.conv.padding)
        return out + lora_update

    def merge_weights(self):
        ba = (self.lora_a @ self.lora_b) * self.scaling
        merged = ba.t().view(self.conv.out_channels, self.conv.in_channels,
                             self.conv.kernel_size[0], self.conv.kernel_size[1])
        self.conv.weight.data += merged
        self.lora_a.data.zero_()
        self.lora_b.data.zero_()


class SpriteLoRAWrapper(nn.Module):
    def __init__(self, pretrained_unet=None, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        self.conv_in = LoRAConv2d(4, 64, 3, rank=rank, alpha=alpha, padding=1)
        self.conv1 = LoRAConv2d(64, 128, 4, rank=rank, alpha=alpha, stride=2, padding=1)
        self.conv2 = LoRAConv2d(128, 128, 4, rank=rank, alpha=alpha, stride=2, padding=1)
        self.conv3 = LoRAConv2d(128, 256, 4, rank=rank, alpha=alpha, stride=2, padding=1)
        self.conv4 = LoRAConv2d(256, 256, 3, rank=rank, alpha=alpha, padding=1)

        self.mid_linear = LoRALinear(256 * 4 * 4, 256 * 4 * 4, rank=rank, alpha=alpha)

        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.deconv1.weight.requires_grad = False
        self.deconv1.bias.requires_grad = False
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.deconv2.weight.requires_grad = False
        self.deconv2.bias.requires_grad = False
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.deconv3.weight.requires_grad = False
        self.deconv3.bias.requires_grad = False
        self.deconv_out = nn.Conv2d(32, 4, 3, padding=1)
        self.deconv_out.weight.requires_grad = False
        self.deconv_out.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv_in(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.mid_linear(x))
        x = x.view(x.size(0), 256, 4, 4)

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = self.deconv_out(x)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv_in(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.mid_linear(x))
        return x.view(x.size(0), 256, 4, 4)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.deconv1(z))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = self.deconv_out(x)
        return x

    def generate(self, num_samples: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn(num_samples, 256, 4, 4, device=device)
        return self.decode(z)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def lora_parameters(self) -> list:
        lora_params = []
        for module in self.modules():
            if isinstance(module, (LoRALinear, LoRAConv2d)):
                lora_params.append(module.lora_a)
                lora_params.append(module.lora_b)
        return lora_params
