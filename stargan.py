import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class AdaptiveInstanceNormalization1d(nn.Module):
    def __init__(self, input_channels=256, style_channels=256, eps=1e-6):
        super().__init__()
        self.to_mu = nn.Conv1d(style_channels, input_channels, 1, 1, 0)
        self.to_sigma = nn.Conv1d(style_channels, input_channels, 1, 1, 0)
        self.eps = eps

    def forward(self, x, y):
        mu = self.to_mu(y)
        sigma = self.to_sigma(y)
        # normalize x
        x = (x - x.mean(dim=2, keepdim=True)) / x.std(dim=2, keepdim=True) + self.eps
        # scale and shift
        x = x * mu + sigma
        return x


class ResBlock(nn.Module):
    def __init__(self, channels=256, norm=False):
        super().__init__()
        self.c1 = nn.Conv1d(channels, channels, 5, 1, 2)
        self.act = nn.LeakyReLU(0.1)
        self.c2 = nn.Conv2d(channels, channels, 5, 1, 2)
        if norm:
            self.c1 = spectral_norm(self.c1)
            self.c2 = spectral_norm(self.c2)

    def forward(self, x):
        res = x
        x = self.c1(x)
        x - self.act(x)
        x = self.c2(x)
        return x + res


class StyleEncoder(nn.Module):
    def __init__(self, input_channels=64, internal_channels=256, num_layers=4):
        super().__init__()
        self.input_layer = nn.Conv1d(input_channels, internal_channels, 5, 1, 2)
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.Conv1d(nn.Conv1d(internal_channels, internal_channels, 5, 1, 2)))
            self.layers.append(nn.LeakyReLU(0.1))
            # Downsample layer
            self.layers.append(nn.Conv1d(internal_channels, internal_channels, 2, 2, 0))

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.layers:
            x = layer(x)
        return x


class SpeakerPredictor(nn.Module):
    def __init__(self, input_channels=256, num_speakers=512):
        super().__init__()
        self.to_speakers = nn.Conv1d(input_channels, num_speakers, 1, 1, 0, bias=False)

    def forward(self, x):
        return self.to_speakers(x)

