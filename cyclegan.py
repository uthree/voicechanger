import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels=384, spectral_norm=True, alpha=0.01, dropout=0):
        super().__init__()
        self.c1 = nn.Conv1d(channels, channels, 5, 1, 2)
        self.act = nn.LeakyReLU(alpha)
        self.c2 = nn.Conv1d(channels, channels, 5, 1, 2)
        self.dropout_rate = dropout
        if spectral_norm:
            self.c1 = torch.nn.utils.spectral_norm(self.c1)
            self.c2 = torch.nn.utils.spectral_norm(self.c2)

    def forward(self, x):
        res = x
        x = self.act(self.c1(x))
        x = F.dropout(x, p=self.dropout_rate)
        x = self.c2(x)
        x = F.dropout(x, p=self.dropout_rate)
        return x + res


class Generator(nn.Module):
    def __init__(self, input_channels=771, internal_channels=384, num_layers=8):
        super().__init__()
        self.input_layer = nn.Conv1d(input_channels, internal_channels, 1, 1, 0)
        self.mid_layers = nn.Sequential(
                *[ResBlock(internal_channels, False) for _ in range(num_layers)])
        self.output_layer = nn.Conv1d(internal_channels, input_channels, 1, 1, 0)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.mid_layers(x)
        x = self.output_layer(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_channels=771, internal_channels=384, num_layers=8):
        super().__init__()
        self.input_layer = nn.utils.spectral_norm(
                nn.Conv1d(input_channels, internal_channels, 5, 1, 0))
        self.mid_layers = nn.Sequential(
                *[ResBlock(internal_channels, True, dropout=0.1, alpha=0.2) for _ in range(num_layers)])
        self.output_layer = nn.Conv1d(internal_channels, 1, 1, 1, 0)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.mid_layers(x)
        x = self.output_layer(x)
        return x
