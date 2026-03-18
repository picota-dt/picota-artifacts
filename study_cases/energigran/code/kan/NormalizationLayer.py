import torch

from torch import nn

import Device
from kan.ParametricSigmoid import ParametricSigmoid


class NormalizationParametricSigmoid(nn.Module):
    eps = 0.01

    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = nn.Parameter(torch.tensor([max(std, self.eps)], device=Device.get_device()))
        self.sigmoid = ParametricSigmoid()

    def forward(self, x):
        z = (x - self.mean) / self.std
        z = torch.clamp(z, min=-3.0, max=3.0)
        return self.sigmoid.forward(z)


class NormalizationLayer(nn.Module):
    def __init__(self, means: list[float], stds: list[float]):
        super().__init__()
        self.size = len(means)
        self.layers = nn.ModuleList([
            NormalizationParametricSigmoid(means[i], stds[i]) for i in range(len(means))
        ])

    def forward(self, x):
        if x.dim() == 1:
            assert x.shape[0] == self.size
            outputs = [normalization_layers(x[i]) for i, normalization_layers in enumerate(self.layers)]
            return torch.stack(outputs).squeeze(1)
        if x.dim() == 2:
            assert x.shape[1] == self.size
            outputs = [layer(x[:, i]) for i, layer in enumerate(self.layers)]
            return torch.stack(outputs, dim=-1)
        if x.dim() == 3:
            assert x.shape[2] == self.size
            outputs = [layer(x[:, :, i]) for i, layer in enumerate(self.layers)]
            return torch.stack(outputs, dim=-1)
        raise ValueError(f"Unsupported input shape: {x.shape}")
