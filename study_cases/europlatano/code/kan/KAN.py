import torch
from torch import nn

from kan.KAL import KAL
from kan.NormalizationLayer import NormalizationLayer
from kan.ParametricSigmoid import ParametricSigmoid


class KAN(nn.Module):

    def __init__(self, input_features, lookback_size, means, stds, output_features):
        super().__init__()
        if len(means) != len(stds):
            raise ValueError('means and stds must have same length')
        self.normalization_layer = NormalizationLayer(means, stds)
        self.kan = self.build(input_features, lookback_size, output_features)

    def build(self, input_features, lookback_size, output_features):
        return nn.Sequential(
            KAL(input_features + lookback_size * input_features, 50),
            KAL(50, output_features),
            ParametricSigmoid()
        )

    def forward(self, batch):
        x = self.normalize(batch)
        return self.output(x)

    def normalize(self, batch):
        t = batch['t']
        t_features = batch['numerical_t_features']
        categorical_t_features = batch['categorical_t_features']
        lookback_t = batch['lookback_t']
        lookback_features = batch['numerical_lookback_features']
        categorical_lookback_features = batch['categorical_lookback_features']
        normalized_numerical_t_features = self.normalization_layer(t_features)
        if lookback_features.size(1) != 0:
            normalized_lookback_features = self.normalization_layer(lookback_features)
            numerical_lookback_features_flat = normalized_lookback_features.view(normalized_lookback_features.size(0),
                                                                                 -1)
        else:
            numerical_lookback_features_flat = torch.zeros((t_features.size(0), 0), device=t_features.device)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        if lookback_t.dim() == 1:
            lookback_t = lookback_t.unsqueeze(1)
        lookback_t_flat = lookback_t.view(lookback_t.size(0), -1)
        categorical_lookback_features_flat = categorical_lookback_features.view(categorical_lookback_features.size(0),
                                                                                -1)
        x = torch.cat([
            t,
            normalized_numerical_t_features,
            categorical_t_features,
            lookback_t_flat,
            numerical_lookback_features_flat,
            categorical_lookback_features_flat,
        ], dim=1)
        return x

    def output(self, x):
        return self.kan(x)
