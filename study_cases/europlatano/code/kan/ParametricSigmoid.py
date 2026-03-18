from torch import nn

import torch

class ParametricSigmoid(nn.Module):
    def __init__(self, alpha_init=1.0, beta_init=0.0):
        super(ParametricSigmoid, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))

    def forward(self, x):
        return 1 / (1 + torch.exp(-self.alpha * x - self.beta))