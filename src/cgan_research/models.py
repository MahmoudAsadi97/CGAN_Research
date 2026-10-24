"""Audited reference architecture; upstream attribution in docs/research."""
import torch
from torch import nn
from torch.nn import functional as F


class ConditionalNetwork(nn.Module):
    def __init__(self, num_conditions, widths, dropout=False):
        super().__init__()
        if not isinstance(num_conditions, int) or num_conditions < 1:
            raise ValueError('num_conditions must be a positive integer')
        self.num_conditions = num_conditions
        layers = []
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
            if dropout and i == len(widths) - 2:
                layers.append(nn.Dropout(0.4))
            layers.append(nn.Linear(a, b))
            if i < len(widths) - 2:
                layers.append(nn.LeakyReLU(0.2))
        self.network = nn.Sequential(*layers)

    def forward(self, x, labels):
        if x.ndim != 2 or x.shape[1] != 24 or not x.is_floating_point():
            raise ValueError('profiles/noise must be floating (B,24)')
        if labels.shape != (len(x),) or labels.dtype != torch.long:
            raise ValueError('conditions must be int64 (B,)')
        if labels.numel() and (labels.min() < 0 or labels.max() >= self.num_conditions):
            raise ValueError('unsupported condition')
        return self.network(torch.cat((x, F.one_hot(labels, self.num_conditions).to(x.dtype)), dim=1))


class Generator(ConditionalNetwork):
    def __init__(self, num_conditions):
        super().__init__(num_conditions, [24 + num_conditions, 64, 128, 64, 24])


class Discriminator(ConditionalNetwork):
    def __init__(self, num_conditions):
        super().__init__(num_conditions, [24 + num_conditions, 64, 128, 256, 1], dropout=True)
