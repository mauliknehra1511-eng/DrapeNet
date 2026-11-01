"""CNN that maps a 64x64 top-view drape silhouette -> normalised fabric parameters.

Targets (all in log10 except w), standardised with dataset mean/std:
  log10 B1, log10 B2, log10 G, w
Weight is usually known (weigh the specimen), so the model can take it as an extra input:
  use_weight=True concatenates w to the CNN features (recommended, matches the real test).
"""
import torch, torch.nn as nn


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU(), nn.MaxPool2d(2))


class DrapeCNN(nn.Module):
    def __init__(self, n_out=3, use_weight=True, width=32):
        super().__init__()
        self.use_weight = use_weight
        self.f = nn.Sequential(block(1, width), block(width, 2 * width), block(2 * width, 4 * width),
                               block(4 * width, 4 * width), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(4 * width + int(use_weight), 128), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(128, n_out))

    def forward(self, img, w=None):
        z = self.f(img)
        if self.use_weight:
            z = torch.cat([z, w[:, None]], 1)
        return self.head(z)
