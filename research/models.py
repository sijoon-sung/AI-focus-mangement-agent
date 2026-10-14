"""Scratch CNN and a temporal ablation; these are not claimed as novel methods."""
import torch
from torch import nn


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = 3
        for width in [16, 32, 64, 128]:
            layers.extend([nn.Conv2d(channels, width, 3, stride=2, padding=1, bias=False),
                           nn.GroupNorm(8, width), nn.ReLU(inplace=True)])
            channels = width
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


class EngagementModel(nn.Module):
    def __init__(self, architecture='tiny_tcn', pretrained=False):
        super().__init__()
        if architecture not in {'tiny_mean', 'tiny_tcn', 'resnet18_tcn'}:
            raise ValueError(architecture)
        if architecture.startswith('tiny'):
            if pretrained:
                raise ValueError('TinyCNN is trained from scratch')
            self.encoder, width = TinyCNN(), 128
        else:
            from torchvision.models import resnet18, ResNet18_Weights
            self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            width = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        self.temporal = nn.Identity() if architecture == 'tiny_mean' else nn.Sequential(
            nn.Conv1d(width, 128, 3, padding=1), nn.ReLU(), nn.Dropout(.2),
            nn.Conv1d(128, 128, 3, padding=2, dilation=2), nn.ReLU())
        self.classifier = nn.Linear(width if architecture == 'tiny_mean' else 128, 4)

    def forward(self, x):
        b, t, c, h, w = x.shape
        features = self.encoder(x.reshape(b * t, c, h, w)).reshape(b, t, -1)
        return self.classifier(self.temporal(features.transpose(1, 2)).mean(dim=-1))
