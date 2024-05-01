# model.py
import torch
import torch.nn as nn

class ShadingNet(nn.Module):
    def __init__(self):
        super(ShadingNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.shading_layers = nn.ModuleDict({
            'directional': nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 1, kernel_size=1)
            ),
            'spot': nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 1, kernel_size=1)
            )
        })

    def forward(self, x, light_types):
        features = self.feature_extractor(x)
        shadings = {}
        for light in light_types:
            shadings[light] = self.shading_layers[light](features)
        return shadings
