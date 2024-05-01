import torch
import torch.nn as nn
import torch.nn.functional as F

class ShadingNet(nn.Module):
    def __init__(self):
        super(ShadingNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.shading_layers = nn.ModuleDict({
            'directional': self.create_directional_shading_layer(),
            'spot': self.create_spot_shading_layer(),
            'point': self.create_point_shading_layer()
        })

    def create_directional_shading_layer(self):
        # NdotL: cosine of the angle between normal and light direction
        return nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)  # Assuming light direction is constant or passed separately
        )

    def create_spot_shading_layer(self):
        # Spot light specific calculations could involve distance and angle attenuation
        return nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)  # Further customization may be needed
        )

    def create_point_shading_layer(self):
        # Point light calculations typically involve distance attenuation
        return nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)  # Further customization may be needed
        )

    def forward(self, x, light_types):
        features = self.feature_extractor(x)
        shadings = {}
        for light in light_types:
            shadings[light] = self.shading_layers[light](features)
        return shadings
