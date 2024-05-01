# model.py

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

        self.directional_light_params = nn.Parameter(torch.randn(3))  # 방향
        self.spot_light_params = nn.Parameter(torch.randn(4))  # 위치, 각도
        self.point_light_params = nn.Parameter(torch.randn(4))  # 위치, 강도

    def forward(self, x, light_types):
        features = self.feature_extractor(x)
        shadings = {}
        for light_type in light_types:
            if light_type == 'directional':
                shadings[light_type] = self.compute_directional_shading(features, self.directional_light_params)
            elif light_type == 'spot':
                shadings[light_type] = self.compute_spot_shading(features, self.spot_light_params)
            elif light_type == 'point':
                shadings[light_type] = self.compute_point_shading(features, self.point_light_params)
        return shadings

    def compute_directional_shading(self, features, params):
        light_dir = F.normalize(params[:3], p=2, dim=0)
        ndotl = torch.sum(features * light_dir, dim=1, keepdim=True)
        return F.relu(ndotl)

    def compute_spot_shading(self, features, params):
        light_pos = params[:3]
        cutoff = params[3]
        # 스포트라이트의 각도 및 거리 감쇠
        distance = torch.sqrt(torch.sum((features - light_pos) ** 2, dim=1, keepdim=True))
        angle = torch.acos(F.relu(torch.dot(features, light_pos) / (torch.norm(features, dim=1) * torch.norm(light_pos))))
        attenuation = torch.exp(-distance) * ((angle < cutoff).float())
        return attenuation

    def compute_point_shading(self, features, params):
        light_pos = params[:3]
        intensity = params[3]
        # 포인트 라이트 거리 감쇠
        distance = torch.sqrt(torch.sum((features - light_pos) ** 2, dim=1, keepdim=True))
        attenuation = intensity / (distance + 1e-6)
        return attenuation
