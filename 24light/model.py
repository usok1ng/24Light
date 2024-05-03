import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageFeatureExtraction(nn.Module):
    def __init__(self):
        super(ImageFeatureExtraction, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=7, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.layers(x)

class LightEstimator(nn.Module):
    def __init__(self):
        super(LightEstimator, self).__init__()
        self.directional_layer = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.spot_layer = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.point_layer = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, features):
        # print(features.shape)
        directional_shading = self.directional_layer(features)
        spot_shading = self.spot_layer(features)
        point_shading = self.point_layer(features)

        return directional_shading, spot_shading, point_shading

    # def compute_directional_shading(self, features, params):
    #   light_dir = F.normalize(params[:3], p=2, dim=0)
    #   print(light_dir.shape)
    #   print(features.shape)
    #   ndotl = torch.sum(features * light_dir, dim=1, keepdim=True)
    #   return F.relu(ndotl)

    #def compute_spot_shading(self, features, params):
    #    light_pos = params[:3]
    #    cutoff = params[3]
    #    distance = torch.norm(features - light_pos, dim=1, keepdim=True)
    #    angle = torch.acos(F.relu(torch.dot(features, light_pos) / (torch.norm(features, dim=1) * torch.norm(light_pos))))
    #    attenuation = torch.exp(-distance) * ((angle < cutoff).float())
    #    return attenuation

    #def compute_point_shading(self, features, params):
    #    light_pos = params[:3]
    #    intensity = params[3]
    #    distance = torch.norm(features - light_pos, dim=1, keepdim=True)
    #    attenuation = intensity / (distance + 1e-6)
    #   return attenuation

class ShadingNet(nn.Module):
    def __init__(self):
        super(ShadingNet, self).__init__()
        self.image_feature_extractor = ImageFeatureExtraction()
        self.normal_feature_extractor = ImageFeatureExtraction()
        self.light_estimator = LightEstimator()

    def forward(self, image, normal):
        image_features = self.image_feature_extractor(image)
        normal_features = self.normal_feature_extractor(normal)
                
        combined_features = torch.cat([image_features, normal_features], dim=1)
        directional_shading, spot_shading, point_shading = self.light_estimator(combined_features)
        return directional_shading, spot_shading, point_shading
