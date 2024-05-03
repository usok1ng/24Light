import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from light_type import parse_light_types
from model import *
import cv2
import numpy as np

IMAGE_SIZE=128

image_path = "data/LSMI/image/Place630_1.jpg"
normal_path = "data/LSMI/normal/Normal630_1.jpg"

image = Image.open(image_path).convert('RGB')
normal = Image.open(normal_path).convert('RGB')
shading = Image.open(image_path).convert('L')


transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

shading_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5], std=[0.5])
        ])

image = transform(image).unsqueeze(0)
normal = transform(normal).unsqueeze(0)
shading = shading_transform(shading)

model = ShadingNet()

model_path = "./model.pth"

model = ShadingNet()
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)

model.eval()

dir_shading, spot_shading, point_shading = model(image, normal)
print(dir_shading)

dir_image = (dir_shading.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
spot_image = (spot_shading.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
point_image = (point_shading.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)

cv2.imwrite("./dir.jpg", dir_image)
cv2.imwrite("./spot.jpg", spot_image)
cv2.imwrite("./point.jpg", point_image)


