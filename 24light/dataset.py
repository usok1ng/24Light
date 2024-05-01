# dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class LightingDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.jpg')])

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        img = Image.open(img_name).convert('RGB')

        img_tensor = self.transforms(img)
        return img_tensor, img_name

    def get_normal_depth(self, img_name):
        normal_name = img_name.replace('Place', 'Normal')
        depth_name = img_name.replace('Place', 'Depth')
        normal = Image.open(normal_name).convert('RGB')
        depth = Image.open(depth_name).convert('L')
        
        normal_tensor = self.transforms(normal)
        depth_tensor = self.transforms(depth)
        return normal_tensor, depth_tensor
