# dataset.py

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class LightingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'LSMI', 'image')
        self.normals_dir = os.path.join(root_dir, 'LSMI', 'normal')
        self.image_files = sorted(os.listdir(self.images_dir))
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.orig_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        normal_path = os.path.join(self.normals_dir, img_name.replace('image', 'normal'))  # Adjust if naming convention differs
        
        # Load image and normal map
        img = Image.open(img_path).convert('RGB')
        normal_img = Image.open(normal_path).convert('RGB')
        
        # Transform
        img_tensor = self.transform(img)
        normal_tensor = self.transform(normal_img)
        orig_img_tensor = self.orig_transform(img)  # 원본 이미지 텐서

        return normal_tensor, orig_img_tensor, img_name  # Return normal map as the main input

