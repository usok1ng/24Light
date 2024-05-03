import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from light_type import parse_light_types
IMAGE_SIZE = 128

class LSMIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'LSMI', 'image')
        self.normals_dir = os.path.join(root_dir, 'LSMI', 'normal')
        self.image_files = sorted(os.listdir(self.images_dir))
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.shading_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_name)
        
        normal_name = image_name.replace('Place', 'Normal') if 'Place' in image_name else image_name
        normal_path = os.path.join(self.normals_dir, normal_name)
        
        image = Image.open(image_path).convert('RGB')
        normal = Image.open(normal_path).convert('RGB')
        shading = Image.open(image_path).convert('L')
        
        image_tensor = self.transform(image)
        normal_tensor = self.transform(normal)
        shading_tensor = self.shading_transform(shading)
        light_types = parse_light_types([image_name])

        return image_tensor, normal_tensor, shading_tensor, light_types