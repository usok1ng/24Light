import torch
from model import ShadingNet
from dataset import LightingDataset
from train import train_model
from torch.utils.data import DataLoader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = LightingDataset(image_dir='LSMI Data')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = ShadingNet().to(device)
    train_model(model, dataloader, device)

if __name__ == "__main__":
    main()