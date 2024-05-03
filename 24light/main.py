import torch
from model import ShadingNet
from dataset import LSMIDataset
from train import train_model
from torch.utils.data import DataLoader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = LSMIDataset(root_dir='data')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=32)
    model = ShadingNet().to(device)

    train_model(model, dataloader, device)    

if __name__ == "__main__":
    main()