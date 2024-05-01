# train.py

import torch
import torch.optim as optim
from functions import parse_light_types

def train_model(model, dataloader, device):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(10):
        for inputs, original_images, img_names in dataloader:
            inputs = inputs.to(device)
            original_images = original_images.to(device)
            optimizer.zero_grad()

            light_types = parse_light_types(img_names)
            shadings = model(inputs, light_types)

            combined_shading = torch.zeros_like(original_images)
            for light_type in shadings:
                combined_shading += shadings[light_type]

            loss = criterion(combined_shading, original_images)
            loss.backward()
            optimizer.step()

            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
