# train.py

import torch
import torch.optim as optim
from light_type import parse_light_types
from tqdm import tqdm

def train_model(model, dataloader, device):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    best_loss = float('inf')

    model.train()
    for epoch in range(30):
        epoch_loss = 0
        for image_tensor, normal_tensor, shading_tensor, light_types in tqdm(dataloader):
            input_image = image_tensor.to(device)
            input_normal = normal_tensor.to(device)
            input_shading = shading_tensor.to(device)
            light_types = light_types.to(device)
            optimizer.zero_grad()

            directional_shading, spot_shading, point_shading = model(input_image, input_normal)
            shadings = torch.cat([directional_shading, spot_shading, point_shading], dim=1)
            light_types_ex = light_types.expand(128, 128, -1, -1).permute(2, 3, 0, 1)
            print(light_types_ex.shape)
            course_shadings = torch.sum(light_types_ex.float() * shadings, dim=1) / 3

            loss = criterion(course_shadings, input_shading)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)

        if avg_loss < best_loss:
            print(epoch)
            best_loss = avg_loss
            torch.save(model.state_dict(), "./model.pth")

        print(f'Epoch {epoch+1}, Loss: {avg_loss}')