# train.py
import torch
import torch.optim as optim

def train_model(model, dataloader, device):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(10):  # 적절한 에폭 수 설정
        for inputs, img_names in dataloader:
            inputs = inputs.to(device)
            optimizer.zero_grad()

            # 광원 정보를 파싱해서 모델에 전달
            light_types = parse_light_types(img_names)
            shadings = model(inputs, light_types)

            # TODO: 쉐이딩을 합성하여 최종 이미지를 생성하고 원본 이미지와 비교하여 loss 계산
            loss.backward()
            optimizer.step()