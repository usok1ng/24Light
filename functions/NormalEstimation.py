import torch
import cv2
import numpy as np
import os

image_folder = "dataset/image"
output_folder = "dataset/normal"

image_files = [file for file in os.listdir(image_folder)]

normal_predictor = torch.hub.load("hugoycj/DSINE-hub", "DSINE", trust_repo=True)

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)

    image_name, _ = os.path.splitext(image_file)
    normal_map_file = image_name.replace("Place", "Normal")

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    h, w = image.shape[:2]

    with torch.inference_mode():
        normal = normal_predictor.infer_cv2(image)[0]
        normal = (normal + 1) / 2

    normal = (normal * 255).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
    normal = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)

    output_path = os.path.join(output_folder, f"{normal_map_file}.jpg")
    cv2.imwrite(output_path, normal)
