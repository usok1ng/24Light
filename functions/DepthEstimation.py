from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import os

image_folder = "dataset/image"
depth_folder = "dataset/depth"

image_files = [file for file in os.listdir(image_folder)]

image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf")

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = Image.open(image_path)

    image_name, _ = os.path.splitext(image_file)
    depth_map_file = image_name.replace("Place", "Depth")

    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)

    depth.save(os.path.join(depth_folder, f"{depth_map_file}.jpg"))