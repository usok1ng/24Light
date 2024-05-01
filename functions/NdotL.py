import numpy as np
import cv2
import os

normal_path = 'dataset/normal/Normal483_1.jpg'
normal = cv2.imread(normal_path, cv2.IMREAD_COLOR)
normal = normal.astype(np.float32) / 255.0 * 2 - 1
lighting = np.array([0.4, 0, 0.6])

shading = np.maximum(np.sum(normal * lighting, axis=2), 0)
shading_gray = (shading * 255).astype(np.uint8)
shading_path = 'dataset/shading/Shading0_1.jpg'
cv2.imwrite(shading_path, shading_gray)