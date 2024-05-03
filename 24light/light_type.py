import torch

def parse_light_types(image_filenames):
    light_type = torch.zeros(3)
    for filename in image_filenames:
        parts = filename.split('_')
        if len(parts) > 1:
            light_info = parts[-1].rstrip('.jpg')
            light_numbers = [int(num) for num in light_info if num.isdigit()]

            # one hot encoding
            current_lights = []

            # 0: directional, 1: spot, 2: point
            if light_numbers[0] > 0:
                light_type[0] = 1
            if len(light_numbers) > 1 and light_numbers[1] > 0:
                light_type[1] = 1
            if len(light_numbers) > 2 and light_numbers[2] > 0:
                light_type[2] = 1   
    return light_type
