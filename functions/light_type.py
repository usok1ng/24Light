def parse_light_types(image_filenames):
    light_types = []

    for filename in image_filenames:
        parts = filename.split('_')
        if len(parts) > 1:
            light_info = parts[-1]
            light_numbers = [int(char) for char in light_info if char.isdigit()]

            current_lights = []
            if len(light_numbers) > 0:
                if light_numbers[0] > 0:
                    current_lights.append('directional')

                if len(light_numbers) > 1 and light_numbers[1] > 0:
                    current_lights.append('spot')

                
                if len(light_numbers) > 2 and light_numbers[2] > 0:
                    current_lights.append('point')

            light_types.append(current_lights)
        else:
            light_types.append([])

    return light_types
