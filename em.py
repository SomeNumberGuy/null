import os
import random
import numpy as np
from PIL import Image

def load_image(image_path):
    return Image.open(image_path)

def create_random_rect(image_size, rect_size):
    max_x = image_size[0] - rect_size[0]
    max_y = image_size[1] - rect_size[1]
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    return (x, y, x + rect_size[0], y + rect_size[1])

def is_rect_in_mask(rect, mask):
    mask_crop = mask.crop(rect)
    return np.array(mask_crop).sum() == 0

def save_image_part(image, rect, output_folder, index):
    part = image.crop(rect)
    part.save(os.path.join(output_folder, f'part_{index}.png'))

def main(image_path, mask_path, output_folder, rect_size=(100, 100), num_parts=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image = load_image(image_path)
    mask = load_image(mask_path).convert('L')  # Convert mask to grayscale

    image_size = image.size
    parts_saved = 0
    attempts = 0
    max_attempts = num_parts * 10

    while parts_saved < num_parts and attempts < max_attempts:
        rect = create_random_rect(image_size, rect_size)
        if is_rect_in_mask(rect, mask):
            save_image_part(image, rect, output_folder, parts_saved)
            parts_saved += 1
        attempts += 1

    if parts_saved < num_parts:
        print(f"Only {parts_saved} parts could be saved. Consider increasing max_attempts or changing rect_size.")

if __name__ == '__main__':
    image_path = 'path_to_image.jpg'
    mask_path = 'path_to_mask.png'
    output_folder = 'output_parts'
    main(image_path, mask_path, output_folder)
