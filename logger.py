import torch
from torch import nn, optim
import numpy as np
import os
import time
import json
import logging
from PIL import Image
import numpy as np
import cv2

class Logger:
    def __init__(self, name='MyExperiment'):
        if not os.path.exists('logs'):
            os.makedirs('logs')
        self.logger = None
        self.experiment_id = name+str(time.strftime("%Y%m%d%H%M%S"))
        os.makedirs(os.path.join('logs', self.experiment_id), exist_ok=False)
        self.config = {}
        self.metrics = {}

    def init(self, name='MyExperiment', resume=False, anonymous=True):
        if not resume and not os.path.exists('logs'):
            os.makedirs('logs')
        #self.experiment_id = name+str(time.strftime("%Y%m%d%H%M%S"))
        log_file_path = f'logs/{self.experiment_id}.log'
        self.logger = logging.getLogger(self.experiment_id)
        self.logger.setLevel(logging.INFO)
        #self.logger.removeHandler(self.logger.handlers[0])
        #null_handler = NullHandler()
        #self.logger.addHandler(null_handler)
        file_handler = logging.FileHandler(log_file_path, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        if not resume:
            file_handler = logging.FileHandler(log_file_path, mode='w')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def log(self, epoch, step, metrics, important=True):
        if important:
            print(epoch,step, metrics)
        #    self.logger.info(f"{key}: {value}")

    def log_config(self):
        config_file_path = f'logs/{self.experiment_id}/config.json'
        with open(config_file_path, 'w') as f:
            json.dump(self.config, f)

    def set_config(self, config):
        self.config = config
        config_file_path = f'logs/{self.experiment_id}/config.json'
        with open(config_file_path, 'w') as f:
            json.dump(config, f)
    
    #def log_img(epoch, global_step, imgs):

    def write_log_img(self, epoch, global_step, imgs, name="Img"):
        if name == "Img":
            img_np = np.array(imgs[0].cpu())*255
        else:
            img_np = np.array(imgs.cpu())*255
        # If the image has a single channel (grayscale), convert it to RGB
        if len(img_np.shape) == 2:
            img_np = np.stack((img_np,) * 3, axis=-1)

        # Convert image data type to uint8
        img_np = img_np.astype(np.uint8)

        # Create PIL Image from numpy array
        img_pil = Image.fromarray(img_np)

        # Save the image as a JPG file
        img_pil.save(f'logs/experiment_{name}_epoch_{epoch}_step_{global_step}.jpg')

    def log_imgs(self, epoch, global_step, imgs, name="Img"):
        for index in range(len(imgs)):
            imgs[index] = np.array(imgs[index].cpu())*255
        min_height, min_width = min(img.shape[:2] for img in imgs)

        # Resize and save all images
        resized_images = []
        for img in imgs:
            img = cv2.resize(img, (min_width, min_height))
            resized_images.append(img)

        # Concatenate images side by side
        result_image = np.concatenate(resized_images, axis=1)
        output_filename = f'logs/experiment_{name}_epoch_{epoch}_step_{global_step}.jpg'
        # Save the result
        cv2.imwrite(output_filename, result_image)



    