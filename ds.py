import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import torchvision
from torchvision.transforms import ToTensor, Normalize

import random


def extract_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def calculate_bounding_box(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return x, y, x + w, y + h


def get_most_common_color(image, x1, y1, x2, y2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cropped = gray[y1:y2, x1:x2]
    flattened_image = cropped.flatten()
    counts = np.bincount(flattened_image)
    most_common_color = np.argmax(counts)
    return most_common_color


def process_image(image):
    contours = extract_contours(image)
    boxes = []
    labels = []
    masks = []
    for contour in contours:
        x1, y1, x2, y2 = calculate_bounding_box(contour)
        class_label = get_most_common_color(image, x1, y1, x2, y2)
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), cv2.FILLED)
        cropped_mask = mask[y1:y2, x1:x2]
        boxes.append([x1, y1, x2, y2])
        labels.append(class_label)

        gray_image = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
        masks.append(binary_image)
    return boxes, labels, masks


def process_images(images):
    boxes_list = []
    labels_list = []
    masks_list = []
    for image in images:
        boxes, labels, masks = process_image(image)
        boxes_list.append(boxes)
        labels_list.append(labels)
        masks_list.append(masks)
    return (
        torch.tensor(boxes_list, dtype=torch.float),
        torch.tensor(labels_list, dtype=torch.int64),
        torch.tensor(masks_list, dtype=torch.uint8)
    )


transform = torchvision.transforms.Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class MaskRCNNDataset(Dataset):
    def __init__(self, root_folder, device, target_size=(224, 224)):
        self.root_folder = root_folder
        self.target_size = target_size
        self.device = device
        self.image_paths = sorted(
            [os.path.join(root_folder, "images", img) for img in os.listdir(os.path.join(root_folder, "images"))])
        self.mask_paths = sorted(
            [os.path.join(root_folder, "masks", mask) for mask in os.listdir(os.path.join(root_folder, "masks"))])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image_resized = cv2.resize(image, self.target_size)
        mask = cv2.imread(self.mask_paths[idx])
        mask_resized = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        boxes, labels, masks = process_image(mask_resized)
        out_boxes = []
        for box, label, mask in zip(boxes, labels, masks):
            # for i in range(len(box)):
            #    box[i] = 1
            # print(box, torch.tensor(box, dtype=torch.float32, device=self.device))
            out_boxes.append({"boxes": torch.tensor(box, dtype=torch.float32, device=self.device),
                              "labels": torch.tensor(label, device=self.device),
                              "masks": torch.tensor(mask, device=self.device)})
        # print(out_boxes)
        return transform(image_resized).to(
            self.device), out_boxes  # {"boxes": torch.tensor(boxes), "labels": torch.tensor(labels), "masks": torch.tensor(masks)} #"labels"
        return torch.tensor(image_resized).permute(2, 0,
                                                   1), out_boxes  # {"boxes": torch.tensor(boxes), "labels": torch.tensor(labels), "masks": torch.tensor(masks)} #"labels"


def loadData(root_folder, batchSize=4, imageSize=(224, 224)):
    image_paths = sorted(
        [os.path.join(root_folder, "images", img) for img in os.listdir(os.path.join(root_folder, "images"))])
    mask_paths = sorted(
        [os.path.join(root_folder, "masks", mask) for mask in os.listdir(os.path.join(root_folder, "masks"))])
    batch_Imgs = []
    batch_Data = []  # load images and masks
    for i in range(batchSize):
        idx = random.randint(0, len(image_paths) - 1)
        img = cv2.imread(image_paths[idx])
        img = cv2.resize(img, imageSize, cv2.INTER_LINEAR)
        mask = cv2.imread(mask_paths[idx])
        mask_resized = cv2.resize(mask, imageSize, interpolation=cv2.INTER_NEAREST)
        _, _, masks = process_image(mask_resized)

        num_objs = len(masks)
        if num_objs == 0: return loadData(root_folder)  # if image have no objects just load another image
        boxes = torch.zeros([num_objs, 4], dtype=torch.float32)
        for i in range(num_objs):
            x, y, w, h = cv2.boundingRect(masks[i])
            boxes[i] = torch.tensor([x, y, x + w, y + h])
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        img = torch.as_tensor(img, dtype=torch.float32)
        data = {}
        data["boxes"] = boxes
        data["labels"] = torch.ones((num_objs,), dtype=torch.int64)  # there is only one class
        data["masks"] = masks
        batch_Imgs.append(img)
        batch_Data.append(data)  # load images and masks
    batch_Imgs = torch.stack([torch.as_tensor(d) for d in batch_Imgs], 0)
    batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
    return batch_Imgs, batch_Data
