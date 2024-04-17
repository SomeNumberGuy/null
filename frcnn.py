import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor
from torchvision.transforms.functional import convert_image_dtype
from torchvision.io import read_image
from torchvision.transforms.functional import InterpolationMode
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

class DarknetDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "labels"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        label_path = os.path.join(self.root, "labels", self.labels[idx])

        # Load image
        img = read_image(img_path)

        # Parse annotations
        boxes = []
        labels = []
        with open(label_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                class_id, x_center, y_center, width, height = map(float, line.split())
                xmin = int((x_center - width / 2) * img.shape[-1])
                ymin = int((y_center - height / 2) * img.shape[-2])
                xmax = int((x_center + width / 2) * img.shape[-1])
                ymax = int((y_center + height / 2) * img.shape[-2])
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(class_id))

        # Convert to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transform:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_transform(train):
    transforms = []
    transforms.append(convert_image_dtype)
    transforms.append(to_tensor)
    return transforms

def collate_fn(batch):
    return tuple(zip(*batch))

# Dataset and DataLoader
dataset = DarknetDataset(root="path/to/dataset", transform=get_transform(train=True))
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Model
model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 80  # COCO dataset classes
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, targets in data_loader:
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {losses.item()}")

# Save the trained model
torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')
