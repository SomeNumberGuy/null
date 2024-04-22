import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50 # Importing the U-Net architecture
from PIL import Image
import os
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define custom dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = os.listdir(os.path.join(root_dir, 'img'))[:200]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'img', self.image_files[idx])
        mask_name = os.path.join(self.root_dir, 'mask', self.image_files[idx].replace('.jpg', '.png'))
        image = Image.open(img_name)
        mask = Image.open(mask_name)

        mask = mask.convert('L')
        mask = mask.point(lambda x: 255 if x > 0 else 0, '1')

        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask

# Dice coefficient metric
def dice_coefficient(pred, target):
    smooth = 1e-6
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

# Initialize the U-Net model
model = deeplabv3_resnet50(pretrained=False, progress=True, num_classes=1).to(device) # 1 class for binary segmentation

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss() # Binary Cross Entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Define datasets and data loaders
train_dataset = CustomDataset('')
val_dataset = CustomDataset('')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(images.to(device))['out'] # Extracting the output from the model dictionary
        loss = criterion(outputs, masks.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)
    #print(train_loss)

    # Validation loop
    model.eval()
    total_dice = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            outputs = model(images.to(device))['out'] # Extracting the output from the model dictionary
            dice = dice_coefficient(outputs, masks.to(device))
            total_dice += dice.item()
    val_dice = total_dice / len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Dice Coefficient: {val_dice:.4f}")
