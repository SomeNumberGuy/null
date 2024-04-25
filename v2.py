import torch
import torchvision
from torchvision import datasets, models, transforms
import argparse
import os
import copy
import time

# Set training mode
train_mode = 'finetune'  # Default to finetuning

# Construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, help="Training mode: finetue/transfer/scratch")
args = vars(ap.parse_args())

if args["mode"] != train_mode:
    if args["mode"] == 'finetune':
        train_mode = 'finetune'
    elif args["mode"] == 'scratch':
        train_mode = 'scratch'
    elif args["mode"] == 'transfer':
        train_mode = 'transfer'

# Set device (GPU or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define transforms
transform = transforms.Compose([transforms.Resize(224),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])])

# Load the dataset
train_dataset = datasets.ImageFolder('path/to/train/directory', transform=transform)
test_dataset = datasets.ImageFolder('path/to/test/directory', transform=transform)

# Define data loaders
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the model
if train_mode == 'finetune':
    model_ft = models.resnet50(pretrained=True)
elif train_mode == 'scratch':
    model_ft = models.resnet50()
else:  # train_mode == 'transfer'
    model_ft = models.densenet121(pretrained=True)

# Freeze the feature extractor
for param in model_ft.parameters():
    param.requires_grad = False

# Unfreeze the last layer (if applicable)
last_layer_params = list(model_ft.parameters())[-2:]
for param in last_layer_params:
    param.requires_grad = True

model_ft.to(device)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Train the model
def train_model(model, criterion, optimizer, scheduler):
    since = time.time()
    
    best_acc = 0.0
    for epoch in range(30):  # Change to your desired number of epochs
        print('Epoch {}/{}'.format(epoch, 29))
        print('-'*10)
        
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
            
            epoch_loss = running_loss / len(train_dataset)
            epoch_acc = running_corrects / len(train_dataset)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            if phase == 'test':
                writer.add_scalar('Test/Loss', epoch_loss, epoch)
                writer.add_scalar('Test/Accuracy', epoch_acc, epoch)
                writer.flush()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

# Train the model
train_model(model_ft, criterion, optimizer_ft, scheduler)

# Save the entire model
torch.save(model_ft.state_dict(), 'model.pth')
