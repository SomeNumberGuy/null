import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
from torchvision.datasets import CocoDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
from torchvision import utils
from dataset2 import MaskRCNNDataset, loadData

from multiprocessing import freeze_support
from tqdm import tqdm
import random

def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)

    return model

if __name__ == '__main__':
    freeze_support()
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA is available! Using GPU.')
    else:
        exit('CUDA is not available. Exiting.')

    # Define your dataset and dataloader
    dataset = MaskRCNNDataset('C:\\Users\\test\\Downloads\\test-dataset2', device)
    total_samples = len(dataset)
    subset_size = int(0.0001 * total_samples)
    subset_indices = random.sample(range(total_samples), subset_size)
    dataset = Subset(dataset, subset_indices)

    # Assuming you have defined num_classes somewhere
    num_classes = 300  # 91 classes in COCO dataset

    # Get the model
    model = get_instance_segmentation_model(num_classes).to(device)

    # Define your optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params)

    # Define your learning rate scheduler if needed
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Define your data loader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)#, collate_fn=utils.collate_fn)

    # Define your training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        #running_loss = 0.
        #last_loss = 0.
        #amount = len(data_loader)
        #print(f"epoch {epoch}")
        # Training
        images, targets = loadData('C:\\Users\\Mario\\Downloads\\test-dataset2')
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        print(epoch, 'loss:', losses.item())
        if epoch % 500 == 0:
            torch.save(model.state_dict(), str(epoch) + ".torch")
        continue
        for images, targets in tqdm(data_loader):
            images = list(image for image in images)
            #images = list(images[0])
            #print(type(images),"|",type(targets),"len=",len(targets),"  [0]=",type(targets[0]))
            #print("IMAGE\n",images.size())
            #print("TARGET\n",targets[0].keys())
            #print(targets[0]['boxes'].size(),targets[0]['labels'].size(),targets[0]['masks'].size())
            #print(targets)
            #exit()

            if len(targets) == 0 or torch.sum(targets[0]['boxes']) < 1:
                continue
            #print(targets[0]['boxes'])

            #targets = [targets]#[{k: v for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            #print(losses)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            running_loss += losses.item()/amount
        print("epoch:",epoch,"loss:",running_loss)
        running_loss = 0
        # Update the learning rate
        lr_scheduler.step()

    # Save the trained model
    torch.save(model.state_dict(), 'mask_rcnn_model.pth')
