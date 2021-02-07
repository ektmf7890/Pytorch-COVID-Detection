# Image Transformations
from torchvision import transforms
from torchvision.models import resnet

image_transforms = { 
    'train': transforms.Compose([
        transforms.Resize(size=256),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # image size for resnet50: (224, 224)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid_test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),   # image size for resnet50: (224, 224)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}


# Create Train/Valid/Test Datasets
import os
from dataset import ChestXRayDataset

root_dir = 'COVID-19RadiographyDatabase'
class_names = ['covid', 'normal', 'viral']

train_image_dirs = {
    'covid': os.path.join(root_dir, 'train', 'covid'),
    'normal': os.path.join(root_dir, 'train', 'normal'),
    'viral': os.path.join(root_dir, 'train', 'viral')
}
valid_image_dirs = {
    'covid': os.path.join(root_dir, 'valid', 'covid'),
    'normal': os.path.join(root_dir, 'valid', 'normal'),
    'viral': os.path.join(root_dir, 'valid', 'viral')
}
test_image_dirs = {
    'covid': os.path.join(root_dir, 'test', 'covid'),
    'normal': os.path.join(root_dir, 'test', 'normal'),
    'viral': os.path.join(root_dir, 'test', 'viral')
}

train_dataset = ChestXRayDataset(
    image_dirs=train_image_dirs, 
    transform=image_transforms['train'], 
    class_names=class_names
)

valid_dataset = ChestXRayDataset(
    image_dirs=valid_image_dirs, 
    transform=image_transforms['valid_test'], 
    class_names=class_names
)

test_dataset = ChestXRayDataset(
    image_dirs=test_image_dirs, 
    transform=image_transforms['valid_test'], 
    class_names=class_names
)

train_data_size = len(train_dataset)
valid_data_size = len(valid_dataset)
test_data_size = len(valid_dataset)


# DataLoader for each Dataset
from torch.utils.data import DataLoader

batch_size = 6

data = {
    'train_dataloader': DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    ),

    'valid_dataloader': DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=True
    ),

    'test_dataloader': DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True
    )
}


# Creating the Model
from torchvision import models

# Load pretrained ResNet50 model and freeze parameters
resnet50 = models.resnet50(pretrained=True)
for param in resnet50.parameters():
    param.requires_grad = False

# Transform the final fully connected layer for transfer learning
import torch.nn as nn

fc_inputs = resnet50.fc.in_features
num_classes = len(class_names)

# Activation: ReLU
# Regularization: Dropuout
# Final layer: Softmax

resnet50.fc = nn.Sequential(
    nn.Linear(in_features=fc_inputs, out_features=256),
    nn.ReLU(),
    nn.Dropout(p=0.4),
    nn.Linear(in_features=256, out_features=num_classes),
    nn.LogSoftmax(dim=1) #dim=1 -> calcualte probability along row
)


# Loss: Negtive Log Likelihood
loss_func = nn.NLLLoss()

# Optimizer: Adam Optimization Alogrithm
import torch.optim as optim
optimizer = optim.Adam(params=resnet50.parameters())

# Getting the device
import torch
device = torch.device("cpu")


if __name__ == "__main__":

    from data_prepare import data_prepare
    data_prepare()

    # Train model and get the best epoch
    from train_model import train_and_validate
    trained_model, best_epoch = train_and_validate(
        model=resnet50,
        loss_func=loss_func,
        optimizer=optimizer,
        data=data,
        epochs=20
    )
    best_epoch_model = torch.load(f"COVID19_model_{best_epoch}.pt")

    # Get accuracy of trained model on test set
    from test import computeTestSetAccuracy
    print('<Trained model>')
    computeTestSetAccuracy(trained_model, loss_func=loss_func, optimizer=optimizer)

    # Get accuracy of best epoch model on test set
    from test import computeTestSetAccuracy
    print('<Best epoch model>')
    computeTestSetAccuracy(best_epoch_model, loss_func=loss_func, optimizer=optimizer)

    # Make prediction on random image
    from predict import predict
