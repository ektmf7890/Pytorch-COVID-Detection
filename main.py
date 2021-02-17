from train_model import train_and_validate
from test import computeTestSetAccuracy
from dataset import ChestXRayDataset
from predict import predict

import os
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from datetime import date

today_date = date.today().strftime("%d_%m_%Y")

# Image Transformations
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
root_dir = 'COVID-19 Radiography Database'
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
test_data_size = len(test_dataset)

# DataLoader for each Dataset
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
# Load pretrained ResNet50 model and freeze parameters
resnet50 = models.resnet50(pretrained=True)
for param in resnet50.parameters():
    param.requires_grad = False

# Transform the final fully connected layer for transfer learning
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
optimizer = optim.Adam(params=resnet50.parameters())

if __name__ == "__main__":
    # Train model and get the best epoch
    # train_and_validate(data, train_data_size, valid_data_size, model, loss_func, optimizer, epochs=25)
    trained_model, best_epoch = train_and_validate(data, resnet50, loss_func, optimizer, 20)
    torch.save(trained_model, os.path.join(today_date, 'COVID19_final_trained_model.pt'))

    # Loading the best_epoch_model
    best_epoch_path = os.path.join(today_date, 'COVID19'+'_model_'+str(best_epoch)+'.pt')
    if os.path.exists(best_epoch_path):
        best_epoch_model = torch.load(os.path.join(today_date, 'COVID19'+'_model_'+str(best_epoch)+'.pt'))

    # Get accuracy of trained model on test set
    print('<Trained model>')
    computeTestSetAccuracy(trained_model, resnet50, loss_func, optimizer)


    
