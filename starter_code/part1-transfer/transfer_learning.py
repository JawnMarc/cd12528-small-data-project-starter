# Starter code for Part 1 of the Small Data Solutions Project
#

# Set up image data for train and test
import os

import torch
# import torch.nn as nn
from torch import nn
# import torch.optim as optim
from torch import optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from TrainModel import train_model
from TestModel import test_model
from torchvision import models


# use this mean and sd from torchvision transform documentation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Set up Transforms (train, val, and test)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)

])

# Set up DataLoaders (train, val, and test)
batch_size = 32
num_workers = 2

# <<<YOUR CODE HERE>>>
# hint, create a variable that contains the class_names. You can get them from the ImageFolder

data_dir = 'imagedata-50'  # data diretory in root folder
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

trainset = datasets.ImageFolder(train_dir, transform=train_transform)
validset = datasets.ImageFolder(valid_dir, transform=val_test_transform)
testset = datasets.ImageFolder(test_dir, transform=val_test_transform)

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(
    validset, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, num_workers=num_workers)

class_names = trainset.classes


# Using the VGG16 model for transfer learning
# 1. Get trained model weights
# 2. Freeze layers so they won't all be trained again with our data
# 3. Replace top layer classifier with a classifer for our 3 categories

# <<<YOUR CODE HERE>>>
# model = models.vgg16(weights='DEFAULT')
model = models.resnet50(weights='DEFAULT')

# freeze fc layer
for param in model.parameters():
    model.requires_grad = False

# in_feat = model.classifier[6].in_features
in_feat = model.fc.in_features
hidden_unit = 512

# design custom classifier
classifier = nn.Sequential(
    nn.Linear(in_feat, hidden_unit),
    nn.ReLU(),
    nn.Dropout(0.6),
    nn.Linear(hidden_unit, 3)
)

# replace pre-trained classifier
# model.classifier[6] = classifier
model.fc = classifier


# Train model with these hyperparameters
# 1. num_epochs
# 2. criterion
# 3. optimizer
# 4. train_lr_scheduler

# <<<YOUR CODE HERE>>>
num_epochs = 5
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.classifier.parameters(), lr=0.0004)
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
train_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)


# When you have all the parameters in place, uncomment these to use the functions imported above
def main():
    trained_model = train_model(model, criterion, optimizer, train_lr_scheduler,
                                train_loader, valid_loader, num_epochs=num_epochs)
    test_model(test_loader, trained_model, class_names)


if __name__ == '__main__':
    main()
    print("done")
