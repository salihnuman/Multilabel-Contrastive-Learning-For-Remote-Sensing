import argparse
from typing import Counter
import torch
import os
from os import listdir
from os.path import isfile, join
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from utils import setSeed, printDiagnostics
import matplotlib.pyplot as plt

import wandb
wandb.login()

"""
We'll handle the UCM dataset with 2100 images.
21 classes, 100 samples per class. Each of them being 256x256 pixels and RGB color.

Single label, custom architecture.

"""

SEED = 42
NUM_CLASSES = 21
LR = 0.1
GAMMA = 0.7
EPOCHS = 25
LOG_INTERVAL = 10
BATCH_SIZE = 16

class UCMDataset(Dataset):
    def __init__(self, root, transform=None):
        self.rootPath = root
        self.transform = transform

        # get the list of folders in the root path
        self.listOfClasses = next(os.walk(self.rootPath))[1]
        self.listOfClasses.sort()
        #print(self.listOfClasses)

        # load up all the filenames into a list
        # and also calculate the mean of the dataset
        self.sampleFilenames = []
        self.targets = []
        
        print("Creating dataset")
        for dir in self.listOfClasses:
            files = os.listdir(os.path.join(self.rootPath, dir))
            #print(len(files))
            for file in files:
                filePath = os.path.join(self.rootPath, dir, file)
                self.sampleFilenames.append(filePath)
                self.targets.append(self.listOfClasses.index(dir))

        #print(len(self.sampleFilenames))
        #print(len(self.targets))
        print("Dataset created")

    def __len__(self):
        """
        return the total number of samples in this dataset
        """
        return len(self.sampleFilenames)

    def __getitem__(self, idx):
        """
        return the next sample
        """
        image = Image.open(self.sampleFilenames[idx])

        if self.transform:
            image = self.transform(image)

        return image, self.targets[idx]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)  # input,output,kernel_size,stride, padding
        self.conv2 = nn.Conv2d(64, 32, 3, 1, 1)

        self.pool = nn.MaxPool2d(2, 2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, NUM_CLASSES)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = self.gap(x)             # Global Average Pooling
        x = torch.flatten(x, 1)     # Flatten for fully connected layer
        x = self.fc(x)              # Final output
        
        return x  

def train(model, device, train_loader, optimizer, epoch, running_loss, wandb):

    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % LOG_INTERVAL == 0:
            
            wandb.log({'training loss': running_loss / LOG_INTERVAL})
            running_loss = 0.0
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def validate(model, device, val_loader, epoch, wandb):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    
    wandb.log({'validation loss': val_loss})

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

def main():
    use_cuda = torch.cuda.is_available()
    setSeed(SEED)
   
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #printDiagnostics()
    #printDatasetStats(dataset)

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4842,0.4901,0.4505], [0.1734,0.1635,0.1554])
        ])

    dataset = UCMDataset("../../data/UCM/Images/", trans)

    # Stratified Sampling for train and val
    # Train, Val, Test: 70/10/20
    # 1470, 210 and 420 samples respectively
    train_indices, test_indices, train_label, _ = train_test_split(
                                                    range(len(dataset)),
                                                    dataset.targets,
                                                    stratify=dataset.targets,
                                                    test_size=0.2,
                                                    random_state=SEED)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_indices, val_indices, _, _ = train_test_split(
                                        range(len(train_dataset)),
                                        train_label,
                                        stratify=train_label,
                                        test_size=0.125,
                                        random_state=SEED)

    # tricky bug: be careful not to overwrite train_dataset
    # before you create the val_dataset..it cost me a couple of hours
    val_dataset = Subset(train_dataset, val_indices)
    train_dataset = Subset(train_dataset, train_indices)

    """ # verify that the split is stratified
    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))

    # let's print the respective labels of each class at each subset, to make sure it's stratified
    trainClasses = [label for _, label in train_dataset]
    valClasses = [label for _, label in val_dataset]
    testClasses = [label for _, label in test_dataset]

    trainClasses.sort()
    valClasses.sort()
    testClasses.sort()
    
    print(Counter(trainClasses))
    print(Counter(valClasses))
    print(Counter(testClasses)) """

    """ # now visualize the images, to make sure they look right.
    for image, label in train_dataset:
        plt.imshow(image.permute(1, 2, 0))
        plt.show()
        break """


    # create the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = Net().to(device)

    # check the model size and assert health
    # summary(model,input_size=(BATCH_SIZE, 3, 256, 256))

    # logger - start a new wandb run to track this script
    wandb.init(
        project="UCM_single_label",
        config={
        "learning_rate": LR,
        "architecture": "CNN2D",
        "dataset": "UCM_single",
        "epochs": EPOCHS,
        }
    )
 
    running_loss = 0.0
    optimizer = optim.Adadelta(model.parameters(), lr = LR)
    scheduler = StepLR(optimizer, step_size=1, gamma = GAMMA)

    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch, running_loss, wandb)
        validate(model, device, val_loader, epoch, wandb)
        scheduler.step()

if __name__ == '__main__':
    main()