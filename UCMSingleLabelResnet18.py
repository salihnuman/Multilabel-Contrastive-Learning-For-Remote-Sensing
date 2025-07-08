import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from utils import setSeed, printDiagnostics, verifySplitStratification
import matplotlib.pyplot as plt
from torchvision import models
from datasets import UCMDataset
import wandb

"""
We'll handle the UCM dataset with 2100 images.
21 classes, 100 samples per class. Each of them being 256x256 pixels and RGB color.
Resized to 32x32 so as to be able to compare against the contrastive one, which didn't
fit into memory otherwise.

256x256
Single label cross entropy, with a Resnet-18:
25 epochs, training from scratch, 80% at test, 78% at validation
25 epochs, pretrained, 97% at test, 97% at validation


32x32
Single label cross entropy, with a Resnet-18:
25 epochs, training from scratch, 58% at test, 56.66% at validation
25 epochs, pretrained, 82% at test, 86.19% at validation
"""

SEED = 42
NUM_CLASSES = 21
LR = 0.1
GAMMA = 0.7
EPOCHS = 500
LOG_INTERVAL = 10
BATCH_SIZE = 16
TRAIN_MODE = False

def train(model, device, train_loader, optimizer, epoch, criterion, wandb):

    model.train()
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # to avoid the initial zero-th case
        if batch_idx % LOG_INTERVAL == 0 and batch_idx >= LOG_INTERVAL:
            wandb.log({'training loss': running_loss / LOG_INTERVAL})
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), running_loss / LOG_INTERVAL))
            running_loss = 0.0

def validate(model, device, val_loader, epoch, criterion, wandb):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = 100. * correct / len(val_loader.dataset)
    
    wandb.log({'validation loss': val_loss})
    wandb.log({'validation acc.': val_acc})


    print('\nVal. set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),val_acc))
    
def test(model,device,path,test_loader):
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),test_acc))

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
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.4842,0.4901,0.4505], [0.1734,0.1635,0.1554])
        ])

    dataset = UCMDataset("data/UCM/Images/", trans)

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

    #verify that the split is stratified
    #verifySplitStratification(train_dataset, val_dataset, test_dataset)

    """ # now visualize the images, to make sure they look right.
    for image, label in train_dataset:
        plt.imshow(image.permute(1, 2, 0))
        plt.show()
        break """

    # create the dataloaders
    # drop the last batch during training to avoid weird spikes
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # pretrained with imagenet
    model = models.resnet18(weights='IMAGENET1K_V1')

    # from scratch
    # model = models.resnet18()

    # Modify the last layer to match UCM classes (21 classes)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    # check the model size and assert health
    # summary(model,input_size=(BATCH_SIZE, 3, 256, 256))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr = LR)
    scheduler = StepLR(optimizer, step_size=1, gamma = GAMMA)

    if TRAIN_MODE:
        # logger - start a new wandb run to track this script
        wandb.login()
        wandb.init(
            project="UCM_single_label",
            config={
            "learning_rate": LR,
            "architecture": "Resnet18",
            "dataset": "UCM_single",
            "epochs": EPOCHS,
            }
        )

        # save every epoch's model, and then select the best depending on validation performance.
        for epoch in range(1, EPOCHS + 1):
            train(model, device, train_loader, optimizer, epoch, criterion, wandb)
            validate(model, device, val_loader, epoch, criterion, wandb)
            torch.save(model.state_dict(), "ucm_resnet18_epoch_{}.pt".format(epoch))
            scheduler.step()
    else:
        # TEST_MODE with whatever is the best validation model
        test(model,device,'ucm_resnet18_epoch_11.pt',test_loader)

    

if __name__ == '__main__':
    main()