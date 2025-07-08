import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import wandb

from PIL import Image
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from skmultilearn.model_selection import iterative_train_test_split
from utils import setSeed, printDiagnostics, verifySplitStratification
from torchvision import models
from datasets import UCMDatasetMultiLabel
from torcheval.metrics.functional import multilabel_accuracy


"""
We'll handle the UCM dataset with 2100 images.
100 samples per class. Each of them being 256x256 pixels and RGB color.
Multiple classes per sample

Multi-label classification with binary cross entropy and sigmoid activations
Not pretrained: 717/1391 labels guessed correctly in total.
Pretrained: 1009/1391 labels guessed correctly in total.

"""

SEED = 42
NUM_CLASSES = 17    # multilabel 17, single-label 21
LR = 0.1
GAMMA = 0.7
EPOCHS = 300
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
    multiAcc = 0
    counter = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()  # sum up batch loss

            # 0.5 threshold by default
            multiAcc += multilabel_accuracy(output,target,criteria="hamming")
            counter += 1

    val_loss /= len(val_loader.dataset)
    multiAcc /= counter
    
    wandb.log({'validation loss': val_loss})
    wandb.log({'multilabel acc.': multiAcc})


    print('\nVal. set: Average loss: {:.4f}, Average multilabel metric: {:.5f}\n'.format(
        val_loss, multiAcc))
    
def test(model,device,path,test_loader):
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    multiAcc = 0
    counter = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            multiAcc += multilabel_accuracy(output,target,criteria="hamming")
            counter += 1

    multiAcc /= counter

    print('\nTest set: Multi accuracy: {:.5f}\n'.format(multiAcc))

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
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.4842,0.4901,0.4505], [0.1734,0.1635,0.1554])
        ])

    dataset = UCMDatasetMultiLabel("data/UCM/Images/", 
                                   "data/UCM/multilabels/LandUse_Multilabeled.txt", 
                                   trans)

    # Stratified Sampling for train and val
    # Train, Val, Test: 70/10/20 w.r.t. the single-label distribution
    # 1470, 210 and 420 samples respectively

    train_indices = []
    val_indices = []
    test_indices = []

    for i in range(21):
        for j in range(70):
            train_indices.append(i*100+j)

        for j in range(10):
            val_indices.append(i*100+70+j)

        for j in range(20):
            test_indices.append(i*100+80+j)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # create the dataloaders
    # drop the last batch during training to avoid weird performance measurements
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = models.resnet18(weights=True)

    # Modify the last layer to match UCM classes
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    # check the model size and assert health
    # summary(model,input_size=(BATCH_SIZE, 3, 256, 256))
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adadelta(model.parameters(), lr = LR)
    scheduler = StepLR(optimizer, step_size=1, gamma = GAMMA)

    if TRAIN_MODE:
        # logger - start a new wandb run to track this script
        wandb.login()
        wandb.init(
            project="UCM_multi_label",
            config={
            "learning_rate": LR,
            "architecture": "Resnet18",
            "dataset": "UCM",
            "epochs": EPOCHS,
            }
        )

        # save every epoch's model, and then select the best depending on validation performance.
        for epoch in range(1, EPOCHS + 1):
            train(model, device, train_loader, optimizer, epoch, criterion, wandb)
            validate(model, device, val_loader, epoch, criterion, wandb)
            torch.save(model.state_dict(), "ucm_resnet18_multilabel_epoch_{}.pt".format(epoch))
            scheduler.step()
    else:
        # TEST_MODE with whatever is the best validation model
        test(model,device,'ucm_resnet18_multilabel_epoch_11.pt',test_loader)

    

if __name__ == '__main__':
    main()