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
from torch.backends import cudnn
from models import SupCEResNet, SupConResNet, LinearClassifier
from losses import SupConLoss
from utils import TwoCropTransform
import wandb
import math

"""
We'll handle the UCM dataset with 2100 images.
100 samples per class. Each of them being 256x256 pixels and RGB color.
Let's try supervised contrastive loss, and see if we can reduce it.
Resnet-18 architecture
Single label supervised contrastive loss.


92.85% during validation, and 91% during testing.
It was 58% at test, 56.66% at validation..but with 25 epochs.
Here we used 300 epochs of pretraining, and 100 epochs for training.
Contrastive learning is definitely promising.
Next let's try to extend this to the multi-label scenario with mulsupcon.

"""

SEED = 42
NUM_CLASSES = 21
LR = 0.1    # 0.5 for training
GAMMA = 0.7 
EPOCHS = 500 # 500 for training
LOG_INTERVAL = 10
BATCH_SIZE = 16 # 16 for training
PRETRAIN_MODE = True
TRAIN_MODE = False

def trainContrastive(model, train_loader, optimizer, epoch, criterion, wandb):

    model.train()
    running_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(train_loader):

        images = torch.cat([images[0], images[1]], dim=0)
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        
        # [32,128]
        features = model(images)
        
        # split into 2, each of size bsz
        # each is [16,128]
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        
        loss = criterion(features, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # to avoid the initial zero-th case
        if batch_idx % LOG_INTERVAL == 0 and batch_idx >= LOG_INTERVAL:
            wandb.log({'training loss': running_loss / LOG_INTERVAL})
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx * len(images)//2, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), running_loss / LOG_INTERVAL))
            running_loss = 0.0

def train(model, classifier, device, train_loader, optimizer, epoch, criterion, wandb):

    model.eval()
    classifier.train()
    running_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # to avoid the initial zero-th case
        if batch_idx % LOG_INTERVAL == 0 and batch_idx >= LOG_INTERVAL:
            wandb.log({'training loss': running_loss / LOG_INTERVAL})
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx * len(images), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), running_loss / LOG_INTERVAL))
            running_loss = 0.0

def validate(val_loader, model, classifier, criterion, wandb):
    model.eval()
    classifier.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.float().cuda(), target.cuda()

            output = classifier(model.encoder(data))
            val_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = 100. * correct / len(val_loader.dataset)
    
    wandb.log({'validation loss': val_loss})
    wandb.log({'validation acc.': val_acc})

    print('\nVal. set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),val_acc))

def test(model, classifier, device, test_loader):
    model.eval()
    classifier.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.float().to(device), target.to(device)
            output = classifier(model.encoder(data))
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

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            #transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.4842,0.4901,0.4505], [0.1734,0.1635,0.1554])
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.4842,0.4901,0.4505], [0.1734,0.1635,0.1554])
        ]
    )

    datasetWithTrainTransforms = None
    
    if PRETRAIN_MODE:
        datasetWithTrainTransforms = UCMDataset("data/UCM/Images/", TwoCropTransform(train_transform))
    else:
        datasetWithTrainTransforms = UCMDataset("data/UCM/Images/", train_transform)

    # Stratified Sampling for train and val
    # Train, Val, Test: 70/10/20
    # 1470, 210 and 420 samples respectively
    train_indices, test_val_indices, _, test_val_labels = train_test_split(range(len(datasetWithTrainTransforms)),
                                                        datasetWithTrainTransforms.targets,
                                                        stratify=datasetWithTrainTransforms.targets,
                                                        test_size=0.3,
                                                        random_state=SEED)

    train_dataset = Subset(datasetWithTrainTransforms, train_indices)

    datasetWithTestTransforms = UCMDataset("data/UCM/Images/", test_transform)
    test_val_dataset = Subset(datasetWithTestTransforms, test_val_indices)

    test_indices, val_indices, _, _ = train_test_split(range(len(test_val_dataset)),
                                                        test_val_labels,
                                                        stratify=test_val_labels,
                                                        test_size=0.3333333,
                                                        random_state=SEED)
    
    val_dataset = Subset(test_val_dataset, val_indices)
    test_dataset = Subset(test_val_dataset, test_indices)

    #verifySplitStratification(train_dataset, val_dataset, test_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    if PRETRAIN_MODE:
        # logger - start a new wandb run to track this script
        wandb.login()
        wandb.init(
            project="UCM_single_label_contrastive",
            config={
            "learning_rate": LR,
            "architecture": "Resnet18",
            "dataset": "UCM_single",
            "epochs": EPOCHS,
            }
        )

        model = SupConResNet(name='resnet18')
        criterion = SupConLoss(0.1)
        
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        # check the model size and assert health
        # summary(model,input_size=(BATCH_SIZE, 3, 256, 256))

        #optimizer = optim.Adadelta(model.parameters(), lr = LR)

        optimizer = optim.SGD(model.parameters(),lr=LR,momentum=0.9,weight_decay=1e-4)
        #scheduler = StepLR(optimizer, step_size=1, gamma = GAMMA)

        # save every epoch's model, and then select the best depending on validation performance.
        for epoch in range(1, EPOCHS + 1):
            # adjust the lr
            lr = LR
            eta_min = lr * (0.1**3)
            lr = (
                eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / EPOCHS)) / 2
            )
        
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            print("The learning rate is {}".format(lr))

            trainContrastive(model, train_loader, optimizer, epoch, criterion, wandb)
            #scheduler.step()
        # save the last model
        torch.save(model.state_dict(), "ucm_resnet18_contrastive_epoch.pt".format(epoch))
    elif TRAIN_MODE:
        # logger - start a new wandb run to track this script
        wandb.login()
        wandb.init(
            project="UCM_single_label_contrastive_classifier",
            config={
            "learning_rate": LR,
            "architecture": "Resnet18",
            "dataset": "UCM_single",
            "epochs": EPOCHS,
            }
        )

        # now first train with cross entropy loss
        model = SupConResNet(name='resnet18')
        criterion = torch.nn.CrossEntropyLoss()
        classifier = LinearClassifier(name='resnet18', num_classes=NUM_CLASSES)

        state_dict = torch.load('ucm_resnet18_contrastive_epoch.pt', map_location="cpu")

        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

        #optimizer = optim.SGD(classifier.parameters(), lr=LR,momentum=0.9, weight_decay=0)
        optimizer = optim.Adadelta(classifier.parameters(), lr = LR)
        scheduler = StepLR(optimizer, step_size=1, gamma = GAMMA)

        for epoch in range(1, EPOCHS + 1):
            # adjust the lr
            """ lr = LR
            steps = np.sum(epoch > np.asarray([60,75,90]))
            if steps > 0:
                lr = lr * (0.2**steps)
            
            print("The learning rate is {}".format(lr)) """

            train(model, classifier, device, train_loader, optimizer, epoch, criterion, wandb)
            validate(val_loader, model, classifier, criterion, wandb)
            torch.save(classifier.state_dict(), "ucm_linear_epoch_{}.pt".format(epoch))
            scheduler.step()
    else:
        model = SupConResNet(name='resnet18')
        classifier = LinearClassifier(name='resnet18', num_classes=NUM_CLASSES)
        state_dict = torch.load('ucm_resnet18_contrastive_epoch.pt', map_location="cpu")
        classifier.load_state_dict(torch.load('ucm_linear_epoch_43.pt', weights_only=True))

        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

        # TEST_MODE with whatever is the best validation model
        test(model, classifier, device,test_loader)


if __name__ == '__main__':
    main()