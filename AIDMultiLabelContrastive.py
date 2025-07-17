import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from utils import setSeed, printDiagnostics, verifySplitStratification
import matplotlib.pyplot as plt
from torchvision import models
from datasets import AIDDatasetMultiLabel
from torch.backends import cudnn
from models import SupCEResNet, SupConResNet, LinearClassifier
from losses import MulSupCosineLossCustom, SupConLoss, MulSupConLossCustom, MulSupCosineLossCustomOneCrop, MulSupCosineLoss, WeightedMulSupConLossCustom, WeightedMulSupCosineLossCustom
from utils import TwoCropTransform
import wandb
import math
from torcheval.metrics.functional import multilabel_accuracy
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score

"""
We'll handle the AID dataset with 3000 images.
100 samples per class for training, 20 per class for test. Each of them being 600x600 pixels and RGB color.
Supervised contrastive loss for pretraining, followed by cross-entropy
Resnet-18 architecture
Multi-label supervised contrastive loss.

"""

SEED = 42
NUM_CLASSES = 17     # multilabel 17, single-label 21
LR = 0.1
GAMMA = 0.7
EPOCHS_PRETRAIN = 200
EPOCHS_TRAIN = 100 # 500 for pretraining, 100 for training
LOG_INTERVAL = 10
BATCH_SIZE_PRETRAIN = 128
BATCH_SIZE_TRAIN = 16
PRETRAIN_MODE = False
TRAIN_MODE = False

def trainContrastive(model, train_loader, optimizer, epoch, criterion, wandb):

    model.train()
    running_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(train_loader):

        # 32,3,32,32
        # 2 crops of the same image
        #images = torch.cat([images[0], images[1]], dim=0)               # comment for one crop case
        images = images.cuda(non_blocking=True)

        # 16,17
        labels = labels.cuda(non_blocking=True)

        # 16
        bsz = labels.shape[0]

        # 32, 128
        features = model(images)                        # model(images)

        # 16, 128 and 16, 128
        #f1, f2 = torch.split(features, [bsz, bsz], dim=0)                   # comment for one crop case

        # 16, 2, 128 -> everything is in order.
        #features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)        # comment for one crop case

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

def train(model, classifier, device, train_loader, optimizer, epoch, criterion, wandb, finetune=False):
    if finetune:
        model.train()
    else:
        model.eval()
    
    classifier.train()
    running_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # compute loss
        if finetune:
            output = classifier(model.encoder(images))
        else:
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
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.float().cuda(), target.cuda()

            output = classifier(model.encoder(data))
            val_loss += criterion(output, target).item()  # sum up batch loss
            
            # Collect all predictions and targets
            preds = torch.sigmoid(output).cpu().numpy() > 0.5
            all_preds.extend(preds)
            all_targets.extend(target.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    
    # Calculate macro-F1 score
    macro_f1 = f1_score(all_targets, all_preds, average='macro')

    wandb.log({'validation loss': val_loss})
    wandb.log({'macro-f1': macro_f1})
    print('\nVal. set: Average loss: {:.4f}, Macro-F1: {:.5f}\n'.format(
        val_loss, macro_f1))
    
def classWiseF1(all_preds, all_targets):
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    f1_scores = f1_score(all_targets, all_preds, average=None)
    # Print the number of instances for each class
    print(f"Number of instances for each class: {np.sum(all_targets, axis=0)}")
    print(f"F1 scores for each class: {f1_scores.round(3)}")

def test(model, classifier, device, test_loader, test_loader_default = None):
    model.eval()
    classifier.eval()
    counter = 0
    multiAcc = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.float().to(device), target.to(device)
            output = classifier(model.encoder(data))
            preds = torch.sigmoid(output) > 0.5
            # 0.5 threshold by default
            multiAcc += multilabel_accuracy(output,target,criteria="hamming")
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            counter += 1
    all_labels = test_loader.dataset.dataset.classes
    print(all_labels)
    # Plot first 5 images
    """
    index = 0
    for data, target in test_loader_default:
        # Plot the image and predictions & truth
        preds = all_preds[index]
        target = all_targets[index]
        all_labels = test_loader.dataset.dataset.classes
        pred_targets = [all_labels[i] for i in range(len(preds[0])) if preds[0][i]]
        true_targets = [all_labels[i] for i in range(len(target[0])) if target[0][i]]
        plt.imshow(data[0].permute(1, 2, 0).cpu().numpy(), aspect="auto")
        plt.axis("off")
        #plt.title(f"Predicted: {pred_targets}")
        #plt.text(110, 240, f"True: {str(true_targets)}", ha="center", fontsize=12)
        print(f"Image {index}: Predicted: {pred_targets}, True: {true_targets}")
        plt.savefig(f"MultilabelContrastive/images/aid_test_image_{index}.png")
        plt.clf()
        if index == 9:
            break
        index += 1
    """
    multiAcc /= counter
    print('\nTest set: Multi accuracy: {:.5f}\n'.format(multiAcc))
    #classWiseF1(all_preds, all_targets)

def calculate_metrics(model, classifier, device, dataloader):
    model.eval()
    classifier.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = classifier(model.encoder(inputs))
            preds = torch.sigmoid(outputs) > 0.5

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Mean Average Precision (mAP)
    mAP = average_precision_score(all_labels, all_preds, average='macro')

    # Macro-F1
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    # Micro-F1
    micro_f1 = f1_score(all_labels, all_preds, average='micro')

    print(f'Mean Average Precision (mAP): {mAP:.4f}')
    print(f'Macro-F1: {macro_f1:.4f}')
    print(f'Micro-F1: {micro_f1:.4f}')

def calculateLabelCooccurenceMatrix(dataset, num_classes):
    label_cooccurence_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(dataset)):
        labels = dataset[i][1]  # Assuming dataset[i][1] is the multi-hot encoded label
        for j in range(num_classes):
            if labels[j] == 1:
                for k in range(num_classes):
                    if j != k and labels[k] == 1:
                        label_cooccurence_matrix[j][k] += 1
    # Normalize the matrix row-wise 
    label_cooccurence_matrix = label_cooccurence_matrix / label_cooccurence_matrix.sum(axis=1, keepdims=True)
    return label_cooccurence_matrix

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
            transforms.Normalize([0.4024, 0.4090, 0.3708], [0.1564, 0.1431, 0.1378])  # AID-specific (all 3000 images, 600x600)
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.4024, 0.4090, 0.3708], [0.1564, 0.1431, 0.1378])  # AID-specific (all 3000 images, 600x600)
        ]
    )

    datasetWithTrainTransforms = None
    
    if PRETRAIN_MODE:
        datasetWithTrainTransforms = AIDDatasetMultiLabel("data/AID/images/", 
                                   "data/AID/multilabel .csv", 
                                   train_transform,
                                   split='train')
                                   #TwoCropTransform(train_transform))
    else:
        datasetWithTrainTransforms = AIDDatasetMultiLabel("data/AID/images/", 
                                   "data/AID/multilabel .csv", 
                                   train_transform,
                                   split='train')
        
    datasetWithTestTransforms = AIDDatasetMultiLabel("data/AID/images/", 
                                   "data/AID/multilabel .csv",
                                   test_transform,
                                   split='both')  # Use both for getting label frequencies and creating val/test splits
    default_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    #datasetWithDefaultTransforms = AIDDatasetMultiLabel("data/AID/images/", 
    #                               "data/AID/multilabel .csv", 
    #                               default_transform,
    #                               split='both')
    # For AID dataset: Train split has 2400 images (80 per class * 30 classes)
    # Test split has 600 images (20 per class * 30 classes)  
    # We'll use ALL train split data and divide it with same ratio as UCM for train/val
    # Final test uses the original test split

    ### Weighted Sampling
    label_freq = np.array(datasetWithTestTransforms.label_freq[0])  # Get frequencies, not counts
    #print("Label freq: ", label_freq)
    label_weights = 1.0 / (label_freq + 1e-8)  # Add small epsilon to avoid division by zero
    # Normalize label weights
    #label_weights = label_weights / label_weights.sum()
    ### Weighted Sampling
    
    #print("Label weights: ", label_weights)
    
    # Create datasets for train/val from training split and test from test split
    train_dataset_full = AIDDatasetMultiLabel("data/AID/images/", 
                                             "data/AID/multilabel .csv", 
                                             train_transform,
                                             split='train')  # 2400 samples
    
    val_test_dataset = AIDDatasetMultiLabel("data/AID/images/", 
                                           "data/AID/multilabel .csv", 
                                           test_transform,
                                           split='test')   # 600 samples for final testing
    
    # For AID: Use same train/val ratio as UCM within available training data
    # UCM: 70/(70+10) = 87.5% train, 10/(70+10) = 12.5% val from train+val data  
    # AID: Apply same ratio to 80 available training samples per class
    # 87.5% of 80 = 70 samples for train, 12.5% of 80 = 10 samples for val
    # This uses ALL available training data while maintaining UCM's methodology
    train_per_class = 70  # 87.5% of 80 training samples
    val_per_class = 10    # 12.5% of 80 training samples
    
    train_indices = []
    val_indices = []
    test_indices = list(range(len(val_test_dataset)))  # All test data (600 samples)
    
    # AID dataset processes directories alphabetically, 80 samples per class
    for class_idx in range(30):  # 30 classes in AID
        class_start = class_idx * 80  # 80 samples per class in train split
        
        # Train: first 70 samples of each class
        for j in range(train_per_class):
            train_indices.append(class_start + j)
        
        # Val: remaining 10 samples of each class
        for j in range(val_per_class):
            val_indices.append(class_start + train_per_class + j)
        
        # Now we use ALL 80 training samples: 70*30=2100 train, 10*30=300 val, 20*30=600 test

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(train_dataset_full, val_indices)  # Use same dataset as train for val  
    test_dataset = Subset(val_test_dataset, test_indices)
    #test_dataset_default = Subset(datasetWithDefaultTransforms, test_indices)

    #verifySplitStratification(train_dataset, val_dataset, test_dataset)
    
    if PRETRAIN_MODE:
        BATCH_SIZE = BATCH_SIZE_PRETRAIN
        EPOCHS = EPOCHS_PRETRAIN
    else:
        BATCH_SIZE = BATCH_SIZE_TRAIN
        EPOCHS = EPOCHS_TRAIN
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    #test_loader_default = DataLoader(test_dataset_default, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    # Calculate the label co-occurence matrix in the train dataset
    label_cooccurence_matrix = calculateLabelCooccurenceMatrix(train_dataset, NUM_CLASSES)

    if PRETRAIN_MODE:
        """
        Multilabel contrastive mode, push similar classes together,
        and the distinct classes far away from each other
        """
                
        # logger - start a new wandb run to track this script
        wandb.login()
        wandb.init(
            project="AID_multi_label_contrastive",
            config={
            "learning_rate": LR,
            "architecture": "Resnet18",
            "dataset": "AID_multi",
            "epochs": EPOCHS,
            }
        )

        model = SupConResNet(name='resnet18')
        #model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        #criterion = MulSupCosineLossCustomOneCrop(device, temperature=0.1, contrast_mode = "one")
        criterion = WeightedMulSupCosineLossCustom(t=0.1, co_oc_mat=label_cooccurence_matrix, device=device)
        
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        # check the model size and assert health
        # summary(model,input_size=(BATCH_SIZE, 3, 256, 256))

        #optimizer = optim.Adadelta(model.parameters(), lr = LR)
        optimizer = optim.SGD(model.parameters(),lr=LR,momentum=0.9,weight_decay=1e-4)
        #optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
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
        torch.save(model.state_dict(), f"MultilabelContrastive/save/AID/aid_resnet18_multilabel_WCosine_one_onecrop_{BATCH_SIZE}batch_{EPOCHS}epoch.pt")
        print(f"Model saved to MultilabelContrastive/save/AID/aid_resnet18_multilabel_WCosine_one_onecrop_{BATCH_SIZE}batch_{EPOCHS}epoch.pt")
    elif TRAIN_MODE:
        # logger - start a new wandb run to track this script
        wandb.login()
        wandb.init(
            project="AID_multi_label_contrastive_classifier",
            config={
            "learning_rate": LR,
            "architecture": "Resnet18",
            "dataset": "AID_multi",
            "epochs": EPOCHS,
            }
        )

        # now first train with cross entropy loss
        model = SupConResNet(name='resnet18')
        criterion = nn.BCEWithLogitsLoss()
        classifier = LinearClassifier(name='resnet18', num_classes=NUM_CLASSES)
        
        state_dict = torch.load('MultilabelContrastive/save/AID/aid_resnet18_multilabel_WCosine_one_onecrop_128batch_200epoch.pt', map_location="cpu")

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
        
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        #optimizer = optim.SGD(classifier.parameters(), lr=LR,momentum=0.9, weight_decay=0)
        optimizer = optim.Adadelta(classifier.parameters(), lr = LR)
        #scheduler = StepLR(optimizer, step_size=1, gamma = GAMMA)
        #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        for epoch in range(1, EPOCHS + 1):
            # adjust the lr
            lr = LR
            steps = np.sum(epoch > np.asarray([60,75,90]))
            if steps > 0:
               lr = lr * (0.2**steps)
            
            print("The learning rate is {}".format(lr))

            train(model, classifier, device, train_loader, optimizer, epoch, criterion, wandb, finetune=False)
            validate(val_loader, model, classifier, criterion, wandb)
            #scheduler.step()
        torch.save(classifier.state_dict(), f"MultilabelContrastive/save/AID/aid_linear_classifier.pt")
        #torch.save(classifier.state_dict(), f"MultilabelContrastive/save/AID/aid_linear_WCosine_one_onecrop_200_128batch_16batchtrain_{EPOCHS}epoch_classifier.pt")
        #print(f"Model saved to MultilabelContrastive/save/AID/aid_linear_WCosine_one_onecrop_200_128batch_16batchtrain_{EPOCHS}epoch_classifier.pt")
    else:
        model = SupConResNet(name='resnet18')
        classifier = LinearClassifier(name='resnet18', num_classes=NUM_CLASSES)
        state_dict = torch.load('MultilabelContrastive/save/AID/aid_resnet18_multilabel_WCosine_one_onecrop_128batch_200epoch.pt', map_location="cpu")
        #classifier.load_state_dict(torch.load('MultilabelContrastive/save/AID/aid_linear_WCosine_one_onecrop_200_128batch_16batchtrain_100epoch_classifier.pt', weights_only=True))
        classifier.load_state_dict(torch.load('MultilabelContrastive/save/AID/aid_linear_classifier.pt', weights_only=True))

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
        test(model, classifier, device, test_loader)
        calculate_metrics(model, classifier, device, test_loader)

if __name__ == '__main__':
    main()