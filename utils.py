import torch
import numpy as np
import random

from torch.utils.data import DataLoader
from typing import Counter

class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def printDiagnostics():
    print("There are " + str(torch.cuda.device_count()) + " device(s) on this system.")
    print("The current device is " + str(torch.cuda.get_device_name(torch.cuda.current_device())))
    print("Pytorch version is " + torch.__version__)
    print("Cuda version is " + torch.version.cuda)

def verifySplitStratification(train_dataset, val_dataset, test_dataset):

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
    print(Counter(testClasses))


def printDatasetStats(dataset):

    loader = DataLoader(dataset,batch_size=10,num_workers=0,shuffle=False)

    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)

    print(mean)
    print(std)