import os
import torch
import numpy as np
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import pandas as pd
#from pycocotools.coco import COCO
from torchvision import datasets, transforms

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
    
class UCMDatasetMultiLabel(Dataset):
    def __init__(self, images_root, labels_path, transform=None):
        self.rootPath = images_root
        self.transform = transform
        self.labels_path = labels_path

        self.classes = ['airplane','bare-soil','buildings','cars','chaparral','court',
                        'dock','field','grass','mobile-home','pavement','sand','sea',
                        'ship','tanks','trees','water']

        # get the list of directories in the root path
        self.listOfDirs = next(os.walk(self.rootPath))[1]
        self.listOfDirs.sort()

        self.gt_dataFrame = pd.read_csv(self.labels_path, delimiter='\t')

        # load up all the filenames into a list
        # and also calculate the mean of the dataset
        self.sampleFilenames = []
        self.targets = []
        
        print("Creating dataset")
        for dir in self.listOfDirs:
            files = os.listdir(os.path.join(self.rootPath, dir))
            files.sort()    # make sure the files are in the same order as in the GT file
            #print(len(files))
            
            for file in files:
                filePath = os.path.join(self.rootPath, dir, file)
                self.sampleFilenames.append(filePath)
                
                # strip the extension so as to match it 
                # to the filename column of the dataframe
                df = self.gt_dataFrame.loc[self.gt_dataFrame['IMAGE\LABEL']==os.path.splitext(file)[0]]
                #tensorLabel = torch.tensor(df[self.classes].values)
                labels = df[self.classes].values
                labels = labels[0,...]  # it's 2-dim (1,17) convert to (17), a 1D array
                labels = labels.astype(float)    # otherwise they remain integers
                self.targets.append(labels)

        #print(len(self.sampleFilenames))
        #print(len(self.targets))
        self.label_freq = self.get_label_frequencies()
        print("Dataset created")

    def __len__(self):
        """
        return the total number of samples in this dataset
        """
        return len(self.sampleFilenames)

    def get_label_frequencies(self):
        """
        Return the frequency of each label in the dataset.
        """
        label_counts = np.zeros(len(self.classes))
        
        for labels in self.targets:
            label_counts += labels
        
        total_samples = len(self.targets)
        label_frequencies = label_counts / total_samples
        
        return label_frequencies.tolist(), label_counts.tolist()
    
    def __getitem__(self, idx):
        """
        return the next sample
        """
        image = Image.open(self.sampleFilenames[idx])

        if self.transform:
            image = self.transform(image)

        return image, self.targets[idx]


class AIDDatasetMultiLabel(Dataset):
    def __init__(self, images_root, labels_path, transform=None, split='both'):
        """
        Args:
            images_root: Path to the images directory (should contain images_tr and images_test folders)
            labels_path: Path to the multilabel.csv file
            transform: Transform to apply to images
            split: 'train', 'test', or 'both' to specify which split to use
        """
        self.rootPath = images_root
        self.transform = transform
        self.labels_path = labels_path
        self.split = split

        self.classes = ['airplane','bare-soil','buildings','cars','chaparral','court',
                        'dock','field','grass','mobile-home','pavement','sand','sea',
                        'ship','tanks','trees','water']

        # AID uses CSV format (comma-separated), not tab-separated like UCM
        self.gt_dataFrame = pd.read_csv(self.labels_path, delimiter=',')

        # load up all the filenames into a list
        self.sampleFilenames = []
        self.targets = []
        
        print("Creating AID dataset")
        
        # Determine which folders to process based on split
        folders_to_process = []
        if self.split == 'train' or self.split == 'both':
            train_path = os.path.join(self.rootPath, 'images_tr')
            if os.path.exists(train_path):
                folders_to_process.append(('images_tr', train_path))
        
        if self.split == 'test' or self.split == 'both':
            test_path = os.path.join(self.rootPath, 'images_test')
            if os.path.exists(test_path):
                folders_to_process.append(('images_test', test_path))
        
        for split_name, split_path in folders_to_process:
            # get the list of directories in the split path
            listOfDirs = next(os.walk(split_path))[1]
            listOfDirs.sort()
            
            for dir in listOfDirs:
                files = os.listdir(os.path.join(split_path, dir))
                files.sort()    # make sure the files are in the same order as in the GT file
                
                for file in files:
                    filePath = os.path.join(split_path, dir, file)
                    self.sampleFilenames.append(filePath)
                    
                    # strip the extension so as to match it 
                    # to the filename column of the dataframe
                    # AID naming convention: airport_1.jpg -> airport_1
                    file_key = os.path.splitext(file)[0]
                    df = self.gt_dataFrame.loc[self.gt_dataFrame['IMAGE\LABEL']==file_key]
                    
                    if len(df) > 0:
                        labels = df[self.classes].values
                        labels = labels[0,...]  # it's 2-dim (1,17) convert to (17), a 1D array
                        labels = labels.astype(float)    # otherwise they remain integers
                        self.targets.append(labels)
                    else:
                        print(f"Warning: No labels found for {file_key}")
                        # Create zero labels if no match found
                        labels = np.zeros(len(self.classes), dtype=float)
                        self.targets.append(labels)

        print(f"Found {len(self.sampleFilenames)} samples")
        print(f"Found {len(self.targets)} labels")
        self.label_freq = self.get_label_frequencies()
        print("AID dataset created")

    def __len__(self):
        """
        return the total number of samples in this dataset
        """
        return len(self.sampleFilenames)

    def get_label_frequencies(self):
        """
        Return the frequency of each label in the dataset.
        """
        label_counts = np.zeros(len(self.classes))
        
        for labels in self.targets:
            label_counts += labels
        
        total_samples = len(self.targets)
        label_frequencies = label_counts / total_samples
        
        return label_frequencies.tolist(), label_counts.tolist()
    
    def __getitem__(self, idx):
        """
        return the next sample
        """
        image = Image.open(self.sampleFilenames[idx])

        if self.transform:
            image = self.transform(image)

        return image, self.targets[idx]

"""
class COCODataset(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)
        self.classes = np.array([
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                        'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                        'hair drier', 'toothbrush'
                    ])

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros(80, dtype=torch.float)
        for obj in target:
            output[self.cat2cat[obj['category_id']]] = 1
        target = output

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)          # Squeeze the target tensor
        return img, target
"""

class COCODataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.num_classes = len(self.coco.getCatIds())
        self.cat_ids = self.coco.getCatIds()
        self.classes = np.array([
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                        'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                        'hair drier', 'toothbrush'
                    ])
        self.cat_id_to_label = {cat_id: idx for idx, cat_id in enumerate(self.cat_ids)}

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        labels = np.zeros(self.num_classes, dtype=np.float32)
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id in self.cat_id_to_label:
                labels[self.cat_id_to_label[cat_id]] = 1.0
                
        labels = labels.astype(float)
        return img, labels, #torch.tensor(labels)

    def __len__(self):
        return len(self.ids)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dotenv import load_dotenv
    load_dotenv()
    path_to_coco = os.getenv("PATH_TO_COCO")
    
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])



    dataset = COCODataset(path_to_coco + "/train2017", 
                          path_to_coco + "/annotations/instances_train2017.json", 
                          transform)
    
    for i in range(5):
        image, label = dataset[i+1750]
        
        plt.imshow(image.permute(1, 2, 0), aspect='auto')
        # Decode the labels
        labels = dataset.classes
        image_labels = [labels[i] for i in range(len(label)) if label[i]==1]
        print(f"Image {i} labels: "+str(image_labels))
        plt.axis('off')
        plt.savefig(f'MultilabelContrastive/images/sample/COCO_image{i}.png')
        #plt.show()
    
    # Plot the label frequencies as a histogram
    """
    label_freq, label_counts = dataset.label_freq
    plt.figure(figsize=(10,8))
    plt.bar(dataset.classes, label_counts)
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    plt.title('Label counts in UCM dataset')
    plt.savefig('MultilabelContrastive/images/label_freq_ucm.png')
    """