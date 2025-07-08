import torch
import numpy as np

def forward(labels, co_oc_matrix):
    #batch_size = embeddings.size(0)

    # Normalize co-occurrence matrix
    co_oc_matrix = co_oc_matrix / co_oc_matrix.sum(dim=1, keepdim=True)

    # Compute fuzzy labels based on co-occurrence matrix
    fuzzy_labels = torch.matmul(labels, co_oc_matrix)  # Shape: [batch_size, num_labels]
    return fuzzy_labels

def calculateLabelCooccurenceMatrix(dataset, num_classes):
    label_cooccurence_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(dataset)):
        labels = dataset[i][1]  # Assuming dataset[i][1] is the multi-hot encoded label
        for j in range(num_classes):
            if labels[j] == 1:
                for k in range(num_classes):
                    if j != k and labels[k] == 1:
                        label_cooccurence_matrix[j][k] += 1
    return label_cooccurence_matrix


num_classes = 4

# Create a dummy dataset
class DummyDataset:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.data = [
            (i, [1, 0, 1, 0]) for i in range(5)
        ] + [
            (i, [0, 1, 0, 1]) for i in range(5, 10)
        ]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

num_classes = 4
dataset = DummyDataset(num_classes)
co_oc_matrix = calculateLabelCooccurenceMatrix(dataset, num_classes)
print("Labels:", [dataset.data[i][1] for i in range(len(dataset))])
print("Label co-occurence matrix:\n", co_oc_matrix)