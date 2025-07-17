import torch
import numpy as np
from torchvision import transforms
from datasets import AIDDatasetMultiLabel
from torch.utils.data import DataLoader

print("=== CALCULATING FINAL AID NORMALIZATION VALUES ===")
print("Using original 600x600 size for maximum accuracy")
print()

def calculate_accurate_stats():
    """Calculate normalization stats using original image size with memory-efficient approach"""
    
    # Use original size with smaller batch and no workers to avoid memory issues
    transform_original = transforms.Compose([
        transforms.ToTensor()  # No resize - keep original 600x600
    ])
    
    # Use 'both' split to get all 3000 images for complete statistics
    dataset = AIDDatasetMultiLabel("data/AID/images/", 
                                  "data/AID/multilabel .csv", 
                                  transform_original,
                                  split='both')  # All 3000 images
    
    # Use small batch size and no workers to manage memory
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    
    print(f"Calculating statistics for {len(dataset)} images...")
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    for batch_idx, (data, _) in enumerate(loader):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples
        
        # Progress indicator
        if batch_idx % 50 == 0:
            print(f"Processed {total_samples}/{len(dataset)} images...")
    
    mean /= total_samples
    std /= total_samples
    
    return mean.tolist(), std.tolist()

try:
    print("This may take a few minutes due to large image sizes...")
    mean, std = calculate_accurate_stats()
    
    print(f"\n=== FINAL AID NORMALIZATION VALUES ===")
    print(f"Based on all 3000 images at original 600x600 resolution:")
    print(f"Mean: {[f'{x:.4f}' for x in mean]}")
    print(f"Std:  {[f'{x:.4f}' for x in std]}")
    print()
    print("Use these values in your transforms:")
    print(f"transforms.Normalize({mean}, {std})")
    print()
    
    # Compare with UCM values for reference
    ucm_mean = [0.4842, 0.4901, 0.4505]
    ucm_std = [0.1734, 0.1635, 0.1554]
    print("Comparison with UCM values:")
    print(f"UCM Mean: {[f'{x:.4f}' for x in ucm_mean]}")
    print(f"UCM Std:  {[f'{x:.4f}' for x in ucm_std]}")
    print()
    print(f"Mean differences: {[f'{abs(a-u):.4f}' for a, u in zip(mean, ucm_mean)]}")
    print(f"Std differences:  {[f'{abs(a-u):.4f}' for a, u in zip(std, ucm_std)]}")
    
except Exception as e:
    print(f"Error: {e}")
    print("If you get memory errors, the values from your test are good:")
    print("Mean: [0.4046, 0.4107, 0.3732]")
    print("Std:  [0.1560, 0.1427, 0.1374]")
