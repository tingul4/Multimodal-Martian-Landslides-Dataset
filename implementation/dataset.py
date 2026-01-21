import os
import numpy as np
import torch
from torch.utils.data import Dataset
try:
    import tifffile
except ImportError:
    print("Warning: tifffile not found. Please install it using `pip install tifffile`")

class MarsLSDataset(Dataset):
    def __init__(self, root_dir, partition='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and masks.
                               e.g., '/raid/danielchen/Mars-LS-challenge/Dataset'
            partition (string): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.partition = partition
        self.transform = transform
        
        self.image_dir = os.path.join(self.root_dir, partition, 'images')
        self.mask_dir = os.path.join(self.root_dir, partition, 'masks')
        
        # Checking if directories exist
        if not os.path.exists(self.image_dir):
             raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if self.partition != 'test' and not os.path.exists(self.mask_dir):
             # For test partition, masks might not be available or required depending on the challenge phase
             # But the folder structure implies they might be there or not. 
             # Assuming standard supervised learning setup where val has masks.
             raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        self.images = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.tif')])
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        try:
            image = tifffile.imread(img_path)
        except NameError:
             raise ImportError("tifffile library is not installed.")

        # Handle different channel orderings if necessary
        # Assuming image shape is (H, W, C) or (C, H, W). 
        # Tifffile usually returns (H, W, C) for multi-channel tiffs if preserved.
        # Check dimensions. Expected 7 channels, 128x128.
        if image.ndim == 2:
            # Single channel case (unlikely for inputs but possible for masks)
            image = image[:, :, np.newaxis]
        
        # Transpose to (C, H, W) for PyTorch if input is (H, W, C)
        # Usually (128, 128, 7) -> (7, 128, 128)
        if image.shape[2] == 7 or image.shape[2] == 10: # Assuming channels are last
            image = image.transpose((2, 0, 1))
        
        # Normalization (Simple Min-Max or Standardization can be added here)
        # For now, converting to float32
        image = image.astype(np.float32)
        
        # Normalize to [0, 1] to prevent exploding gradients in higher order terms (x^2) of ConvPE
        # Assuming Max Value ~5211 based on inspection, but potentially higher. 
        # Using per-image normalization for safety or fixed scaling if domain known.
        # Robust per-image min-max:
        img_min = image.min()
        img_max = image.max()
        if img_max > img_min:
            image = (image - img_min) / (img_max - img_min)
        else:
            image = np.zeros_like(image)


        sample = {'image': image, 'id': img_name}

        if self.partition != 'test':
            mask_path = os.path.join(self.mask_dir, img_name)
            if os.path.exists(mask_path):
                mask = tifffile.imread(mask_path)
                # Mask should be (H, W) or (H, W, 1)
                if mask.ndim == 3:
                     # If (H, W, C), take the first channel or squeeze
                     if mask.shape[2] == 1:
                         mask = mask[:, :, 0]
                     else:
                         # Ensure it's single channel binary mask
                         mask = mask[:, :, 0] 
                
                mask = mask.astype(np.float32)
                # Add channel dimension for BCE loss: (1, H, W)
                mask = mask[np.newaxis, :, :]
                
                # Binarize if necessary (assuming GT is 0 and 1, or 0 and 255)
                # mask = (mask > 0).astype(np.float32) 
                
                sample['mask'] = mask
            else:
                # Fallback if mask file missing but partition is train/val (should not happen if data is clean)
                print(f"Warning: Mask for {img_name} not found in {self.mask_dir}")

        if self.transform:
            sample = self.transform(sample)

        return sample
