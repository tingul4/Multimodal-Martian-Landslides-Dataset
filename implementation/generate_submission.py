import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import tifffile
import numpy as np
import zipfile
import shutil

from dataset import MarsLSDataset
from model import MarsLSNet

def generate_submission(args):
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Model
    print(f"Loading model architecture with patch_size={args.patch_size}...")
    model = MarsLSNet(in_channels=7, num_classes=1, base_filters=48, patch_size=args.patch_size).to(device)
    
    # Load Checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
         model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    # Dataset & Loader
    print(f"Initializing Test Dataset from {args.data_dir}...")
    test_dataset = MarsLSDataset(root_dir=args.data_dir, partition='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Output Directory
    temp_dir = 'submission_temp'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    print(f"Starting inference on {len(test_dataset)} images...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch['image'].to(device)
            image_names = batch['id'] # List of filenames
            
            # Forward
            preds = model(images) # (B, 1, H, W)
            
            # Apply Sigmoid
            pred_probs = torch.sigmoid(preds)
            
            # Resize if needed (Target is usually 128x128)
            if pred_probs.shape[-1] != 128:
                pred_probs = F.interpolate(pred_probs, size=(128, 128), mode='bilinear', align_corners=False)
            
            # Threshold
            pred_masks = (pred_probs > 0.5).float()
            
            # Convert to numpy
            pred_masks = pred_masks.cpu().numpy().astype(np.uint8) # (B, 1, 128, 128)
            
            # Save Batch
            for i, mask_name in enumerate(image_names):
                mask = pred_masks[i, 0] # (128, 128)
                save_path = os.path.join(temp_dir, mask_name)
                tifffile.imwrite(save_path, mask)
            
    # Zip
    zip_filename = args.output_zip
    print(f"Compressing to {zip_filename}...")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Arcname should be the filename only (root level in zip)
                zipf.write(file_path, arcname=file)
                
    # Cleanup
    shutil.rmtree(temp_dir)
    print(f"Done! Submission saved to {zip_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Submission for MarsLS-Net')
    parser.add_argument('--data_dir', type=str, default='/raid/danielchen/Mars-LS-challenge/Dataset', help='Path to dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best_model.pth')
    parser.add_argument('--output_zip', type=str, default='submission.zip', help='Output zip filename')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size used in training')
    
    args = parser.parse_args()
    
    generate_submission(args)
