import os
import argparse
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import random

from dataset import MarsLSDataset
from model import MarsLSNet
from utils import compute_iou, save_checkpoint, calculate_metrics

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):
    # Set Seed
    setup_seed(args.seed)

    # Setup Logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.save_dir, f'run_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    
    # File Logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    
    # TensorBoard Writer
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"Logging to {log_dir}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Data
    logger.info("Initializing Data Loaders...")
    train_dataset = MarsLSDataset(root_dir=args.data_dir, partition='train')
    val_dataset = MarsLSDataset(root_dir=args.data_dir, partition='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    logger.info("Initializing Model...")
    model = MarsLSNet(in_channels=7, num_classes=1, base_filters=48, patch_size=args.patch_size).to(device)

    # Loss and Optimizer
    criterion_bce = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_miou = 0.0

    logger.info("Starting Training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss_epoch = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for i, batch in enumerate(loop):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion_bce(outputs, masks)
            
            # Check for NaN
            if torch.isnan(loss):
                logger.error(f"Loss is NaN at epoch {epoch+1}, batch {i}")
                continue

            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            train_loss_epoch += loss.item()
            loop.set_postfix(loss=loss.item())
            
            # Global Step for TensorBoard
            global_step = epoch * len(train_loader) + i
            writer.add_scalar('Train/Loss', loss.item(), global_step)

        avg_train_loss = train_loss_epoch / len(train_loader)
        writer.add_scalar('Train/Epoch_Loss', avg_train_loss, epoch + 1)

        # Validation
        model.eval()
        val_loss = 0.0
        
        # Metrics Accumulators
        metrics_sum = np.zeros(6) # miou, iou_fg, iou_bg, precision, recall, f1
        num_val_batches = 0
        
        with torch.no_grad():
            loop_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for batch in loop_val:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)

                outputs = model(images)
                loss = criterion_bce(outputs, masks)
                val_loss += loss.item()
                
                # Apply sigmoid for metric calculation
                preds_prob = torch.sigmoid(outputs)
                
                # Calculate metrics for the batch
                # Note: calculate_metrics handles the batch flattening internally
                batch_metrics = calculate_metrics(preds_prob, masks)
                metrics_sum += np.array(batch_metrics)
                num_val_batches += 1

        avg_val_loss = val_loss / len(val_loader)
        
        # Average Metrics
        avg_metrics = metrics_sum / num_val_batches
        miou, iou_fg, iou_bg, precision, recall, f1 = avg_metrics
        
        # Update Scheduler
        scheduler.step()
        curr_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/LR', curr_lr, epoch + 1)
        
        # Logging
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        logger.info(f"  mIoU: {miou:.4f} | Fg IoU: {iou_fg:.4f} | Bg IoU: {iou_bg:.4f}")
        logger.info(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        
        # TensorBoard Validation Logging
        writer.add_scalar('Val/Loss', avg_val_loss, epoch + 1)
        writer.add_scalar('Val/mIoU', miou, epoch + 1)
        writer.add_scalar('Val/IoU_Fg', iou_fg, epoch + 1)
        writer.add_scalar('Val/IoU_Bg', iou_bg, epoch + 1)
        writer.add_scalar('Val/Precision', precision, epoch + 1)
        writer.add_scalar('Val/Recall', recall, epoch + 1)
        writer.add_scalar('Val/F1', f1, epoch + 1)

        # Save Best Model
        if miou > best_miou:
            best_miou = miou
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_score': best_miou,
            }, filename=os.path.join(log_dir, 'best_model.pth'))
            logger.info(f"New best model saved with mIoU: {best_miou:.4f}")

    writer.close()
    logger.info(f"Training Complete. Best mIoU: {best_miou:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MarsLS-Net')
    parser.add_argument('--data_dir', type=str, default='/raid/danielchen/Mars-LS-challenge/Dataset', help='Path to dataset')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size for MHSA downsampling')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    train(args)
