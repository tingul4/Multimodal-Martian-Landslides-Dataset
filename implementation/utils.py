import torch

def compute_iou(preds, labels, smooth=1e-6):
    """
    Computes Intersection over Union (IoU).
    Args:
        preds: (B, 1, H, W) Tensor of predicted logits or probabilities.
        labels: (B, 1, H, W) Tensor of ground truth labels (0 or 1).
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        iou: Scalar tensor.
    """
    # Sigmoid to get probabilities if logits (assumed logits usually in training loop, but let's assume thresholded here for metric)
    # Actually, metric calculation usually takes binary predictions.
    
    preds = (torch.sigmoid(preds) > 0.5).float()
    
    intersection = (preds * labels).sum()
    union = preds.sum() + labels.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou

def calculate_metrics(pred_mask, gt_mask):
    """
    Calculates detailed metrics: mIoU, Fg IoU, Bg IoU, Precision, Recall, F1.
    Args:
        pred_mask: (B, 1, H, W) or (B, H, W) Logits or Probabilities.
        gt_mask: (B, 1, H, W) or (B, H, W) Ground Truth (0/1).
    """
    # Ensure inputs are binary and flat
    pred = (pred_mask > 0.5).long().view(-1)
    gt = (gt_mask > 0.5).long().view(-1)
    
    TP = (pred * gt).sum().item()
    FP = (pred * (1-gt)).sum().item()
    FN = ((1-pred) * gt).sum().item()
    TN = ((1-pred) * (1-gt)).sum().item()
    
    # Foreground IoU
    union_fg = TP + FP + FN
    iou_fg = TP / union_fg if union_fg > 0 else 1.0
    
    # Background IoU
    union_bg = TN + FN + FP
    iou_bg = TN / union_bg if union_bg > 0 else 1.0
    
    # Mean IoU
    miou = (iou_fg + iou_bg) / 2
    
    # Precision, Recall, F1 (Foreground)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return miou, iou_fg, iou_bg, precision, recall, f1

def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(filename, model, optimizer=None):
    if not torch.cuda.is_available():
        checkpoint = torch.load(filename, map_location='cpu')
    else:
        checkpoint = torch.load(filename)
    
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    print(f"Checkpoint loaded from {filename}")
    return checkpoint.get('epoch', 0), checkpoint.get('best_score', 0.0)
