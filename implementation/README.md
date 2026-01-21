# MarsLS-Net Implementation

This folder contains the implementation of the **MarsLS-Net** architecture for Martian Landslide Segmentation, based on the WACV 2024 paper *Paheding, S. et al.*

## Files Structure
- **`dataset.py`**: Custom PyTorch Dataset loader for handling the 7-channel TIFF images and masks. Requires `tifffile`.
- **`model.py`**: The MarsLS-Net architecture implementation, featuring:
    - **ConvPE (Convolutional Progressive Expansion)** layer with a polynomial expansion formulation ($S_u = c_1 x + c_2 x^2$).
    - **PEN-Attention Block** incorporating parallel ConvPE branches and Multi-Head Self-Attention.
    - **MarsLSNet** main model class.
- **`train.py`**: Main training script with validation loop, BCE+Dice loss (placeholder logic included), and checkpoint saving.
- **`utils.py`**: Helper functions for calculating metrics (IoU) and managing checkpoints.
- **`verify.py`**: A standalone script to verify the model architecture, parameter count, and forward pass dimensionality.

## Dependency Requirements
The implementation depends on the following libraries (which seem to be available in your notebook environment):
- `torch`
- `torchvision`
- `numpy`
- `tifffile` (For loading `.tif` dataset images)
- `tqdm` (For progress bars)

## Usage

### 1. Verify Architecture
Run the verification script to confirm the model is constructed correctly and can process 7-channel inputs of size 128x128.
```bash
python verify.py
```

### 2. Train the Model
Run the training script. You can configure hyperparameters via command-line arguments.
```bash
python train.py --data_dir /raid/danielchen/Mars-LS-challenge/Dataset --epochs 50 --batch_size 16 --lr 1e-3
```

### 3. Progressive Expansion Note
The "Progressive Expansion" layer is implemented as a channel-wise learnable polynomial sum of order 2 ($x^1$ and $x^2$), consistent with the concepts described in the paper summary.

## Dataset
The code assumes the dataset is located at `/raid/danielchen/Mars-LS-challenge/Dataset` (or provided via `--data_dir`) and follows the structure:
- `train/images/*.tif`
- `train/masks/*.tif`
- `val/images/*.tif`
- `val/masks/*.tif`
