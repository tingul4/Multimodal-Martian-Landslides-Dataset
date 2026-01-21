import torch
import warnings
from model import MarsLSNet

# Suppress warnings
warnings.filterwarnings("ignore")

def verify_model():
    print("Verifying MarsLS-Net Architecture...")
    
    # 1. Instantiate Model
    try:
        model = MarsLSNet(in_channels=7, num_classes=1)
        print("Model instantiated successfully.")
    except Exception as e:
        print(f"Model instantiation failed: {e}")
        return

    # 2. Check Parameter Count
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {params}")
    
    # 3. Forward Pass Check
    print("Checking Forward Pass with random tensor (1, 7, 128, 128)...")
    input_tensor = torch.randn(1, 7, 128, 128)
    try:
        output = model(input_tensor)
        print(f"Output shape: {output.shape}")
        
        expected_shape = (1, 1, 128, 128)
        if output.shape == expected_shape:
            print("[\u2713] Output shape matches expected.")
        else:
            print(f"[X] Shape Mismatch! Expected {expected_shape}, got {output.shape}")
            
    except Exception as e:
        print(f"[X] Forward pass failed: {e}")

    # 4. Check PE coefficients
    print("Checking Progressive Expansion Coefficients initialization...")
    # Access a PE layer
    try:
        pe_layer = model.block1.branch1.pe
        print(f"PE Coefficients shape: {pe_layer.coeffs.shape} (Expected: (2, 24, 1, 1))")
        # 48 filters / 2 branches = 24
        if pe_layer.coeffs.shape == (2, 24, 1, 1):
             print("[\u2713] PE Coefficients shape correct.")
        else:
             print(f"[X] PE Coefficients shape incorrect.")
    except Exception as e:
        print(f"Could not access PE layer for verification: {e}")

if __name__ == "__main__":
    verify_model()
