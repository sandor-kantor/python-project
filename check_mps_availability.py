import torch

# Check if MPS is available
if torch.backends.mps.is_available():
    print("MPS device is available")
    print(f"Number of MPS devices: {torch.mps.device_count()}")
else:
    print("MPS device is not available")
    if not torch.backends.mps.is_built():
        print("PyTorch was not built with MPS support")