
import torch
import os

def inspect_file(path):
    print(f"Inspecting: {path}")
    if not os.path.exists(path):
        print(f"  Error: File does not exist")
        return
    
    is_link = os.path.islink(path)
    print(f"  Is Symlink: {is_link}")
    if is_link:
        print(f"  Link Target: {os.readlink(path)}")

    size = os.path.getsize(path)
    print(f"  Size: {size} bytes")
    
    # Try reading first few bytes
    try:
        with open(path, 'rb') as f:
            header = f.read(20)
            print(f"  Header (hex): {header.hex()}")
    except Exception as e:
        print(f"  Error reading header: {e}")

    try:
        # Try loading with torch
        # Use r+ to ensure we have read access and handles correctly on Windows if it's weird
        checkpoint = torch.load(path, map_location='cpu')
        print(f"  Type: PyTorch Checkpoint")
        print(f"  Keys: {list(checkpoint.keys())}")
        
        metrics = {}
        for k, v in checkpoint.items():
            if any(m in k.lower() for m in ['loss', 'metric', 'acc', 'epoch']):
                metrics[k] = v
        if metrics:
            print(f"  Metrics: {metrics}")
    except Exception as e:
        print(f"  Error loading with torch: {e}")
    print("-" * 40)

base_dir = r"c:\Users\michael\Projects\knee-landmark"
paths = [
    os.path.join(base_dir, "tmp", "checkpoints", "axial_landmarks", "best_model.pt"),
    os.path.join(base_dir, "tmp", "checkpoints", "axial_landmarks", "checkpoint_keypoint_model_9.pt"),
    os.path.join(base_dir, "tmp", "checkpoints", "sagittal_bounds", "best_model.pt"),
    os.path.join(base_dir, "tmp", "checkpoints", "sagittal_bounds", "checkpoint_keypoint_model_34.pt")
]

for p in paths:
    inspect_file(p)
