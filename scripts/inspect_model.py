"""Quick script to inspect a PyTorch model checkpoint."""
import sys
import os
import torch
import pickle

def inspect_model(path: str):
    # Resolve to absolute path to avoid Windows issues
    path = os.path.abspath(path)
    print(f"Loading: {path}")
    print(f"File size: {os.path.getsize(path)} bytes")
    
    try:
        # Try standard torch.load first
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e1:
        print(f"torch.load failed: {e1}")
        try:
            # Try opening file explicitly
            with open(path, 'rb') as f:
                checkpoint = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e2:
            print(f"torch.load with file handle failed: {e2}")
            try:
                # Try raw pickle
                with open(path, 'rb') as f:
                    checkpoint = pickle.load(f)
            except Exception as e3:
                print(f"pickle.load failed: {e3}")
                return

    
    if isinstance(checkpoint, dict):
        print(f"\nCheckpoint keys: {list(checkpoint.keys())}")
        
        # Check for common patterns
        if "model" in checkpoint:
            sd = checkpoint["model"]
        elif "state_dict" in checkpoint:
            sd = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            sd = checkpoint["model_state_dict"]
        else:
            # Assume it's a raw state_dict
            sd = checkpoint
        
        if isinstance(sd, dict):
            print(f"\nModel layers ({len(sd)} parameters):")
            for k, v in sd.items():
                if hasattr(v, "shape"):
                    print(f"  {k}: {tuple(v.shape)}")
                else:
                    print(f"  {k}: {type(v)}")
    else:
        # It's a full model object
        print(f"\nModel type: {type(checkpoint)}")
        print(checkpoint)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_model.py <model_path>")
        sys.exit(1)
    inspect_model(sys.argv[1])
