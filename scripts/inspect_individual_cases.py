"""
Interactive inspection of individual sulcus angle curves.
Allows jumping through cases using arrow keys.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

def _resample_angles(angles: List[float], num_points: int, min_angle: Optional[float] = None) -> np.ndarray:
    """Resample angles to fixed number of points using linear interpolation."""
    vals = np.array(angles, dtype=np.float64)
    if min_angle is not None:
        vals[vals < min_angle] = np.nan
    
    valid_mask = np.isfinite(vals)
    valid_vals = vals[valid_mask]
    
    if valid_vals.size == 0:
        return np.full(num_points, np.nan, dtype=np.float64)
    if valid_vals.size == 1:
        return np.full(num_points, valid_vals[0], dtype=np.float64)
    
    x_old = np.linspace(0.0, 1.0, len(vals))[valid_mask]
    x_new = np.linspace(0.0, 1.0, num_points)
    
    return np.interp(x_new, x_old, valid_vals)

def load_dataset(
    directory: Path, 
    num_points: int, 
    min_angle: Optional[float] = None, 
    max_jump: Optional[float] = None
) -> List[Tuple[str, np.ndarray]]:
    """Loads and filters angle curves, returning (filename, curve) pairs."""
    json_files = sorted(directory.rglob("sulcus_angles_*.json"))
    dataset = []
    
    for json_path in json_files:
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            
            angles = data.get("angles", [])
            if not angles:
                continue
            
            if max_jump is not None and len(angles) > 1:
                jumps = np.abs(np.diff(np.array(angles, dtype=np.float64)))
                if np.any(jumps > max_jump):
                    continue
            
            resampled = _resample_angles(angles, num_points, min_angle)
            if np.isfinite(resampled).any():
                dataset.append((json_path.name, resampled))
                
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            
    return dataset

class CaseInspector:
    def __init__(self, dataset, num_points):
        self.dataset = dataset
        self.num_points = num_points
        self.index = 0
        self.x_vals = np.linspace(0.0, 100.0, num_points)
        
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Pre-calculate background
        self.all_curves = np.vstack([c for _, c in self.dataset])
        
        self.update_plot()
        print("\nInteractive Inspector Started:")
        print("  - Use Arrow Right / 'n' for next case")
        print("  - Use Arrow Left / 'p' for previous case")
        print("  - Close window to exit")

    def update_plot(self):
        self.ax.clear()
        name, current_curve = self.dataset[self.index]
        
        # Plot all in background
        for _, c in self.dataset:
            self.ax.plot(self.x_vals, c, color='gray', alpha=0.1, linewidth=0.5)
            
        # Highlight current
        self.ax.plot(self.x_vals, current_curve, color='magenta', linewidth=3, label=f"Current: {name}")
        
        self.ax.set_xlabel("Normalized Slice Position (%)")
        self.ax.set_ylabel("Sulcus Angle (degrees)")
        self.ax.set_title(f"Case {self.index + 1}/{len(self.dataset)}: {name}")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='best')
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key in ['right', 'n']:
            self.index = (self.index + 1) % len(self.dataset)
            self.update_plot()
        elif event.key in ['left', 'p']:
            self.index = (self.index - 1) % len(self.dataset)
            self.update_plot()

def main():
    parser = argparse.ArgumentParser(description="Interactively inspect individual sulcus curves")
    parser.add_argument("--input_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--num_points", type=int, default=100, help="Resampling points")
    parser.add_argument("--min_angle", type=float, default=None, help="Exclude values below this angle")
    parser.add_argument("--max_jump", type=float, default=None, help="Exclude cases with jumps > this value")
    args = parser.parse_args()

    directory = Path(args.input_dir)
    print(f"Loading dataset from {directory}...")
    dataset = load_dataset(directory, args.num_points, args.min_angle, args.max_jump)
    
    if not dataset:
        print("No valid cases found after filtering.")
        return
    
    print(f"Loaded {len(dataset)} cases.")
    
    inspector = CaseInspector(dataset, args.num_points)
    plt.show()

if __name__ == "__main__":
    main()
