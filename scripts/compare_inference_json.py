"""
Compare sulcus angles from two different sets of per-case JSON files.
Includes filtering for jumps and minimum angle values.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import matplotlib
# Try to use non-interactive backend for batch processing
try:
    matplotlib.use("Agg")
except:
    pass
import matplotlib.pyplot as plt

def _resample_angles(angles: List[float], num_points: int, min_angle: Optional[float] = None) -> np.ndarray:
    """Resample angles to fixed number of points using linear interpolation.
    Applies min_angle filter by converting values below threshold to NaN.
    """
    vals = np.array(angles, dtype=np.float64)
    
    # Filter by minimum angle
    if min_angle is not None:
        vals[vals < min_angle] = np.nan
    
    # Extract only finite values for interpolation
    valid_mask = np.isfinite(vals)
    valid_vals = vals[valid_mask]
    
    if valid_vals.size == 0:
        return np.full(num_points, np.nan, dtype=np.float64)
    if valid_vals.size == 1:
        return np.full(num_points, valid_vals[0], dtype=np.float64)
    
    # We still want to resample across the original "span", so we use the original indices
    # for the non-NaN points relative to the original sequence length.
    x_old = np.linspace(0.0, 1.0, len(vals))[valid_mask]
    x_new = np.linspace(0.0, 1.0, num_points)
    
    return np.interp(x_new, x_old, valid_vals)

def load_dataset(
    directory: Path, 
    num_points: int, 
    min_angle: Optional[float] = None, 
    max_jump: Optional[float] = None,
    whitelist: Optional[set] = None,
) -> Tuple[List[np.ndarray], int, int]:
    """Loads and filters angle curves from a directory of JSON files.
    Returns (curves, total_found, num_excluded_jump).
    """
    json_files = sorted(directory.rglob("sulcus_angles_*.json"))
    if whitelist is not None:
        json_files = [jf for jf in json_files if jf.parent.name in whitelist]
    curves: List[np.ndarray] = []
    excluded_jump = 0
    
    for json_path in json_files:
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            
            angles = data.get("angles", [])
            if not angles:
                continue
            
            # Check for jumps (before resampling/interpolation)
            if max_jump is not None and len(angles) > 1:
                jumps = np.abs(np.diff(np.array(angles, dtype=np.float64)))
                if np.any(jumps > max_jump):
                    excluded_jump += 1
                    continue
            
            resampled = _resample_angles(angles, num_points, min_angle)
            if np.isfinite(resampled).any():
                curves.append(resampled)
                
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            
    return curves, len(json_files), excluded_jump

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two inference JSON datasets")
    parser.add_argument("--input_dir_1", type=str, required=True, help="First dataset directory")
    parser.add_argument("--label_1", type=str, default="Dataset 1", help="Label for first dataset")
    parser.add_argument("--input_dir_2", type=str, required=True, help="Second dataset directory")
    parser.add_argument("--label_2", type=str, default="Dataset 2", help="Label for second dataset")
    parser.add_argument("--output", type=str, default="comparison_summary.png", help="Output plot path")
    parser.add_argument("--num_points", type=int, default=100, help="Resampling points")
    parser.add_argument("--min_angle", type=float, default=None, help="Exclude values below this angle")
    parser.add_argument("--max_jump", type=float, default=None, help="Exclude cases with jumps > this value")
    parser.add_argument("--show_individual", action="store_true", help="Plot individual curves for both datasets")
    parser.add_argument("--show_individual_1", action="store_true", help="Plot individual curves for first dataset")
    parser.add_argument("--show_individual_2", action="store_true", help="Plot individual curves for second dataset")
    parser.add_argument("--individual_alpha", type=float, default=0.1, help="Alpha for individual curves")
    parser.add_argument("--whitelist_1", type=str, default=None,
                        help="Text file with allowed directory names (one per line) for dataset 1")
    parser.add_argument("--whitelist_2", type=str, default=None,
                        help="Text file with allowed directory names (one per line) for dataset 2")
    parser.add_argument("--no_title", action="store_true", help="Omit plot title")
    parser.add_argument("--no_textbox", action="store_true", help="Omit bottom-left stats textbox")
    args = parser.parse_args()

    dir1 = Path(args.input_dir_1)
    dir2 = Path(args.input_dir_2)

    def _load_whitelist(path_str):
        if path_str is None:
            return None
        names = set()
        with open(path_str, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    names.add(line)
        return names

    wl1 = _load_whitelist(args.whitelist_1)
    wl2 = _load_whitelist(args.whitelist_2)

    print(f"Loading {args.label_1} from {dir1}...")
    curves1, total1, jump1 = load_dataset(dir1, args.num_points, args.min_angle, args.max_jump, wl1)
    print(f"  Found {total1}, Jump Excluded: {jump1}, Final: {len(curves1)}")

    print(f"Loading {args.label_2} from {dir2}...")
    curves2, total2, jump2 = load_dataset(dir2, args.num_points, args.min_angle, args.max_jump, wl2)
    print(f"  Found {total2}, Jump Excluded: {jump2}, Final: {len(curves2)}")

    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 15,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
    })

    plt.figure(figsize=(12, 7))
    x_vals = np.linspace(0.0, 100.0, args.num_points)

    def plot_data(curves, color, label, show_indiv):
        if not curves:
            return
        stacked = np.vstack(curves)
        mean_v = np.nanmean(stacked, axis=0)
        std_v = np.nanstd(stacked, axis=0)
        
        if show_indiv:
            for c in curves:
                plt.plot(x_vals, c, color=color, alpha=args.individual_alpha, linewidth=0.5)
        
        plt.plot(x_vals, mean_v, color=color, linewidth=2.5, label=f"{label} (n={len(curves)})")
        plt.fill_between(x_vals, mean_v - std_v, mean_v + std_v, color=color, alpha=0.15)

    show1 = args.show_individual or args.show_individual_1
    show2 = args.show_individual or args.show_individual_2

    plot_data(curves1, "tab:blue", args.label_1, show1)
    plot_data(curves2, "tab:red", args.label_2, show2)

    plt.xlabel("Normalized Slice Position [NSP] (%)", fontsize=15, fontweight='bold')
    plt.ylabel("Sulcus Angle (\u00b0)", fontsize=15, fontweight='bold')
    if not args.no_title:
        plt.title("Sulcus Angle Dataset Comparison", fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=13)

    if not args.no_textbox:
        # Stats Summary Text box
        stats_lines = [
            f"Filters: min_angle={args.min_angle}, max_jump={args.max_jump}",
            f"{args.label_1}: {len(curves1)}/{total1} cases",
            f"{args.label_2}: {len(curves2)}/{total2} cases"
        ]
        plt.gcf().text(0.02, 0.02, "\n".join(stats_lines), ha="left", va="bottom", fontsize=9, 
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='silver'))

    # Auto-switch to PDF if output ends with .png
    output_path = args.output
    if output_path.endswith(".png"):
        output_path = output_path.rsplit(".", 1)[0] + ".pdf"

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to: {output_path}")

if __name__ == "__main__":
    main()
