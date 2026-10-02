"""
Summarize sulcus angles from per-case JSON files produced by the inference pipeline.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib
# Try to use non-interactive backend for batch processing
try:
    matplotlib.use("Agg")
except:
    pass
import matplotlib.pyplot as plt

def _resample_angles(angles: List[float], num_points: int) -> np.ndarray:
    """Resample angles to fixed number of points using linear interpolation."""
    values = np.array([a for a in angles if np.isfinite(a)], dtype=np.float64)
    if values.size == 0:
        return np.full(num_points, np.nan, dtype=np.float64)
    if values.size == 1:
        return np.full(num_points, values[0], dtype=np.float64)
    
    x_old = np.linspace(0.0, 1.0, values.size)
    x_new = np.linspace(0.0, 1.0, num_points)
    return np.interp(x_new, x_old, values)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create summary plot from inference JSON files"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing case subfolders with sulcus_angles_*.json"
    )
    parser.add_argument(
        "--output", type=str, default="sulcus_angle_summary.png",
        help="Output plot path"
    )
    parser.add_argument(
        "--num_points", type=int, default=100,
        help="Number of points for normalized resampling"
    )
    parser.add_argument(
        "--show_individual", action="store_true",
        help="Plot each individual case with low alpha"
    )
    parser.add_argument(
        "--individual_alpha", type=float, default=0.15,
        help="Alpha transparency for individual curves"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")

    # Find JSON files
    json_files = sorted(input_dir.rglob("sulcus_angles_*.json"))
    if not json_files:
        print(f"No sulcus_angles_*.json files found in {input_dir}")
        return

    print(f"Found {len(json_files)} JSON files")

    per_case_curves: List[np.ndarray] = []
    
    for json_path in json_files:
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            
            angles = data.get("angles", [])
            if not angles:
                continue
            
            resampled = _resample_angles(angles, args.num_points)
            if np.isfinite(resampled).any():
                per_case_curves.append(resampled)
                
        except Exception as e:
            print(f"Error processing {json_path}: {e}")

    if not per_case_curves:
        print("No valid angle curves found.")
        return

    print(f"Included {len(per_case_curves)} cases in summary.")

    # Convert to stack for statistics
    stacked = np.vstack(per_case_curves)
    mean_angle = np.nanmean(stacked, axis=0)
    std_angle = np.nanstd(stacked, axis=0)
    x_vals = np.linspace(0.0, 100.0, args.num_points)

    # Plotting
    plt.figure(figsize=(10, 6))
    
    if args.show_individual:
        for curve in per_case_curves:
            plt.plot(
                x_vals, curve,
                color="tab:blue",
                linewidth=0.8,
                alpha=args.individual_alpha
            )
    
    plt.plot(x_vals, mean_angle, color="tab:blue", linewidth=2.5, label="Mean Angle")
    plt.fill_between(
        x_vals,
        mean_angle - std_angle,
        mean_angle + std_angle,
        color="tab:blue",
        alpha=0.25,
        label="±1 Std Dev"
    )

    plt.xlabel("Normalized Slice Position (%)", fontsize=12)
    plt.ylabel("Sulcus Angle (degrees)", fontsize=12)
    plt.title(f"Inference Sulcus Angle Summary (n={len(per_case_curves)})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    
    # Add stats text
    angle_min = np.nanmin(stacked)
    angle_max = np.nanmax(stacked)
    stats_text = (
        f"Cases: {len(per_case_curves)}\n"
        f"Range: {angle_min:.1f}° - {angle_max:.1f}°"
    )
    plt.gcf().text(0.02, 0.02, stats_text, ha="left", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved summary plot to: {args.output}")

if __name__ == "__main__":
    main()
