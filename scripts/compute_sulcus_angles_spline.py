"""
Compute sulcus angles from heatmaps using spline fitting for all patients.
Generates a summary plot showing mean ± std across all cases.

Example:
  python scripts/compute_sulcus_angles_spline.py ^
    --input_dir ./inference_output_heatmap ^
    --output ./sulcus_angle_spline_summary.png ^
    --spline_smooth 35
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d


def _argmax_points_numpy(heatmaps: np.ndarray) -> Dict[str, np.ndarray]:
    """Get argmax positions for each channel."""
    num_slices, num_landmarks, _, w = heatmaps.shape
    per_landmark = {}
    for ch in range(num_landmarks):
        flat = heatmaps[:, ch].reshape(num_slices, -1)
        max_idx = flat.argmax(axis=1)
        init_row = (max_idx // w).astype(np.float64)
        init_col = (max_idx % w).astype(np.float64)
        per_landmark[str(ch)] = np.stack([init_row, init_col], axis=1)
    return per_landmark


def fit_spline(points: np.ndarray, smooth_factor: float) -> np.ndarray:
    """Fit a smooth cubic spline through points."""
    n = len(points)
    t = np.arange(n)
    
    try:
        spline_row = UnivariateSpline(t, points[:, 0], s=smooth_factor * n, k=min(3, n-1))
        spline_col = UnivariateSpline(t, points[:, 1], s=smooth_factor * n, k=min(3, n-1))
        
        smooth_row = spline_row(t)
        smooth_col = spline_col(t)
        return np.stack([smooth_row, smooth_col], axis=1)
    except Exception:
        # Fallback to Gaussian smoothing
        sigma = max(1, smooth_factor * 2)
        smooth_row = gaussian_filter1d(points[:, 0], sigma=sigma)
        smooth_col = gaussian_filter1d(points[:, 1], sigma=sigma)
        return np.stack([smooth_row, smooth_col], axis=1)


def _compute_angle_deg(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute angle at p2 between vectors p1-p2 and p3-p2."""
    v1 = p1 - p2
    v2 = p3 - p2
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return float("nan")
    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def compute_angles_from_fitted_points(
    per_landmark_points: Dict[str, np.ndarray],
    channels: List[int],
) -> List[float]:
    """Compute sulcus angle for each slice from fitted landmark points."""
    if len(channels) < 3:
        return []
    
    ch0, ch1, ch2 = channels[0], channels[1], channels[2]
    p0 = per_landmark_points[str(ch0)]
    p1 = per_landmark_points[str(ch1)]
    p2 = per_landmark_points[str(ch2)]
    
    num_slices = min(len(p0), len(p1), len(p2))
    angles = []
    for i in range(num_slices):
        if not (np.isfinite(p0[i]).all() and np.isfinite(p1[i]).all() and np.isfinite(p2[i]).all()):
            angles.append(float("nan"))
        else:
            angles.append(_compute_angle_deg(p0[i], p1[i], p2[i]))
    return angles


def _resample_angles(angles: List[float], num_points: int) -> np.ndarray:
    """Resample angles to fixed number of points."""
    values = np.array([a for a in angles if np.isfinite(a)], dtype=np.float64)
    if values.size == 0:
        return np.full(num_points, np.nan, dtype=np.float64)
    if values.size == 1:
        return np.full(num_points, values[0], dtype=np.float64)
    x_old = np.linspace(0.0, 1.0, values.size)
    x_new = np.linspace(0.0, 1.0, num_points)
    return np.interp(x_new, x_old, values)


def process_case(
    heatmap_path: Path,
    spline_smooth: float,
    channels: List[int],
) -> Tuple[List[float], str]:
    """Process a single case and return angles."""
    data = np.load(heatmap_path, allow_pickle=True)
    heatmaps = data["heatmaps"]  # (num_slices, num_landmarks, H, W)
    
    if heatmaps.size == 0:
        return [], heatmap_path.parent.name
    
    num_landmarks = heatmaps.shape[1]
    
    # Validate channels
    valid_channels = [ch for ch in channels if 0 <= ch < num_landmarks]
    if len(valid_channels) < 3:
        return [], heatmap_path.parent.name
    
    # Get argmax points
    per_landmark_argmax = _argmax_points_numpy(heatmaps)
    
    # Fit splines for each channel
    per_landmark_points = {}
    for ch in valid_channels:
        key = str(ch)
        argmax_pts = per_landmark_argmax[key]
        smooth_pts = fit_spline(argmax_pts, spline_smooth)
        per_landmark_points[key] = smooth_pts
    
    # Compute angles
    angles = compute_angles_from_fitted_points(per_landmark_points, valid_channels)
    
    case_name = heatmap_path.parent.name
    return angles, case_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute sulcus angles from heatmaps using spline fitting"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Root folder containing case subfolders with axial_heatmaps.npz"
    )
    parser.add_argument(
        "--output", type=str, default="sulcus_angle_spline_summary.png",
        help="Output plot path"
    )
    parser.add_argument(
        "--spline_smooth", type=float, default=35.0,
        help="Spline smoothing factor (higher = smoother)"
    )
    parser.add_argument(
        "--channels", type=str, default="0,1,2",
        help="Comma-separated channel indices for angle computation"
    )
    parser.add_argument(
        "--num_points", type=int, default=100,
        help="Number of points for resampling"
    )
    parser.add_argument(
        "--show_individual", action="store_true",
        help="Plot each case curve with low alpha"
    )
    parser.add_argument(
        "--individual_alpha", type=float, default=0.15,
        help="Alpha for individual curves when --show_individual is set"
    )
    parser.add_argument(
        "--angle_min", type=float, default=None,
        help="Minimum angle to include (degrees)"
    )
    parser.add_argument(
        "--angle_max", type=float, default=None,
        help="Maximum angle to include (degrees)"
    )
    parser.add_argument(
        "--max_jump", type=float, default=None,
        help="Exclude cases with angle jumps larger than this threshold"
    )
    parser.add_argument(
        "--save_json", type=str, default=None,
        help="Save per-case angles to JSON file"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")

    # Parse channels
    channels = [int(c.strip()) for c in args.channels.split(",") if c.strip()]
    if len(channels) < 3:
        raise ValueError("Need at least 3 channels for angle computation")

    # Find all heatmap files
    heatmap_files = sorted(input_dir.rglob("axial_heatmaps.npz"))
    if not heatmap_files:
        raise ValueError(f"No axial_heatmaps.npz files found in {input_dir}")

    print(f"Found {len(heatmap_files)} heatmap files")
    print(f"Spline smoothing factor: {args.spline_smooth}")
    print(f"Channels: {channels}")

    # Process all cases
    per_case: List[np.ndarray] = []
    case_names: List[str] = []
    all_angles_raw: List[List[float]] = []
    
    total_cases = 0
    excluded_cases_jump = 0
    total_angles_raw = 0
    excluded_angles_range = 0
    included_angles: List[float] = []

    for heatmap_path in heatmap_files:
        total_cases += 1
        angles, case_name = process_case(heatmap_path, args.spline_smooth, channels)
        
        if not angles:
            print(f"  Skipped (no valid angles): {case_name}")
            continue
        
        # Filter NaN
        angles = [a for a in angles if np.isfinite(a)]
        total_angles_raw += len(angles)
        all_angles_raw.append(angles)
        
        # Apply angle range filter
        if args.angle_min is not None and args.angle_max is not None:
            before = len(angles)
            angles = [a for a in angles if args.angle_min <= a <= args.angle_max]
            excluded_angles_range += before - len(angles)
        
        # Apply max jump filter
        if args.max_jump is not None and len(angles) >= 2:
            jumps = np.abs(np.diff(np.array(angles, dtype=np.float64)))
            if np.any(jumps > float(args.max_jump)):
                excluded_cases_jump += 1
                print(f"  Excluded (jump > {args.max_jump}): {case_name}")
                continue
        
        # Resample
        resampled = _resample_angles(angles, args.num_points)
        if np.isfinite(resampled).any():
            per_case.append(resampled)
            case_names.append(case_name)
            included_angles.extend(angles)
            print(f"  Processed: {case_name} ({len(angles)} angles)")

    if not per_case:
        raise RuntimeError(f"No valid angles found in {input_dir}")

    print(f"\nIncluded cases: {len(per_case)}/{total_cases}")

    # Compute statistics
    stacked = np.vstack(per_case)
    mean_angle = np.nanmean(stacked, axis=0)
    std_angle = np.nanstd(stacked, axis=0)
    x_vals = np.linspace(0.0, 100.0, args.num_points)

    # Save to JSON if requested
    if args.save_json:
        import json
        json_data = {
            "spline_smooth": args.spline_smooth,
            "channels": channels,
            "num_points": args.num_points,
            "cases": {
                name: resampled.tolist() 
                for name, resampled in zip(case_names, per_case)
            },
            "mean": mean_angle.tolist(),
            "std": std_angle.tolist(),
            "x_vals": x_vals.tolist(),
        }
        with open(args.save_json, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"Saved angles to: {args.save_json}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("matplotlib is required for plotting")

    plt.figure(figsize=(10, 6))
    
    if args.show_individual:
        for curve in per_case:
            plt.plot(
                x_vals, curve,
                color="tab:blue",
                linewidth=1.0,
                alpha=float(args.individual_alpha),
            )
    
    plt.plot(x_vals, mean_angle, color="tab:blue", linewidth=2.0, label="Mean angle")
    plt.fill_between(
        x_vals,
        mean_angle - std_angle,
        mean_angle + std_angle,
        color="tab:blue",
        alpha=0.25,
        label="±1 std",
    )
    
    plt.xlabel("Normalized Slice Position (%)", fontsize=12)
    plt.ylabel("Sulcus Angle (degrees)", fontsize=12)
    plt.title(f"Sulcus Angle Summary (Spline smooth={args.spline_smooth})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    
    # Statistics text
    included_min = float(np.min(included_angles)) if included_angles else float("nan")
    included_max = float(np.max(included_angles)) if included_angles else float("nan")
    included_cases = len(per_case)
    excluded_cases_total = total_cases - included_cases
    angles_included = len(included_angles)
    angles_excluded = total_angles_raw - angles_included
    
    range_text = "none"
    if args.angle_min is not None and args.angle_max is not None:
        range_text = f"{args.angle_min:.1f}–{args.angle_max:.1f}"
    jump_text = "none" if args.max_jump is None else f">{args.max_jump:.1f}"
    
    stats_text = (
        f"Cases: {included_cases}/{total_cases} included, {excluded_cases_total} excluded\n"
        f"Angles: {angles_included}/{total_angles_raw} included, {angles_excluded} excluded\n"
        f"Range filter: {range_text} deg, jump filter: {jump_text} deg\n"
        f"Angle range: {included_min:.1f}–{included_max:.1f} deg\n"
        f"Spline smoothing: {args.spline_smooth}"
    )
    plt.gcf().text(0.02, 0.02, stats_text, ha="left", va="bottom", fontsize=9)
    
    #plt.ylim(0, 180)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved summary plot: {args.output}")
    print(f"Cases: {len(per_case)}, points per case: {args.num_points}")
    print(f"Mean angle range: {np.nanmin(mean_angle):.1f}° - {np.nanmax(mean_angle):.1f}°")


if __name__ == "__main__":
    main()
