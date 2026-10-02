"""
Compute sulcus angles using Leo's probability spline fitting (SimpleProbabilitySplineFitter).
Maximum control points are set to 10 by default.

Now supports PHYSICAL ANGLE COMPUTATION using voxel spacing from original NIfTI files.

Example:

python scripts/compute_sulcus_angles_spline_leo.py --input_dir ./inference_output_all --output ./sulcus_angle_leo_summary.png --max_control 10 --show_individual --nifti_dir .\inference_data_all


  python scripts/compute_sulcus_angles_spline_leo.py ^
    --input_dir ./inference_output_all ^
    --nifti_dir ./inference_data_all ^
    --output ./sulcus_angle_leo_summary.png
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

import numpy as np
import matplotlib
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

# Try to use non-interactive backend for batch processing
try:
    matplotlib.use("Agg")
except:
    pass
import matplotlib.pyplot as plt


class SimpleProbabilitySplineFitter:
    def __init__(self, probability_slices):
        """
        Parameters:
        -----------
        probability_slices : np.ndarray
            Shape (n_slices, height, width)
        """
        self.slices = probability_slices
        self.n_slices = probability_slices.shape[0]
        self.height = probability_slices.shape[1]
        self.width = probability_slices.shape[2]

    def get_probability(self, z_idx, x, y):
        """Bilinear interpolation of probability at (x, y) in slice z_idx."""
        x = np.clip(x, 0, self.width - 1.001)
        y = np.clip(y, 0, self.height - 1.001)

        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, self.width - 1), min(y0 + 1, self.height - 1)

        wx, wy = x - x0, y - y0

        return ((1-wx)*(1-wy)*self.slices[z_idx, y0, x0] +
                wx*(1-wy)*self.slices[z_idx, y0, x1] +
                (1-wx)*wy*self.slices[z_idx, y1, x0] +
                wx*wy*self.slices[z_idx, y1, x1])

    def spline_from_params(self, params, n_control):
        """Build spline from flattened control point parameters."""
        xy = params.reshape(n_control, 2)
        z = np.linspace(0, self.n_slices - 1, n_control)

        k = min(3, n_control - 1)
        tck, _ = splprep([xy[:, 0], xy[:, 1], z], s=0, k=k)
        return tck

    def evaluate_at_slices(self, tck):
        """Get (x, y) coordinates where spline intersects each slice."""
        # Sample spline densely
        u = np.linspace(0, 1, 1000)
        x, y, z = splev(u, tck)

        # Find closest point to each slice
        points = np.zeros((self.n_slices, 2))
        for i in range(self.n_slices):
            idx = np.argmin(np.abs(z - i))
            points[i] = [x[idx], y[idx]]

        return points

    def objective(self, params, n_control):
        """Negative sum of log probabilities (to minimize)."""
        try:
            tck = self.spline_from_params(params, n_control)
            points = self.evaluate_at_slices(tck)

            log_prob_sum = 0
            for i in range(self.n_slices):
                prob = self.get_probability(i, points[i, 0], points[i, 1])
                log_prob_sum += np.log(prob + 1e-10)

            return -log_prob_sum
        except:
            return 1e10

    def fit(self, min_control=3, max_control=None, tol=0.01, verbose=False):
        """
        Fit spline with adaptive number of control points.
        """
        if max_control is None:
            max_control = self.n_slices
        
        max_control = max(int(max_control), min_control)

        # Initial guess: max probability in each slice
        init_points = np.zeros((self.n_slices, 2))
        for i in range(self.n_slices):
            flat_idx = np.argmax(self.slices[i])
            y_max, x_max = np.unravel_index(flat_idx, self.slices[i].shape)
            init_points[i] = [x_max, y_max] # [col, row], leo style

        history = []
        best_result = None
        prev_score = -np.inf

        # Optimization loop
        for n in range(min_control, max_control + 1):
            z_all = np.arange(self.n_slices)
            z_control = np.linspace(0, self.n_slices - 1, n)
            x_init = np.interp(z_control, z_all, init_points[:, 0])
            y_init = np.interp(z_control, z_all, init_points[:, 1])
            x0 = np.column_stack([x_init, y_init]).flatten()

            bounds = [(0, self.width-1), (0, self.height-1)] * n

            result = minimize(
                fun=lambda p: self.objective(p, n),
                x0=x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-6}
            )

            score = -result.fun
            improvement = score - prev_score

            tck = self.spline_from_params(result.x, n)
            control_points = result.x.reshape(n, 2)

            history.append({
                'n_control': n,
                'score': score,
                'improvement': improvement,
                'tck': tck,
                'control_points': control_points
            })

            if best_result is None or score > best_result['score']:
                best_result = history[-1]

            if n > min_control and improvement < tol:
                break

            prev_score = score
            init_points = self.evaluate_at_slices(tck)

        return {
            'best_tck': best_result['tck'],
            'best_score': best_result['score'],
            'best_n': best_result['n_control'],
            'best_points': self.evaluate_at_slices(best_result['tck'])
        }


def _compute_angle_deg(
    p1: np.ndarray, 
    p2: np.ndarray, 
    p3: np.ndarray,
    pixel_spacing: Optional[Tuple[float, float]] = None
) -> float:
    """Compute angle at p2 between vectors p1-p2 and p3-p2.
    
    If pixel_spacing is provided, coordinates are scaled to physical space.
    pixel_spacing = (row_spacing, col_spacing) in mm.
    """
    # Scale to physical coordinates if spacing provided
    if pixel_spacing is not None:
        # Points are in [col, row] format from Leo fitter
        # Scale: col * col_spacing, row * row_spacing
        row_sp, col_sp = pixel_spacing
        p1 = np.array([p1[0] * col_sp, p1[1] * row_sp])
        p2 = np.array([p2[0] * col_sp, p2[1] * row_sp])
        p3 = np.array([p3[0] * col_sp, p3[1] * row_sp])
    
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
    pixel_spacing: Optional[Tuple[float, float]] = None,
) -> List[float]:
    """Compute sulcus angle for each slice from fitted landmark points.
    
    Args:
        pixel_spacing: (row_spacing, col_spacing) in mm for physical angles
    """
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
            angles.append(_compute_angle_deg(p0[i], p1[i], p2[i], pixel_spacing))
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


def find_matching_nifti(case_name: str, nifti_dir: Path) -> Optional[Path]:
    """Find the NIfTI file matching the case name."""
    # Case name is the folder name from heatmap path, e.g. "FB_100116____FB,3230126791_study_20b85c56"
    # NIfTI files have pattern like "FB_100116____FB,3230126791_study_20b85c56_res_256_epoch_50_contr_0.nii.gz"
    
    # Try direct match first
    for nifti_path in nifti_dir.glob(f"{case_name}*.nii.gz"):
        return nifti_path
    
    for nifti_path in nifti_dir.glob(f"{case_name}*.nii"):
        return nifti_path
    
    # Try partial match (case name might be truncated)
    for nifti_path in nifti_dir.iterdir():
        if nifti_path.suffix in ['.gz', '.nii'] and case_name in nifti_path.name:
            return nifti_path
    
    return None


def get_pixel_spacing(nifti_path: Path) -> Optional[Tuple[float, float]]:
    """Get (row_spacing, col_spacing) from NIfTI header.
    
    Returns pixel spacing in mm for the axial plane (assuming RAS orientation).
    """
    if not HAS_NIBABEL:
        return None
    
    try:
        img = nib.load(str(nifti_path))
        header = img.header
        
        # Get voxel dimensions (pixdim)
        # pixdim[1:4] = voxel sizes in x, y, z
        pixdim = header.get_zooms()
        
        # For axial slices, we care about x and y dimensions
        # pixdim[0] = x spacing (columns), pixdim[1] = y spacing (rows)
        col_spacing = float(pixdim[0])
        row_spacing = float(pixdim[1])
        
        return (row_spacing, col_spacing)
    except Exception as e:
        print(f"Warning: Could not read spacing from {nifti_path}: {e}")
        return None


def process_case(
    heatmap_path: Path,
    max_control: int,
    channels: List[int],
    nifti_dir: Optional[Path] = None,
) -> Tuple[List[float], str, Optional[Tuple[float, float]]]:
    """Process a single case and return angles.
    
    Returns:
        (angles, case_name, pixel_spacing)
    """
    case_name = heatmap_path.parent.name
    
    # Get pixel spacing if nifti_dir provided
    pixel_spacing = None
    if nifti_dir is not None:
        nifti_path = find_matching_nifti(case_name, nifti_dir)
        if nifti_path is not None:
            pixel_spacing = get_pixel_spacing(nifti_path)
    
    try:
        data = np.load(heatmap_path, allow_pickle=True)
        heatmaps = data["heatmaps"]  # (num_slices, num_landmarks, H, W)
    except Exception as e:
        print(f"Error loading {heatmap_path}: {e}")
        return [], case_name, pixel_spacing
    
    if heatmaps.size == 0:
        return [], case_name, pixel_spacing
    
    num_landmarks = heatmaps.shape[1]
    
    # Validate channels
    valid_channels = [ch for ch in channels if 0 <= ch < num_landmarks]
    if len(valid_channels) < 3:
        return [], case_name, pixel_spacing
    
    # Fit splines for each channel
    per_landmark_points = {}
    for ch in valid_channels:
        vol = heatmaps[:, ch, :, :]
        fitter = SimpleProbabilitySplineFitter(vol)
        try:
            res = fitter.fit(min_control=3, max_control=max_control, tol=0.01, verbose=False)
            # Leo fitter returns [col, row] format
            per_landmark_points[str(ch)] = res['best_points']
        except Exception as e:
            print(f"Fitting failed for ch{ch}: {e}")
            return [], case_name, pixel_spacing
    
    # Compute angles with physical spacing
    angles = compute_angles_from_fitted_points(per_landmark_points, valid_channels, pixel_spacing)
    
    return angles, case_name, pixel_spacing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute sulcus angles using Leo's probability spline fitting"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Root folder containing case subfolders with axial_heatmaps.npz"
    )
    parser.add_argument(
        "--output", type=str, default="sulcus_angle_leo_summary.png",
        help="Output plot path"
    )
    parser.add_argument(
        "--max_control", type=int, default=10,
        help="Max control points for spline fitting (default: 10)"
    )
    parser.add_argument(
        "--channels", type=str, default="0,1,2",
        help="Comma-separated channel indices"
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
        help="Alpha for individual curves"
    )
    parser.add_argument(
        "--angle_min", type=float, default=None,
        help="Minimum angle filter"
    )
    parser.add_argument(
        "--angle_max", type=float, default=None,
        help="Maximum angle filter"
    )
    parser.add_argument(
        "--max_jump", type=float, default=None,
        help="Exclude cases with angle jumps > threshold"
    )
    parser.add_argument(
        "--save_json", type=str, default=None,
        help="Save per-case angles to JSON file"
    )
    parser.add_argument(
        "--nifti_dir", type=str, default=None,
        help="Directory containing original NIfTI files for physical spacing"
    )
    args = parser.parse_args()
    
    # Check nibabel availability
    nifti_dir = None
    if args.nifti_dir:
        if not HAS_NIBABEL:
            print("WARNING: nibabel not installed. Install with 'pip install nibabel' for physical angles.")
            print("         Falling back to pixel-space angles.")
        else:
            nifti_dir = Path(args.nifti_dir)
            if not nifti_dir.is_dir():
                print(f"WARNING: NIfTI directory not found: {nifti_dir}")
                nifti_dir = None
            else:
                print(f"Using physical spacing from: {nifti_dir}")

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")

    channels = [int(c.strip()) for c in args.channels.split(",") if c.strip()]
    if len(channels) < 3:
        raise ValueError("Need at least 3 channels for angle computation")

    heatmap_files = sorted(input_dir.rglob("axial_heatmaps.npz"))
    if not heatmap_files:
        raise ValueError(f"No axial_heatmaps.npz files found in {input_dir}")

    print(f"Found {len(heatmap_files)} heatmap files")
    print(f"Max control points: {args.max_control}")
    print(f"Channels: {channels}")
    print(f"Physical angles: {'YES' if nifti_dir else 'NO (pixel space)'}")

    per_case: List[np.ndarray] = []
    case_names: List[str] = []
    all_angles_raw: List[List[float]] = []
    
    total_cases = 0
    excluded_cases_jump = 0
    total_angles_raw = 0
    excluded_angles_range = 0
    included_angles: List[float] = []

    cases_with_spacing = 0
    for heatmap_path in heatmap_files:
        total_cases += 1
        print(f"Processing {heatmap_path.parent.name}...", end="\r")
        angles, case_name, pixel_spacing = process_case(heatmap_path, args.max_control, channels, nifti_dir)
        
        if pixel_spacing is not None:
            cases_with_spacing += 1
        
        if not angles:
            print(f"  Skipped (no valid angles): {case_name}")
            continue
        
        # Filter NaN
        angles = [a for a in angles if np.isfinite(a)]
        total_angles_raw += len(angles)
        all_angles_raw.append(angles)
        
        if not angles:
            continue

        # Filters
        if args.angle_min is not None and args.angle_max is not None:
            before = len(angles)
            angles = [a for a in angles if args.angle_min <= a <= args.angle_max]
            excluded_angles_range += before - len(angles)
        
        if not angles:
            continue

        if args.max_jump is not None and len(angles) >= 2:
            jumps = np.abs(np.diff(np.array(angles, dtype=np.float64)))
            if np.any(jumps > float(args.max_jump)):
                excluded_cases_jump += 1
                print(f"  Exclude (jump): {case_name}")
                continue
        
        resampled = _resample_angles(angles, args.num_points)
        if np.isfinite(resampled).any():
            per_case.append(resampled)
            case_names.append(case_name)
            included_angles.extend(angles)

    print(f"\nIncluded cases: {len(per_case)}/{total_cases}")
    if nifti_dir:
        print(f"Cases with physical spacing: {cases_with_spacing}/{total_cases}")
    if not per_case:
        print("No valid cases found.")
        return

    # Statistics
    stacked = np.vstack(per_case)
    mean_angle = np.nanmean(stacked, axis=0)
    std_angle = np.nanstd(stacked, axis=0)
    x_vals = np.linspace(0.0, 100.0, args.num_points)

    # Save JSON
    if args.save_json:
        import json
        json_data = {
            "max_control": args.max_control,
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
    angle_type = "Physical" if nifti_dir else "Pixel-space"
    plt.title(f"Sulcus Angle Summary (Leo Fit, {angle_type}, max_ctrl={args.max_control})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    
    included_min = float(np.min(included_angles)) if included_angles else float("nan")
    included_max = float(np.max(included_angles)) if included_angles else float("nan")
    
    stats_text = (
        f"Cases: {len(per_case)}/{total_cases}\n"
        f"Angle range: {included_min:.1f}–{included_max:.1f} deg\n"
        f"Max control: {args.max_control}"
    )
    plt.gcf().text(0.02, 0.02, stats_text, ha="left", va="bottom", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved summary plot: {args.output}")


if __name__ == "__main__":
    main()
