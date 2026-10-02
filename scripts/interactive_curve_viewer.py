"""
Interactive 3D viewer for heatmaps with spline-fitted curves.

Combines:
- 3D stacked heatmap visualization
- Smooth spline-fitted landmark curves
- Interactive smoothness slider
- Sulcus angle plot

Example:
  python scripts/interactive_curve_viewer.py ^
    --heatmaps ./inference_output_heatmap/.../axial_heatmaps.npz ^
    --channels 0,1,2 ^
    --spline_smooth 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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


def _compute_angles(
    points_by_channel: Dict[str, np.ndarray],
    angle_channels: Tuple[int, int, int],
) -> List[float]:
    """Compute sulcus angle for each slice."""
    ch1, ch2, ch3 = angle_channels
    p1 = points_by_channel[str(ch1)]
    p2 = points_by_channel[str(ch2)]
    p3 = points_by_channel[str(ch3)]
    num_slices = min(len(p1), len(p2), len(p3))
    angles = []
    for idx in range(num_slices):
        if not (np.isfinite(p1[idx]).all() and np.isfinite(p2[idx]).all() and np.isfinite(p3[idx]).all()):
            angles.append(float("nan"))
        else:
            angles.append(_compute_angle_deg(p1[idx], p2[idx], p3[idx]))
    return angles


def _ensure_interactive_backend() -> None:
    """Ensure matplotlib uses an interactive backend."""
    import matplotlib
    current = matplotlib.get_backend()
    try:
        from matplotlib.backends import backend_registry
        interactive = backend_registry.list_builtin(backend_registry.BackendFilter.INTERACTIVE)
    except Exception:
        interactive = getattr(matplotlib.rcsetup, "interactive_bk", [])
    if current in interactive:
        return
    for candidate in ("TkAgg", "QtAgg", "Qt5Agg", "WXAgg"):
        try:
            matplotlib.use(candidate, force=True)
            return
        except Exception:
            continue


def fit_spline(points: np.ndarray, smooth_factor: float) -> np.ndarray:
    """
    Fit a smooth cubic spline through points.
    
    Args:
        points: (N, 2) array of [row, col] positions
        smooth_factor: Higher = smoother curves (0 = interpolate exactly)
    
    Returns:
        (N, 2) array of smoothed positions
    """
    n = len(points)
    t = np.arange(n)
    
    try:
        # s parameter controls smoothness: s=0 interpolates, s>0 smooths
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive 3D heatmap + spline curve viewer")
    parser.add_argument("--heatmaps", type=str, required=True, help="Path to axial_heatmaps.npz")
    parser.add_argument("--channels", type=str, default=None, help="Comma-separated channels (default: all)")
    parser.add_argument("--spline_smooth", type=float, default=5.0, help="Spline smoothing factor (higher = smoother)")
    parser.add_argument("--slice_stride", type=int, default=2, help="Use every Nth slice for 3D view")
    parser.add_argument("--threshold", type=float, default=0.5, help="Heatmap threshold for 3D scatter")
    parser.add_argument("--elev", type=float, default=20, help="3D view elevation")
    parser.add_argument("--azim", type=float, default=135, help="3D view azimuth")
    parser.add_argument("--save", type=str, default=None, help="Save figure to this path")
    args = parser.parse_args()

    # Load heatmaps
    print(f"Loading heatmaps: {args.heatmaps}")
    heatmap_path = Path(args.heatmaps)
    if not heatmap_path.exists():
        raise FileNotFoundError(f"Heatmap file not found: {heatmap_path}")

    heatmaps_data = np.load(heatmap_path, allow_pickle=True)
    heatmaps = heatmaps_data["heatmaps"]  # (num_slices, num_landmarks, H, W)
    z_indices = heatmaps_data["z_indices"].astype(int)

    if heatmaps.size == 0:
        raise ValueError("No heatmaps found.")

    num_slices, num_landmarks, h, w = heatmaps.shape
    print(f"  Shape: {heatmaps.shape}, slices: {z_indices[0]} to {z_indices[-1]}")

    # Parse channels
    if args.channels is None:
        channels = list(range(num_landmarks))
    else:
        channels = [int(c.strip()) for c in args.channels.split(",") if c.strip() != ""]

    for ch in channels:
        if ch < 0 or ch >= num_landmarks:
            raise ValueError(f"Invalid channel {ch}; valid: [0, {num_landmarks - 1}]")

    # Get argmax points as base data
    per_landmark_argmax = _argmax_points_numpy(heatmaps)
    per_landmark_points: Dict[str, np.ndarray] = {}
    z_vals = z_indices.astype(float)

    # Store curve metrics for display
    curve_metrics = {"curv": 0.0, "smooth": 0.0}

    def fit_all_curves(smooth_factor: float) -> None:
        """Fit spline curves for all channels."""
        total_curv = 0.0
        total_smooth = 0.0
        
        for ch in channels:
            key = str(ch)
            argmax_pts = per_landmark_argmax[key]
            smooth_pts = fit_spline(argmax_pts, smooth_factor)
            per_landmark_points[key] = smooth_pts
            
            # Compute curve metrics
            first_diff = smooth_pts[1:] - smooth_pts[:-1]
            second_diff = smooth_pts[2:] - 2 * smooth_pts[1:-1] + smooth_pts[:-2]
            total_curv += np.mean(np.sum(second_diff ** 2, axis=1))
            total_smooth += np.mean(np.sum(first_diff ** 2, axis=1))
        
        n_ch = len(channels)
        curve_metrics["curv"] = total_curv / n_ch
        curve_metrics["smooth"] = total_smooth / n_ch

    # Initial fit
    fit_all_curves(float(args.spline_smooth))

    # Setup interactive plot
    _ensure_interactive_backend()
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    # Colors for channels
    colors = ["cyan", "lime", "orange", "magenta", "yellow", "red"]

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10), facecolor="#0a0a2e")
    
    # 3D heatmap + curves view
    ax_3d = fig.add_subplot(2, 2, 1, projection="3d", facecolor="#0a0a2e")
    
    # 2D max projection view
    ax_2d = fig.add_subplot(2, 2, 2, facecolor="#0a0a2e")
    
    # Angle plot
    ax_ang = fig.add_subplot(2, 2, 3, facecolor="#0a0a2e")
    
    # Info panel
    ax_info = fig.add_subplot(2, 2, 4, facecolor="#0a0a2e")
    ax_info.axis("off")

    # Summed heatmaps
    summed_heatmaps = np.sum(heatmaps[:, channels], axis=1)

    # === 3D Scatter of heatmap values ===
    stride = max(1, int(args.slice_stride))
    sample_indices = list(range(0, num_slices, stride))
    
    xs_all, ys_all, zs_all, vs_all = [], [], [], []
    for si in sample_indices:
        slice_img = summed_heatmaps[si]
        slice_max = slice_img.max()
        if slice_max < 1e-8:
            continue
        slice_norm = slice_img / slice_max
        thresh = args.threshold
        coords = np.argwhere(slice_norm >= thresh)
        if coords.size == 0:
            continue
        vals = slice_norm[coords[:, 0], coords[:, 1]]
        if len(coords) > 500:
            pick = np.random.choice(len(coords), size=500, replace=False)
            coords = coords[pick]
            vals = vals[pick]
        ys_all.append(coords[:, 0])
        xs_all.append(coords[:, 1])
        zs_all.append(np.full(len(coords), z_vals[si]))
        vs_all.append(vals)

    if xs_all:
        xs = np.concatenate(xs_all)
        ys = np.concatenate(ys_all)
        zs = np.concatenate(zs_all)
        vs = np.concatenate(vs_all)
        ax_3d.scatter(xs, ys, zs, c=vs, cmap="hot", s=2, alpha=0.3)

    # === 3D Fitted curves ===
    line_3d = {}
    for idx, ch in enumerate(channels):
        pts = per_landmark_points[str(ch)]
        line = ax_3d.plot(pts[:, 1], pts[:, 0], z_vals, 
                          color=colors[idx % len(colors)], linewidth=2.5, label=f"ch{ch}")[0]
        line_3d[str(ch)] = line

    # Angle lines between landmarks
    angle_lines = None
    if len(channels) >= 3:
        segments = []
        ch0, ch1, ch2 = map(str, channels[:3])
        pts0, pts1, pts2 = per_landmark_points[ch0], per_landmark_points[ch1], per_landmark_points[ch2]
        for i in range(0, len(z_vals), 3):
            if not all(np.isfinite(per_landmark_points[c][i]).all() for c in [ch0, ch1, ch2]):
                continue
            z = z_vals[i]
            segments.append([[pts0[i, 1], pts0[i, 0], z], [pts1[i, 1], pts1[i, 0], z]])
            segments.append([[pts1[i, 1], pts1[i, 0], z], [pts2[i, 1], pts2[i, 0], z]])
        if segments:
            angle_lines = Line3DCollection(segments, colors="white", linewidths=0.5, alpha=0.4)
            ax_3d.add_collection3d(angle_lines)

    ax_3d.view_init(elev=args.elev, azim=args.azim)
    ax_3d.set_xlabel("")
    ax_3d.set_ylabel("")
    ax_3d.set_zlabel("Slice (z)", color="white", fontsize=10)
    ax_3d.set_xticks([])
    ax_3d.set_yticks([])
    ax_3d.tick_params(axis='z', colors="white")
    ax_3d.xaxis.pane.set_edgecolor('none')
    ax_3d.yaxis.pane.set_edgecolor('none')
    ax_3d.zaxis.pane.set_edgecolor('white')
    ax_3d.zaxis.pane.set_alpha(0.1)
    ax_3d.grid(False)
    ax_3d.set_title("3D Heatmap + Spline Curves", color="white", fontsize=12)
    ax_3d.legend(loc="upper right", fontsize=8, facecolor="#0a0a2e", labelcolor="white")

    # === 2D Max projection with curves ===
    max_proj = np.max(summed_heatmaps, axis=0)
    ax_2d.imshow(max_proj, cmap="hot", aspect="equal")
    
    line_2d = {}
    scatter_2d = {}
    for idx, ch in enumerate(channels):
        pts = per_landmark_points[str(ch)]
        line_2d[str(ch)] = ax_2d.plot(pts[:, 1], pts[:, 0], 
                                       color=colors[idx % len(colors)], linewidth=2, label=f"ch{ch}")[0]
        scatter_2d[str(ch)] = ax_2d.scatter(pts[:, 1], pts[:, 0], 
                                             c=z_vals, cmap="viridis", s=10, alpha=0.6)
    
    ax_2d.set_title("Max Projection + Curves", color="white", fontsize=12)
    ax_2d.axis("off")
    ax_2d.legend(loc="upper right", fontsize=8, facecolor="#0a0a2e", labelcolor="white")

    # === Angle plot ===
    angle_line = None
    if len(channels) >= 3:
        angles = _compute_angles(per_landmark_points, (channels[0], channels[1], channels[2]))
        angle_line = ax_ang.plot(z_vals, angles, "w-o", linewidth=1.5, markersize=3)[0]
        ax_ang.set_xlabel("Slice (z)", color="white", fontsize=10)
        ax_ang.set_ylabel("Sulcus Angle (°)", color="white", fontsize=10)
        ax_ang.set_title("Sulcus Angle vs Slice", color="white", fontsize=12)
        ax_ang.tick_params(colors="white")
        ax_ang.spines['bottom'].set_color('white')
        ax_ang.spines['left'].set_color('white')
        ax_ang.spines['top'].set_color('none')
        ax_ang.spines['right'].set_color('none')
        ax_ang.grid(True, alpha=0.2, color="white")
    else:
        ax_ang.text(0.5, 0.5, "Need 3 channels for angle", ha="center", va="center", color="white")
        ax_ang.axis("off")

    # === Info text ===
    info_text = ax_info.text(0.05, 0.9, "", va="top", color="white", fontsize=11,
                              fontfamily="monospace", transform=ax_info.transAxes)

    # === Slider ===
    fig.subplots_adjust(bottom=0.12)
    
    ax_spline = fig.add_axes([0.15, 0.04, 0.7, 0.03])
    slider_spline = Slider(
        ax=ax_spline,
        label="Spline Smoothing",
        valmin=0.0,
        valmax=100.0,
        valinit=float(args.spline_smooth),
        color="lime",
    )
    ax_spline.set_facecolor("#0a0a2e")
    slider_spline.label.set_color("white")
    slider_spline.valtext.set_color("white")

    def update_info(smooth_val: float, angles: Optional[List[float]] = None):
        txt = "CUBIC SPLINE FITTING\n"
        txt += "─" * 25 + "\n"
        txt += f"Smoothing factor: {smooth_val:.1f}\n\n"
        
        txt += "─── CURVE METRICS ───\n"
        txt += f"Curvature (2nd diff): {curve_metrics['curv']:.4f}\n"
        txt += f"Velocity (1st diff):  {curve_metrics['smooth']:.4f}\n\n"
        
        txt += f"Channels: {channels}\n"
        txt += f"Slices: {z_indices[0]} to {z_indices[-1]} ({num_slices})\n"
        
        if angles and len(channels) >= 3:
            valid_angles = [a for a in angles if np.isfinite(a)]
            if valid_angles:
                txt += f"\nAngle: {min(valid_angles):.1f}° - {max(valid_angles):.1f}°\n"
                txt += f"Mean: {np.mean(valid_angles):.1f}°"
        info_text.set_text(txt)

    def on_slider_change(val: float) -> None:
        smooth_val = float(slider_spline.val)
        fit_all_curves(smooth_val)
        
        # Update 3D lines
        for ch in channels:
            pts = per_landmark_points[str(ch)]
            line_3d[str(ch)].set_data(pts[:, 1], pts[:, 0])
            line_3d[str(ch)].set_3d_properties(z_vals)
        
        # Update 2D lines
        for ch in channels:
            pts = per_landmark_points[str(ch)]
            line_2d[str(ch)].set_data(pts[:, 1], pts[:, 0])
            scatter_2d[str(ch)].set_offsets(np.column_stack([pts[:, 1], pts[:, 0]]))
        
        # Update angle lines
        if angle_lines is not None and len(channels) >= 3:
            segments = []
            ch0, ch1, ch2 = map(str, channels[:3])
            pts0, pts1, pts2 = per_landmark_points[ch0], per_landmark_points[ch1], per_landmark_points[ch2]
            for i in range(0, len(z_vals), 3):
                if not all(np.isfinite(per_landmark_points[c][i]).all() for c in [ch0, ch1, ch2]):
                    continue
                z = z_vals[i]
                segments.append([[pts0[i, 1], pts0[i, 0], z], [pts1[i, 1], pts1[i, 0], z]])
                segments.append([[pts1[i, 1], pts1[i, 0], z], [pts2[i, 1], pts2[i, 0], z]])
            angle_lines.set_segments(segments)
        
        # Update angle plot
        angles = None
        if angle_line is not None and len(channels) >= 3:
            angles = _compute_angles(per_landmark_points, (channels[0], channels[1], channels[2]))
            angle_line.set_ydata(angles)
            ax_ang.relim()
            ax_ang.autoscale_view()
        
        update_info(smooth_val, angles)
        fig.canvas.draw_idle()

    slider_spline.on_changed(on_slider_change)
    
    # Initial update
    angles = _compute_angles(per_landmark_points, (channels[0], channels[1], channels[2])) if len(channels) >= 3 else None
    update_info(float(args.spline_smooth), angles)

    # Save if requested
    if args.save:
        fig.savefig(args.save, dpi=200, facecolor="#0a0a2e", bbox_inches="tight")
        print(f"Saved: {args.save}")

    plt.show()


if __name__ == "__main__":
    main()
