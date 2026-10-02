"""
Interactive 3D viewer for heatmaps with Leo's probability spline fitting.

Combines:
- 3D stacked heatmap visualization
- Smooth probability-based spline fitting (from leo_line_fit.py)
- Interactive smoothness slider (controls max control points)
- Sulcus angle plot

Example:
  python scripts/interactive_curve_viewer_leo.py ^
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
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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

    def fit(self, min_control=3, max_control=None, tol=0.01, verbose=True):
        """
        Fit spline with adaptive number of control points.

        Parameters:
        -----------
        min_control : int
            Starting number of control points
        max_control : int
            Maximum control points (default: n_slices)
        tol : float
            Stop when improvement < tol
        verbose : bool
            Print progress
        """
        if max_control is None:
            max_control = self.n_slices
        
        # Ensure max_control is at least min_control
        max_control = max(int(max_control), min_control)

        # Initial guess: max probability in each slice, then interpolate
        init_points = np.zeros((self.n_slices, 2))
        for i in range(self.n_slices):
            # Safe argmax for 2D slice
            flat_idx = np.argmax(self.slices[i])
            y_max, x_max = np.unravel_index(flat_idx, self.slices[i].shape)
            init_points[i] = [x_max, y_max]

        history = []
        best_result = None
        prev_score = -np.inf

        for n in range(min_control, max_control + 1):
            # Interpolate initial guess to n control points
            z_all = np.arange(self.n_slices)
            z_control = np.linspace(0, self.n_slices - 1, n)
            x_init = np.interp(z_control, z_all, init_points[:, 0])
            y_init = np.interp(z_control, z_all, init_points[:, 1])
            x0 = np.column_stack([x_init, y_init]).flatten()

            # Bounds
            bounds = [(0, self.width-1), (0, self.height-1)] * n

            # Optimize using L-BFGS-B
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

            if verbose:
                print(f"n={n:2d} | score={score:7.3f} | improvement={improvement:6.3f}")

            if best_result is None or score > best_result['score']:
                best_result = history[-1]

            # Check convergence
            if n > min_control and improvement < tol:
                if verbose:
                    print(f"Converged (improvement {improvement:.4f} < {tol})")
                break

            prev_score = score

            # Use current result as warm start for next iteration
            init_points = self.evaluate_at_slices(tck)

        return {
            'best_tck': best_result['tck'],
            'best_score': best_result['score'],
            'best_n': best_result['n_control'],
            'history': history,
            'best_points': self.evaluate_at_slices(best_result['tck'])
        }


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive 3D heatmap + Leo's Spline Fitting viewer")
    parser.add_argument("--heatmaps", type=str, required=True, help="Path to axial_heatmaps.npz")
    parser.add_argument("--channels", type=str, default=None, help="Comma-separated channels (default: all)")
    parser.add_argument("--spline_smooth", type=float, default=10.0, help="Max control points for fitting")
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

    per_landmark_points: Dict[str, np.ndarray] = {}
    z_vals = z_indices.astype(float)

    # Store curve metrics
    curve_metrics = {"log_prob": 0.0, "n_control": 0.0}

    def fit_all_curves(max_control: float) -> None:
        """Fit probability spline curves for all channels."""
        total_log_prob = 0.0
        total_n = 0.0
        
        max_ctrl_int = int(max(3, max_control))
        print(f"Fitting with max_control={max_ctrl_int}...")

        for ch in channels:
            # Extract volume for this channel: (slices, H, W)
            vol = heatmaps[:, ch, :, :]
            fitter = SimpleProbabilitySplineFitter(vol)
            
            # Fitting
            # Note: We use min_control=3 to have at least cubic spline capability
            res = fitter.fit(min_control=3, max_control=max_ctrl_int, tol=0.01, verbose=True)
            
            # Note: swapped coordinates! 
            # Interactive viewer uses [row, col]. 
            # Leo's fitter returns [x, y] which corresponds to [col, row].
            # Let's verify: 
            # In leo_line_fit.py:
            # y_max, x_max = np.unravel_index(np.argmax(self.slices[i]), ...)
            # init_points[i] = [x_max, y_max] --> [col, row]
            # So fit results are [col, row].
            # Interactive viewer plot expects [row, col] for plot(pts[:,1], pts[:,0])
            # Wait, originally interactive viewer:
            # argmax: axis=1 is col (W), axis=0 is row (H).
            # init_row = floor(max_idx / w)
            # init_col = max_idx % w
            # points = [row, col]
            # plot(pts[:, 1], pts[:, 0]) -> plot(col, row) -> x=col, y=row. Correct.
            
            # Leo fitter returns [x, y] = [col, row].
            # So we need to store as [row, col] to match viewer expectations?
            # Or just adapt the plotting code.
            # Viewer expects `per_landmark_points` to be (N, 2)
            # Viewer plot: ax.plot(pts[:, 1], pts[:, 0]) => x=pts[:, 1] (col), y=pts[:, 0] (row)
            # Leo points: [x, y] = [col, row]
            # So if we want pts[:, 1] to be col, we need pts to be [row, col].
            # Leo points are [col, row]. 
            # So pts = np.flip(leo_points, axis=1)
            
            leo_points = res['best_points']
            pts_row_col = np.flip(leo_points, axis=1)
            per_landmark_points[str(ch)] = pts_row_col
            
            total_log_prob += res['best_score']
            total_n += res['best_n']
        
        n_ch = len(channels)
        curve_metrics["log_prob"] = total_log_prob / n_ch
        curve_metrics["n_control"] = total_n / n_ch
        print("Done fitting.")

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
        # pts is [row, col], so x=col, y=row
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
    ax_3d.set_title("3D Heatmap + Probability Spline Curves", color="white", fontsize=12)
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
        label="Max Control Pts",
        valmin=3.0,
        valmax=50.0,
        valinit=float(args.spline_smooth),
        color="lime",
        valstep=1.0
    )
    ax_spline.set_facecolor("#0a0a2e")
    slider_spline.label.set_color("white")
    slider_spline.valtext.set_color("white")

    def update_info(max_control: float, angles: Optional[List[float]] = None):
        txt = "PROBABILITY SPLINE FITTING\n"
        txt += "─" * 25 + "\n"
        txt += f"Max Control Points: {int(max_control)}\n\n"
        
        txt += "─── FIT METRICS ───\n"
        txt += f"Avg Log Prob: {curve_metrics['log_prob']:.2f}\n"
        txt += f"Avg Points Used: {curve_metrics['n_control']:.1f}\n\n"
        
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
