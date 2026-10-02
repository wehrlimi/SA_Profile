"""
Interactive helper to tune lambda_curv using saved axial heatmaps.

Example:
  python scripts/find_lamda_curve.py ^
    --case_dir .\inference_output_all\FB_752409____FB,1923389247_study_2aeb6c57_res_256_epoch_50_contr_0 ^
    --channels 0,1,2

You can also pass --heatmaps directly if you prefer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project root to path to enable imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# _fit_curve_points is kept for compatibility if needed later.
from scripts.fit_3d_lines_from_heatmaps import _fit_curve_points  # noqa: E402


def _resolve_heatmap_path(case_dir: str | None, heatmaps_path: str | None) -> Path:
    if heatmaps_path:
        return Path(heatmaps_path)
    if case_dir:
        return Path(case_dir) / "axial_heatmaps.npz"
    raise ValueError("Provide either --case_dir or --heatmaps.")


def _argmax_points_numpy(heatmaps: np.ndarray) -> Dict[str, np.ndarray]:
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


def _compute_mean_radius_weights(
    points: np.ndarray,
    image_size: Tuple[int, int],
    mean_outlier_frac: float,
) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    n = len(points)
    weights = np.ones(n, dtype=np.float64)
    mean_xy = None
    max_dist = 0.0
    if mean_outlier_frac > 0 and n > 0:
        mean_xy = np.mean(points, axis=0)
        h, w = image_size
        max_dist = float(mean_outlier_frac) * np.hypot(h, w)
        dist = np.linalg.norm(points - mean_xy, axis=1)
        weights = (dist <= max_dist).astype(np.float64)
        if np.sum(weights) == 0:
            # Ensure at least one slice remains to fit.
            keep_idx = int(np.argmin(dist))
            weights[keep_idx] = 1.0
    return weights, mean_xy, max_dist


def _ensure_interactive_backend() -> None:
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
    parser = argparse.ArgumentParser(description="Interactively tune lambda_curv for heatmap curve fitting")
    parser.add_argument("--case_dir", type=str, default=None, help="Case folder containing axial_heatmaps.npz")
    parser.add_argument("--heatmaps", type=str, default=None, help="Path to axial_heatmaps.npz")
    parser.add_argument("--channels", type=str, default=None, help="Comma-separated channels to fit/plot")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=150, help="Optimization steps per iteration")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--lambda_curv", type=float, default=1e-1, help="Curvature weight")
    parser.add_argument(
        "--mean_outlier_frac",
        type=float,
        default=0.5,
        help="Drop points farther than this fraction of the image diagonal from mean",
    )
    parser.add_argument("--point_stride", type=int, default=2, help="Plot every Nth slice point per channel")
    parser.add_argument("--heatmap_stride", type=int, default=4, help="Use every Nth slice in 3D heatmap view")
    parser.add_argument("--heatmap_quantile", type=float, default=0.985, help="Quantile threshold for 3D heatmap points")
    parser.add_argument("--heatmap_max_points", type=int, default=400, help="Max points per slice in 3D heatmap view")
    parser.add_argument("--angle_stride", type=int, default=1, help="Use every Nth slice for angle lines")
    parser.add_argument("--plot_path", type=str, default=None, help="Where to save PNG snapshot")
    args = parser.parse_args()

    heatmap_path = _resolve_heatmap_path(args.case_dir, args.heatmaps)
    if not heatmap_path.exists():
        raise FileNotFoundError(f"Heatmap file not found: {heatmap_path}")

    heatmaps_data = np.load(heatmap_path, allow_pickle=True)
    heatmaps = heatmaps_data["heatmaps"]  # (num_slices, num_landmarks, H, W)
    z_indices = heatmaps_data["z_indices"].astype(int)

    if heatmaps.size == 0:
        raise ValueError("No heatmaps found in the provided .npz file.")

    num_slices, num_landmarks, h, w = heatmaps.shape
    if args.channels is None:
        channels = list(range(num_landmarks))
    else:
        channels = [int(c.strip()) for c in args.channels.split(",") if c.strip() != ""]
    if not channels:
        raise ValueError("--channels must include at least one channel index")
    for ch in channels:
        if ch < 0 or ch >= num_landmarks:
            raise ValueError(f"--channels contains invalid index {ch}; valid range is [0, {num_landmarks - 1}]")

    per_landmark_argmax_numpy = _argmax_points_numpy(heatmaps)
    per_landmark_points: Dict[str, np.ndarray] = {}
    per_landmark_weights: Dict[str, np.ndarray] = {}
    per_landmark_mean: Dict[str, Optional[np.ndarray]] = {}
    per_landmark_mean_radius: Dict[str, float] = {}

    def _run_iteration(
        lambda_val: float,
        mean_outlier_frac: float,
    ) -> Optional[List[float]]:
        for ch in channels:
            key = str(ch)
            raw_points = per_landmark_argmax_numpy[key]
            weights, mean_xy, max_dist = _compute_mean_radius_weights(
                raw_points,
                image_size=(h, w),
                mean_outlier_frac=float(mean_outlier_frac),
            )
            per_landmark_weights[key] = weights
            per_landmark_mean[key] = mean_xy
            per_landmark_mean_radius[key] = max_dist
            keep_mask = weights.astype(bool)
            heatmaps_ch = heatmaps[:, ch].copy()
            heatmaps_keep = heatmaps_ch[keep_mask]

            probs_t = torch.from_numpy(heatmaps_keep.astype(np.float32)).to(args.device)
            probs_t = probs_t.view(len(heatmaps_keep), -1)
            probs_t = torch.softmax(probs_t, dim=1).view(len(heatmaps_keep), h, w)
            probs_t = probs_t.clamp(min=1e-8)

            flat = heatmaps_keep.reshape(len(heatmaps_keep), -1)
            max_idx = flat.argmax(axis=1)
            init_row = (max_idx // w).astype(np.float32)
            init_col = (max_idx % w).astype(np.float32)
            init_points = torch.stack(
                [torch.tensor([init_row[i], init_col[i]], device=args.device) for i in range(len(heatmaps_keep))]
            )

            fit = _fit_curve_points(
                probs=probs_t,
                steps=int(args.steps),
                lr=float(args.lr),
                lambda_curv=float(lambda_val),
                clamp=True,
                init_points=init_points,
            )
            full_points = np.full((num_slices, 2), np.nan, dtype=np.float64)
            full_points[keep_mask] = fit["points"].cpu().numpy()
            per_landmark_points[key] = full_points

        angle_channels = (channels[0], channels[1], channels[2]) if len(channels) >= 3 else None
        angles = _compute_angles(per_landmark_points, angle_channels) if angle_channels else None
        return angles

    _ensure_interactive_backend()
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    summed_heatmaps = np.sum(heatmaps[:, channels], axis=1)
    max_proj = np.max(summed_heatmaps, axis=0)
    z_vals = z_indices.astype(float)

    angle_stride = max(1, int(args.angle_stride))
    plot_path = Path(args.plot_path) if args.plot_path else None

    _run_iteration(float(args.lambda_curv), float(args.mean_outlier_frac))

    fig = plt.figure(figsize=(12, 8))
    ax_hm = fig.add_subplot(2, 2, 1)
    ax_3d = fig.add_subplot(2, 2, 2, projection="3d")
    ax_ang = fig.add_subplot(2, 2, 3)
    ax_info = fig.add_subplot(2, 2, 4)
    ax_info.axis("off")

    ax_hm.imshow(max_proj, cmap="magma")
    ax_hm.set_title("Max-proj heatmap + fitted points")
    ax_hm.set_xlabel("x (heatmap)")
    ax_hm.set_ylabel("y (heatmap)")

    colors = ["cyan", "lime", "orange", "magenta", "yellow", "white"]
    raw_scatter = {}
    fit_scatter = {}
    fit_lines_2d = {}
    out_scatter = {}
    mean_scatter = {}
    mean_circle = {}

    for idx, ch in enumerate(channels):
        key = str(ch)
        raw = per_landmark_argmax_numpy[key]
        raw_scatter[key] = ax_hm.scatter(
            raw[:, 1],
            raw[:, 0],
            s=6,
            alpha=0.25,
            color="white",
        )
        pts = per_landmark_points[key]
        valid_pts = np.isfinite(pts).all(axis=1)
        fit_scatter[key] = ax_hm.scatter(
            pts[valid_pts, 1],
            pts[valid_pts, 0],
            s=8,
            alpha=0.7,
            color=colors[idx % len(colors)],
            label=f"ch{ch}",
        )
        fit_lines_2d[key] = ax_hm.plot(
            pts[:, 1],
            pts[:, 0],
            color=colors[idx % len(colors)],
            linewidth=1.0,
            alpha=0.7,
        )[0]
        out_raw = raw[per_landmark_weights[key] == 0]
        out_scatter[key] = ax_hm.scatter(
            out_raw[:, 1],
            out_raw[:, 0],
            s=14,
            alpha=0.9,
            color="red",
            marker="x",
        )
        mean_xy = per_landmark_mean.get(key)
        radius = per_landmark_mean_radius.get(key, 0.0)
        if mean_xy is not None and radius > 0:
            circle = plt.Circle(
                (mean_xy[1], mean_xy[0]),
                radius=radius,
                edgecolor="white",
                facecolor="none",
                linewidth=1.2,
                alpha=0.9,
            )
            ax_hm.add_patch(circle)
            mean_circle[key] = circle
            mean_scatter[key] = ax_hm.scatter(
                mean_xy[1],
                mean_xy[0],
                s=30,
                color="white",
                edgecolors="black",
                linewidths=0.6,
                zorder=3,
                label=f"mean ch{ch}",
            )
    ax_hm.legend(loc="upper right", fontsize=8)

    stride = max(1, int(args.heatmap_stride))
    sample_indices = list(range(0, summed_heatmaps.shape[0], stride))
    xs_all = []
    ys_all = []
    zs_all = []
    vs_all = []
    for si in sample_indices:
        slice_img = summed_heatmaps[si]
        thresh = np.quantile(slice_img, float(args.heatmap_quantile))
        coords = np.argwhere(slice_img >= thresh)
        if coords.size == 0:
            continue
        vals = slice_img[coords[:, 0], coords[:, 1]]
        if coords.shape[0] > int(args.heatmap_max_points):
            pick = np.random.choice(coords.shape[0], size=int(args.heatmap_max_points), replace=False)
            coords = coords[pick]
            vals = vals[pick]
        ys_all.append(coords[:, 0])
        xs_all.append(coords[:, 1])
        zs_all.append(np.full(coords.shape[0], z_vals[si]))
        vs_all.append(vals)
    if xs_all:
        xs = np.concatenate(xs_all)
        ys = np.concatenate(ys_all)
        zs = np.concatenate(zs_all)
        vs = np.concatenate(vs_all)
        ax_3d.scatter(xs, ys, zs, c=vs, cmap="magma", s=3, alpha=0.15)

    line_3d = {}
    for idx, ch in enumerate(channels):
        pts = per_landmark_points[str(ch)]
        line = ax_3d.plot(pts[:, 1], pts[:, 0], z_vals, color=colors[idx % len(colors)], linewidth=2.0)[0]
        line_3d[str(ch)] = line

    angle_lines = None
    if len(channels) >= 3:
        segments = []
        ch0, ch1, ch2 = map(str, channels[:3])
        pts0 = per_landmark_points[ch0]
        pts1 = per_landmark_points[ch1]
        pts2 = per_landmark_points[ch2]
        for i in range(0, len(z_vals), angle_stride):
            if not (np.isfinite(pts0[i]).all() and np.isfinite(pts1[i]).all() and np.isfinite(pts2[i]).all()):
                continue
            z = z_vals[i]
            segments.append([[pts0[i, 1], pts0[i, 0], z], [pts1[i, 1], pts1[i, 0], z]])
            segments.append([[pts1[i, 1], pts1[i, 0], z], [pts2[i, 1], pts2[i, 0], z]])
        angle_lines = Line3DCollection(segments, colors="white", linewidths=0.6, alpha=0.5)
        ax_3d.add_collection3d(angle_lines)

    ax_3d.set_title("3D heatmap + spline + angle lines")
    ax_3d.set_xlabel("x (heatmap)")
    ax_3d.set_ylabel("y (heatmap)")
    ax_3d.set_zlabel("slice index")
    ax_3d.view_init(elev=20, azim=35)

    angle_line = None
    if len(channels) >= 3:
        angles = _compute_angles(per_landmark_points, (channels[0], channels[1], channels[2]))
        x_vals = np.linspace(0.0, 1.0, len(angles)) if len(angles) > 1 else np.array([0.0])
        angle_line = ax_ang.plot(x_vals, angles, marker="o", linewidth=1.2, markersize=3)[0]
        ax_ang.set_xlabel("Normalized Slice Index (0 to 1)")
        ax_ang.set_ylabel("Angle (degrees)")
        ax_ang.set_title("Sulcus angle (from fitted points)")
        ax_ang.grid(True, alpha=0.3)
    else:
        ax_ang.axis("off")
        ax_ang.text(0.5, 0.5, "Sulcus angle plot requires 3 channels", ha="center", va="center")

    fig.subplots_adjust(bottom=0.2)
    ax_lambda = fig.add_axes([0.15, 0.08, 0.7, 0.03])
    ax_mean = fig.add_axes([0.15, 0.03, 0.7, 0.03])
    ax_zoom = fig.add_axes([0.15, 0.13, 0.7, 0.03])
    slider_lambda = Slider(
        ax=ax_lambda,
        label="log10(lambda_curv)",
        valmin=-4.0,
        valmax=2.0,
        valinit=np.log10(max(float(args.lambda_curv), 1e-12)),
    )
    slider_mean = Slider(
        ax=ax_mean,
        label="mean_outlier_frac",
        valmin=0.05,
        valmax=1.0,
        valinit=float(args.mean_outlier_frac),
    )
    slider_zoom = Slider(
        ax=ax_zoom,
        label="zoom",
        valmin=1.0,
        valmax=6.0,
        valinit=1.0,
    )
    info_text = ax_info.text(0.05, 0.9, "", va="top")

    zoom_state = {"scale": float(slider_zoom.val)}

    def _apply_zoom(scale: float) -> None:
        if not per_landmark_points:
            return
        all_pts = np.concatenate([per_landmark_points[str(ch)] for ch in channels], axis=0)
        if not np.isfinite(all_pts).any():
            return
        valid = np.isfinite(all_pts).all(axis=1)
        if not np.any(valid):
            return
        center_yx = np.mean(all_pts[valid], axis=0)
        if not np.all(np.isfinite(center_yx)):
            return
        extent = max(h, w) / (2.0 * max(scale, 1e-6))
        ax_hm.set_xlim(center_yx[1] - extent, center_yx[1] + extent)
        ax_hm.set_ylim(center_yx[0] + extent, center_yx[0] - extent)
        ax_3d.set_xlim(center_yx[1] - extent, center_yx[1] + extent)
        ax_3d.set_ylim(center_yx[0] - extent, center_yx[0] + extent)

    def _update(_val: float) -> None:
        lambda_val = 10.0 ** float(slider_lambda.val)
        mean_frac = float(slider_mean.val)
        zoom_val = float(slider_zoom.val)
        _run_iteration(lambda_val, mean_frac)
        info_text.set_text(
            "lambda_curv={:.3e}\nmean_outlier_frac={:.2f}\nzoom={:.2f}".format(
                lambda_val, mean_frac, zoom_val
            )
        )

        for idx, ch in enumerate(channels):
            key = str(ch)
            pts = per_landmark_points[key]
            valid_pts = np.isfinite(pts).all(axis=1)
            fit_scatter[key].set_offsets(np.column_stack([pts[valid_pts, 1], pts[valid_pts, 0]]))
            fit_lines_2d[key].set_data(pts[:, 1], pts[:, 0])
            out_raw = per_landmark_argmax_numpy[key][per_landmark_weights[key] == 0]
            out_scatter[key].set_offsets(np.column_stack([out_raw[:, 1], out_raw[:, 0]]) if len(out_raw) else np.empty((0, 2)))
            line_3d[key].set_data(pts[:, 1], pts[:, 0])
            line_3d[key].set_3d_properties(z_vals)
            if key in mean_circle:
                mean_circle[key].set_radius(per_landmark_mean_radius[key])

        if angle_line is not None:
            angles = _compute_angles(per_landmark_points, (channels[0], channels[1], channels[2]))
            x_vals = np.linspace(0.0, 1.0, len(angles)) if len(angles) > 1 else np.array([0.0])
            angle_line.set_data(x_vals, angles)
            ax_ang.relim()
            ax_ang.autoscale_view()

        if angle_lines is not None and len(channels) >= 3:
            segments = []
            ch0, ch1, ch2 = map(str, channels[:3])
            pts0 = per_landmark_points[ch0]
            pts1 = per_landmark_points[ch1]
            pts2 = per_landmark_points[ch2]
            for i in range(0, len(z_vals), angle_stride):
                if not (np.isfinite(pts0[i]).all() and np.isfinite(pts1[i]).all() and np.isfinite(pts2[i]).all()):
                    continue
                z = z_vals[i]
                segments.append([[pts0[i, 1], pts0[i, 0], z], [pts1[i, 1], pts1[i, 0], z]])
                segments.append([[pts1[i, 1], pts1[i, 0], z], [pts2[i, 1], pts2[i, 0], z]])
            angle_lines.set_segments(segments)

        zoom_state["scale"] = zoom_val
        _apply_zoom(zoom_val)

        fig.canvas.draw_idle()

    slider_lambda.on_changed(_update)
    slider_mean.on_changed(_update)
    slider_zoom.on_changed(_update)
    _update(0.0)

    def _on_scroll(event) -> None:
        if event.inaxes != ax_3d:
            return
        direction = 1.1 if event.button == "up" else 0.9
        new_scale = max(1.0, min(12.0, zoom_state["scale"] * direction))
        zoom_state["scale"] = new_scale
        slider_zoom.set_val(new_scale)

    fig.canvas.mpl_connect("scroll_event", _on_scroll)

    if plot_path:
        fig.savefig(plot_path, dpi=150)
        print(f"Saved snapshot: {plot_path}")

    plt.show()


if __name__ == "__main__":
    main()
