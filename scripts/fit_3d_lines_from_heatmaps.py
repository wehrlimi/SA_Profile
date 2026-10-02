"""
Fit smooth 3D lines from axial heatmaps saved by two_stage_inference_heatmaps.py.
Uses a regression objective with curvature regularization and negative log-likelihood.
Also visualizes the fitted curve over stacked heatmap slices.
"""

import json
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F


def _soft_argmax_2d(heatmap: np.ndarray) -> Tuple[float, float]:
    """
    Compute weighted centroid (soft argmax) in heatmap space.
    Returns (x, y) in heatmap coordinates.
    """
    heatmap = np.asarray(heatmap, dtype=np.float64)
    total = heatmap.sum()
    if total <= 0:
        return float("nan"), float("nan")

    h, w = heatmap.shape
    xs = np.arange(h, dtype=np.float64)
    ys = np.arange(w, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")

    x = float((grid_x * heatmap).sum() / total)
    y = float((grid_y * heatmap).sum() / total)
    return x, y


def _heatmap_to_voxel(
    coord_xy: Tuple[float, float],
    pre_rot_shape: Tuple[int, int],
    image_size: Tuple[int, int],
    transpose_mode: str = "swap_alt",
    center_mirror: bool = True,
) -> Tuple[float, float]:
    """
    Map heatmap (resized rotated space) -> voxel (unrotated) coordinates.
    Mirrors _predict_landmarks in scripts/two_stage_inference.py.
    """
    pre_rot_height, _ = pre_rot_shape
    original_size = np.array(np.rot90(np.zeros(pre_rot_shape), k=1).shape)
    scale = original_size / np.array(image_size)

    x, y = coord_xy
    x_rot = x * scale[0]
    y_rot = y * scale[1]

    if transpose_mode == "swap":
        i = y_rot
        j = pre_rot_height - 1 - x_rot
    else:
        # Default mirrors the opposite diagonal (swap_alt).
        i = pre_rot_height - 1 - y_rot
        j = x_rot
    if center_mirror:
        # Point mirror around slice center to align with volume orientation.
        i = pre_rot_shape[0] - 1 - i
        j = pre_rot_shape[1] - 1 - j
    return float(i), float(j)


def _fit_curve_points(
    probs: torch.Tensor,
    steps: int,
    lr: float,
    lambda_curv: float,
    clamp: bool = True,
    init_points: torch.Tensor = None,
    lambda_smooth: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Fit per-slice points in heatmap space using NLL + smoothness regularization.
    
    Args:
        probs: (N, H, W) normalized per slice.
        steps: Number of optimization steps.
        lr: Learning rate.
        lambda_curv: Weight for curvature penalty (2nd order differences).
        clamp: Whether to clamp points to image bounds.
        init_points: Initial point positions.
        lambda_smooth: Weight for smoothness penalty (1st order differences).
    
    Returns dict with fitted points and sampled probabilities.
    """
    n, h, w = probs.shape
    device = probs.device
    if init_points is None:
        init_points = torch.stack([torch.tensor([h / 2.0, w / 2.0], device=device) for _ in range(n)])
    points = torch.nn.Parameter(init_points.to(device))

    optimizer = torch.optim.Adam([points], lr=lr)

    def curve_loss(points_xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grid = points_xy.clone()
        # grid_sample expects (x=col, y=row)
        grid[:, 0] = 2.0 * grid[:, 1] / (w - 1) - 1.0
        grid[:, 1] = 2.0 * grid[:, 0] / (h - 1) - 1.0
        grid = grid.view(n, 1, 1, 2)

        probs_ = probs.unsqueeze(1)
        sampled = F.grid_sample(
            probs_,
            grid,
            align_corners=True,
            mode="bilinear",
        ).view(n)

        data_loss = -torch.log(sampled).mean()
        
        # 2nd order: curvature penalty
        second_diff = points_xy[2:] - 2 * points_xy[1:-1] + points_xy[:-2]
        curvature_loss = (second_diff ** 2).sum(dim=1).mean()
        
        # 1st order: smoothness penalty (penalizes velocity changes)
        first_diff = points_xy[1:] - points_xy[:-1]
        smoothness_loss = (first_diff ** 2).sum(dim=1).mean() if lambda_smooth > 0 else torch.tensor(0.0, device=device)
        
        total_loss = data_loss + lambda_curv * curvature_loss + lambda_smooth * smoothness_loss
        return total_loss, data_loss, curvature_loss

    last_sampled = None
    last_data_loss = None
    last_curv_loss = None
    last_smooth_loss = None
    for _ in range(steps):
        optimizer.zero_grad()
        loss, _, _ = curve_loss(points)
        loss.backward()
        optimizer.step()
        if clamp:
            with torch.no_grad():
                points[:, 0].clamp_(0, h - 1)
                points[:, 1].clamp_(0, w - 1)

    with torch.no_grad():
        # Compute final losses
        grid = points.clone()
        grid[:, 0] = 2.0 * grid[:, 1] / (w - 1) - 1.0
        grid[:, 1] = 2.0 * grid[:, 0] / (h - 1) - 1.0
        grid = grid.view(n, 1, 1, 2)
        
        probs_ = probs.unsqueeze(1)
        sampled = F.grid_sample(probs_, grid, align_corners=True, mode="bilinear").view(n)
        last_sampled = sampled.detach()
        
        last_data_loss = -torch.log(sampled).mean().detach()
        
        second_diff = points[2:] - 2 * points[1:-1] + points[:-2]
        last_curv_loss = (second_diff ** 2).sum(dim=1).mean().detach()
        
        first_diff = points[1:] - points[:-1]
        last_smooth_loss = (first_diff ** 2).sum(dim=1).mean().detach()

    return {
        "points": points.detach(),
        "sampled": last_sampled,
        "data_loss": last_data_loss,
        "curvature_loss": last_curv_loss,
        "smoothness_loss": last_smooth_loss,
    }


def _voxel_to_world(voxel_xyz: np.ndarray, affine: np.ndarray) -> np.ndarray:
    coords = np.concatenate([voxel_xyz, np.ones((voxel_xyz.shape[0], 1))], axis=1)
    world = (affine @ coords.T).T[:, :3]
    return world


def _write_slicer_markups(points_world: List[List[float]], output_path: Path) -> None:
    markup = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "coordinateUnits": "mm",
                "locked": False,
                "fixedNumberOfControlPoints": False,
                "labelFormat": "%N-%d",
                "lastUsedControlPointNumber": len(points_world),
                "controlPoints": [
                    {
                        "id": str(i + 1),
                        "label": f"AxialLabels-{i + 1}",
                        "description": "",
                        "associatedNodeID": "vtkMRMLScalarVolumeNode1",
                        "position": points_world[i],
                        "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                        "selected": True,
                        "locked": True,
                        "visibility": True,
                        "positionStatus": "defined",
                    }
                    for i in range(len(points_world))
                ],
                "measurements": [],
                "display": {
                    "visibility": True,
                    "opacity": 1.0,
                    "color": [0.4, 1.0, 1.0],
                    "selectedColor": [1.0, 0.5000076295109483, 0.5000076295109483],
                    "activeColor": [0.4, 1.0, 0.0],
                    "propertiesLabelVisibility": False,
                    "pointLabelsVisibility": True,
                    "textScale": 3.0,
                    "glyphType": "Sphere3D",
                    "glyphScale": 3.0,
                    "glyphSize": 5.0,
                    "useGlyphScale": True,
                    "sliceProjection": False,
                    "sliceProjectionUseFiducialColor": True,
                    "sliceProjectionOutlinedBehindSlicePlane": False,
                    "sliceProjectionColor": [1.0, 1.0, 1.0],
                    "sliceProjectionOpacity": 0.6,
                    "lineThickness": 0.2,
                    "lineColorFadingStart": 1.0,
                    "lineColorFadingEnd": 10.0,
                    "lineColorFadingSaturation": 1.0,
                    "lineColorFadingHueOffset": 0.0,
                    "handlesInteractive": False,
                    "translationHandleVisibility": True,
                    "rotationHandleVisibility": True,
                    "scaleHandleVisibility": False,
                    "interactionHandleScale": 3.0,
                    "snapMode": "toVisibleSurface",
                },
            }
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(markup, f, indent=4)


def main():
    import argparse
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    parser = argparse.ArgumentParser(description="Fit 3D lines from axial heatmaps")
    parser.add_argument("--heatmaps", type=str, required=True, help="Path to axial_heatmaps.npz")
    parser.add_argument("--input_volume", type=str, required=True, help="Path to input NIfTI volume")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument(
        "--channels",
        type=str,
        default=None,
        help="Comma-separated channels to fit/plot (default: all)",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=1000, help="Optimization steps")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--lambda_curv", type=float, default=1e-1, help="Curvature weight")
    parser.add_argument("--init", type=str, default="argmax", choices=["argmax", "center"], help="Point initialization")
    parser.add_argument("--stack_stride", type=int, default=2, help="Use every Nth slice for plot")
    parser.add_argument("--height_scale", type=float, default=0.0, help="Lift surface by heatmap value")
    parser.add_argument("--vmin", type=float, default=None, help="Colormap min")
    parser.add_argument("--vmax", type=float, default=None, help="Colormap max")
    parser.add_argument("--elev", type=float, default=4.0, help="3D view elevation")
    parser.add_argument("--azim", type=float, default=20.0, help="3D view azimuth")
    parser.add_argument("--plot_path", type=str, default=None, help="Save plot to this path (png)")
    parser.add_argument("--interactive", action="store_true", help="Show interactive plot instead of saving PNG")
    parser.add_argument("--no_plot", action="store_true", help="Skip plotting entirely")
    parser.add_argument("--slicer_output", action="store_true", help="Save Slicer .mrk.json for fitted lines")
    parser.add_argument("--case_id", type=str, default=None, help="Case ID for Slicer output filename")
    parser.add_argument(
        "--no_center_mirror",
        action="store_true",
        help="Disable point mirror around slice center (not recommended)",
    )
    args = parser.parse_args()

    heatmaps_data = np.load(args.heatmaps, allow_pickle=True)
    heatmaps = heatmaps_data["heatmaps"]  # (num_slices, num_landmarks, H, W)
    z_indices = heatmaps_data["z_indices"].astype(int)
    image_size = tuple(heatmaps_data["image_size"].tolist())

    affine = heatmaps_data["affine"] if "affine" in heatmaps_data else None
    nii = nib.load(args.input_volume)
    volume = nii.get_fdata()
    if affine is None:
        affine = nii.affine

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

    per_landmark_voxel = {str(ch): [] for ch in channels}
    per_landmark_world = {str(ch): [] for ch in channels}
    per_landmark_slices = {str(ch): [] for ch in channels}
    per_landmark_sampled = {str(ch): [] for ch in channels}
    per_landmark_points = {}
    per_landmark_probs = {}
    per_landmark_init_points = {}
    per_landmark_argmax_init_points = {}
    last_points = {}

    for ch in channels:
        probs_t = torch.from_numpy(heatmaps[:, ch].astype(np.float32)).to(args.device)
        probs_t = probs_t.view(num_slices, -1)
        probs_t = torch.softmax(probs_t, dim=1).view(num_slices, h, w)
        probs_t = probs_t.clamp(min=1e-8)
        per_landmark_probs[str(ch)] = probs_t

        flat = heatmaps[:, ch].reshape(num_slices, -1)
        max_idx = flat.argmax(axis=1)
        init_row = (max_idx // w).astype(np.float32)
        init_col = (max_idx % w).astype(np.float32)
        argmax_init = torch.stack(
            [torch.tensor([init_row[i], init_col[i]], device=args.device) for i in range(num_slices)]
        )
        per_landmark_argmax_init_points[str(ch)] = argmax_init.clone()

        if args.init == "argmax":
            init_points = argmax_init
        else:
            init_points = None
        per_landmark_init_points[str(ch)] = init_points.clone() if init_points is not None else None
        fit = _fit_curve_points(
            probs=probs_t,
            steps=int(args.steps),
            lr=float(args.lr),
            lambda_curv=float(args.lambda_curv),
            clamp=True,
            init_points=init_points,
        )

        fitted_points = fit["points"].cpu().numpy()
        sampled = fit["sampled"].cpu().numpy()
        per_landmark_points[str(ch)] = fitted_points
        per_landmark_sampled[str(ch)] = sampled.tolist()
        last_points[str(ch)] = fit["points"].detach()

        for idx, z in enumerate(z_indices):
            if z < 0 or z >= volume.shape[2]:
                continue
            pre_rot_shape = volume[:, :, z].shape
            row, col = fitted_points[idx]
            i, j = _heatmap_to_voxel(
                (row, col),
                pre_rot_shape,
                image_size,
                center_mirror=not args.no_center_mirror,
            )
            voxel = np.array([i, j, float(z)], dtype=np.float64)
            per_landmark_voxel[str(ch)].append(voxel.tolist())
            per_landmark_slices[str(ch)].append(int(z))
            world = _voxel_to_world(voxel.reshape(1, 3), affine)[0]
            per_landmark_world[str(ch)].append(world.tolist())

    output = {
        "points_world": per_landmark_world,
        "points_voxel": per_landmark_voxel,
        "slice_indices": per_landmark_slices,
        "image_size": list(image_size),
        "channels": channels,
        "sampled_probs": per_landmark_sampled,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved line fits: {output_path}")

    if args.slicer_output:
        case_id = args.case_id or output_path.stem
        slicer_path = output_path.parent / f"AxialLabels_{case_id}.mrk.json"
        slicer_points = []
        for ch in channels:
            points_world = per_landmark_world[str(ch)]
            for x, y, z in points_world:
                slicer_points.append([-x, -y, z])
        _write_slicer_markups(slicer_points, slicer_path)
        print(f"Saved Slicer markups: {slicer_path}")

    if args.no_plot:
        print("Plotting skipped (--no_plot).")
        return

    plot_path = args.plot_path
    if plot_path is None:
        plot_path = str(output_path.with_suffix(".png"))

    stride = max(1, int(args.stack_stride))
    slice_indices = list(range(0, num_slices, stride))
    xs = np.arange(w)
    ys = np.arange(h)
    grid_x, grid_y = np.meshgrid(xs, ys)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=float(args.elev), azim=float(args.azim))

    summed_heatmaps = np.sum(heatmaps[:, channels], axis=1)
    vmin = float(np.min(summed_heatmaps)) if args.vmin is None else float(args.vmin)
    vmax = float(np.max(summed_heatmaps)) if args.vmax is None else float(args.vmax)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.magma

    height_scale = float(args.height_scale)
    for si in slice_indices:
        z_val = float(z_indices[si]) if si < len(z_indices) else float(si)
        slice_img = summed_heatmaps[si]
        colors = cmap(norm(slice_img))
        z_surface = z_val + height_scale * slice_img
        ax.plot_surface(
            grid_x,
            grid_y,
            z_surface,
            rstride=1,
            cstride=1,
            facecolors=colors,
            shade=True,
            antialiased=False,
        )

    line_colors = ["cyan", "lime", "orange", "magenta", "yellow", "white"]
    line_artists = []
    for idx, ch in enumerate(channels):
        fitted_points = per_landmark_points[str(ch)]
        line_z = z_indices.astype(float)
        if height_scale != 0.0:
            sampled = np.array(per_landmark_sampled[str(ch)], dtype=np.float64)
            line_z = line_z + height_scale * sampled
        line = ax.plot(
            fitted_points[:, 1],
            fitted_points[:, 0],
            line_z,
            color=line_colors[idx % len(line_colors)],
            linewidth=2.0,
            label=f"ch{ch}",
        )[0]
        line_artists.append(line)

    ax.set_title("Fitted curves (summed heatmaps)")
    ax.set_xlabel("x (heatmap)")
    ax.set_ylabel("y (heatmap)")
    ax.set_zlabel("slice z")
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
    ax.legend(loc="upper right")
    if args.interactive:
        fig.subplots_adjust(bottom=0.18)
        log_min = -1.0
        log_max = 4.0
        ax_lambda = fig.add_axes([0.18, 0.06, 0.64, 0.03])
        slider = Slider(
            ax=ax_lambda,
            label="log10(lambda_curv)",
            valmin=log_min,
            valmax=log_max,
            valinit=np.log10(max(float(args.lambda_curv), 1e-12)),
        )
        lambda_text = fig.text(
            0.82,
            0.02,
            f"lambda_curv={10 ** slider.val:.3e}",
            ha="left",
            va="bottom",
        )

        def _refit_lines(val: float) -> None:
            lambda_val = 10.0 ** float(val)
            lambda_text.set_text(f"lambda_curv={lambda_val:.3e}")
            data_losses = []
            curv_losses = []
            for idx, ch in enumerate(channels):
                key = str(ch)
                init = per_landmark_argmax_init_points[key].clone()
                fit = _fit_curve_points(
                    probs=per_landmark_probs[key],
                    steps=int(args.steps),
                    lr=float(args.lr),
                    lambda_curv=lambda_val,
                    clamp=True,
                    init_points=init,
                )
                if fit["data_loss"] is not None:
                    data_losses.append(float(fit["data_loss"].cpu().item()))
                if fit["curvature_loss"] is not None:
                    curv_losses.append(float(fit["curvature_loss"].cpu().item()))
                fitted_points = fit["points"].cpu().numpy()
                sampled = fit["sampled"].cpu().numpy()
                per_landmark_points[key] = fitted_points
                per_landmark_sampled[key] = sampled.tolist()
                last_points[key] = fit["points"].detach()

                line_z = z_indices.astype(float)
                if height_scale != 0.0:
                    line_z = line_z + height_scale * sampled
                line_artists[idx].set_data(fitted_points[:, 1], fitted_points[:, 0])
                line_artists[idx].set_3d_properties(line_z)
            fig.canvas.draw_idle()
            if data_losses or curv_losses:
                data_mean = float(np.mean(data_losses)) if data_losses else float("nan")
                curv_mean = float(np.mean(curv_losses)) if curv_losses else float("nan")
                print(f"lambda_curv={lambda_val:.3e} data_loss={data_mean:.6f} curv_loss={curv_mean:.6f}")

        slider.on_changed(_refit_lines)
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        print(f"Saved plot: {plot_path}")
        plt.close()


if __name__ == "__main__":
    main()
