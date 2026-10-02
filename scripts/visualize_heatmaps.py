"""
Visualize stacked axial heatmaps saved by two_stage_inference_heatmaps.py.
Creates per-channel maximum-intensity projections and optional 3D scatter plots.
"""

from pathlib import Path
from typing import Tuple

import numpy as np


def _soft_argmax_2d(heatmap: np.ndarray) -> Tuple[float, float]:
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


def main():
    import argparse
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    parser = argparse.ArgumentParser(description="Visualize stacked axial heatmaps")
    parser.add_argument("--heatmaps", type=str, help="Path to axial_heatmaps.npz")
    parser.add_argument("--input_dir", type=str, help="Directory with case subfolders or .npz files")
    parser.add_argument("--output", type=str, required=True, help="Output directory for images")
    parser.add_argument("--no_scatter", action="store_true", help="Skip 3D scatter plots")
    parser.add_argument("--stacked", action="store_true", help="Create stacked 2D slice figure per case")
    parser.add_argument("--stack_stride", type=int, default=1, help="Use every Nth slice for stacked figure")
    parser.add_argument("--channel", type=int, default=0, help="Heatmap channel to stack in 3D")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.heatmaps and not args.input_dir:
        raise ValueError("Provide either --heatmaps or --input_dir")

    heatmap_paths = []
    if args.heatmaps:
        heatmap_paths.append(Path(args.heatmaps))
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            raise ValueError(f"--input_dir is not a directory: {input_dir}")
        heatmap_paths.extend(sorted(input_dir.rglob("axial_heatmaps.npz")))
        heatmap_paths.extend(sorted(p for p in input_dir.glob("*.npz") if p.name == "axial_heatmaps.npz"))

    if len(heatmap_paths) == 0:
        raise ValueError("No axial_heatmaps.npz files found.")

    for heatmap_path in heatmap_paths:
        data = np.load(heatmap_path, allow_pickle=True)
        heatmaps = data["heatmaps"]  # (num_slices, num_landmarks, H, W)
        z_indices = data["z_indices"].astype(int)

        if heatmaps.size == 0:
            continue

        case_name = heatmap_path.parent.name
        case_output_dir = output_dir / case_name
        case_output_dir.mkdir(parents=True, exist_ok=True)

        num_slices, num_landmarks, _, _ = heatmaps.shape

        # Per-channel max intensity projection over slices
        for ch in range(num_landmarks):
            mip = heatmaps[:, ch].max(axis=0)
            plt.figure(figsize=(6, 6))
            plt.imshow(mip, cmap="magma")
            plt.title(f"{case_name} ch{ch} MIP ({num_slices} slices)")
            plt.colorbar()
            out_path = case_output_dir / f"heatmap_mip_ch{ch}.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()

        if not args.no_scatter:
            # 3D scatter of soft-argmax points per slice (heatmap space)
            for ch in range(num_landmarks):
                xs, ys, zs = [], [], []
                for idx, z in enumerate(z_indices):
                    hm = heatmaps[idx, ch]
                    x, y = _soft_argmax_2d(hm)
                    if np.isnan(x) or np.isnan(y):
                        continue
                    xs.append(x)
                    ys.append(y)
                    zs.append(float(z))

                if len(xs) == 0:
                    continue

                fig = plt.figure(figsize=(7, 6))
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(xs, ys, zs, s=10, c=zs, cmap="viridis")
                ax.set_title(f"{case_name} ch{ch} soft-argmax (heatmap space)")
                ax.set_xlabel("x (heatmap)")
                ax.set_ylabel("y (heatmap)")
                ax.set_zlabel("slice z")
                out_path = case_output_dir / f"heatmap_scatter_ch{ch}.png"
                plt.tight_layout()
                plt.savefig(out_path, dpi=150)
                plt.close()

        if args.stacked:
            stride = max(1, int(args.stack_stride))
            ch = int(args.channel)
            if ch < 0 or ch >= num_landmarks:
                raise ValueError(f"--channel must be in [0, {num_landmarks - 1}]")

            slice_indices = list(range(0, num_slices, stride))
            if len(slice_indices) == 0:
                continue

            h, w = heatmaps.shape[2], heatmaps.shape[3]
            xs = np.arange(w)
            ys = np.arange(h)
            grid_x, grid_y = np.meshgrid(xs, ys)

            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=4, azim=20)
            vmin = float(np.min(heatmaps[:, ch]))
            vmax = float(np.max(heatmaps[:, ch]))
            vmax = 0.02
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.magma
            height_scale = 100.0  # adjust for how tall you want the peaks
            for si in slice_indices:
                z_val = float(z_indices[si]) if si < len(z_indices) else float(si)
                slice_img = heatmaps[si, ch]
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

            ax.set_title(f"{case_name} stacked slices ch{ch} (stride={stride})")
            ax.set_xlabel("x (heatmap)")
            ax.set_ylabel("y (heatmap)")
            ax.set_zlabel("slice z")
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array([])
            plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
            out_path = case_output_dir / f"heatmap_stacked_ch{ch}_stride{stride}.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()

    print(f"Saved visualizations to {output_dir}")


if __name__ == "__main__":
    main()
