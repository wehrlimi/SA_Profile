"""
Visualize axial heatmaps overlaid on the actual MRI slices.
Creates 2D overlay images showing heatmaps on top of the underlying anatomy.
"""

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from scipy.ndimage import zoom


def _normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range."""
    min_val, max_val = image.min(), image.max()
    if max_val > min_val:
        return (image - min_val) / (max_val - min_val)
    return image


def _resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image to target size using bilinear interpolation."""
    scale_factors = np.array(target_size) / np.array(image.shape)
    return zoom(image, scale_factors, order=1)


def _soft_argmax_2d(heatmap: np.ndarray) -> Tuple[float, float]:
    """Compute weighted centroid (soft argmax) in heatmap space."""
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


def _extract_and_transform_slice(
    volume: np.ndarray,
    z_idx: int,
    image_size: Tuple[int, int],
) -> np.ndarray:
    """
    Extract axial slice and apply the same transformations used during inference.
    This ensures the slice matches the heatmap coordinate system.
    """
    # Extract axial slice (same as in two_stage_inference_heatmaps.py)
    axial_slice = volume[:, :, z_idx]
    
    # Rotate 90 degrees counter-clockwise (same as inference)
    axial_slice = np.rot90(axial_slice, k=1)
    
    # Normalize
    axial_slice = _normalize_image(axial_slice)
    
    # Resize to match heatmap size
    axial_slice_resized = _resize_image(axial_slice, image_size)
    
    return axial_slice_resized


def _create_blue_tinted_image(img: np.ndarray) -> np.ndarray:
    """
    Create a blue-tinted version of the grayscale image.
    Returns an RGB image with shape (H, W, 3).
    """
    # Normalize to [0, 1]
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    # Create RGB with blue tint (dark blue background, lighter blue for anatomy)
    # Similar to the reference image style
    rgb = np.zeros((*img_norm.shape, 3), dtype=np.float32)
    rgb[:, :, 0] = img_norm * 0.3  # Red channel (low)
    rgb[:, :, 1] = img_norm * 0.35  # Green channel (low)
    rgb[:, :, 2] = img_norm * 0.7 + 0.15  # Blue channel (high, with base)
    
    return np.clip(rgb, 0, 1)


def _apply_hot_heatmap_overlay(base_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.8) -> np.ndarray:
    """
    Overlay a heatmap on a base RGB image using a red-yellow-white hot colormap.
    Only applies the heatmap color where heatmap values are significant.
    """
    import matplotlib.pyplot as plt
    
    # Normalize heatmap
    heatmap_norm = heatmap / (heatmap.max() + 1e-8)
    
    # Create hot colormap overlay (red -> yellow -> white)
    cmap = plt.cm.hot
    heatmap_colored = cmap(heatmap_norm)[:, :, :3]  # RGB, drop alpha
    
    # Blend based on heatmap intensity
    # Where heatmap is high, show heatmap colors; where low, show base image
    blend_factor = (heatmap_norm ** 0.7)[:, :, np.newaxis] * alpha  # Power for sharper transition
    result = base_rgb * (1 - blend_factor) + heatmap_colored * blend_factor
    
    return np.clip(result, 0, 1)


def main():
    import argparse
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import nibabel as nib

    parser = argparse.ArgumentParser(description="Visualize heatmaps overlaid on MRI slices")
    parser.add_argument("--heatmaps", type=str, help="Path to axial_heatmaps.npz")
    parser.add_argument("--input_volume", type=str, required=True, help="Path to input NIfTI volume")
    parser.add_argument("--input_dir", type=str, help="Directory with case subfolders containing .npz files")
    parser.add_argument("--output", type=str, required=True, help="Output directory for images")
    parser.add_argument("--channels", type=str, default=None, help="Comma-separated channels to visualize (default: all)")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth slice")
    parser.add_argument("--alpha", type=float, default=0.5, help="Heatmap overlay transparency (0-1)")
    parser.add_argument("--cmap", type=str, default="jet", help="Colormap for heatmaps")
    parser.add_argument("--show_keypoints", action="store_true", help="Show soft-argmax keypoint locations")
    parser.add_argument("--grid", action="store_true", help="Create a grid of all slices in one figure")
    parser.add_argument("--grid_cols", type=int, default=5, help="Number of columns in grid view")
    parser.add_argument("--summed", action="store_true", help="Sum all heatmap channels into one overlay")
    parser.add_argument("--dpi", type=int, default=150, help="Output image DPI")
    parser.add_argument("--style", type=str, default="overlay", choices=["overlay", "blue_hot"],
                        help="Visualization style: 'overlay' (heatmap on gray) or 'blue_hot' (blue tint + hot spots)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.heatmaps and not args.input_dir:
        raise ValueError("Provide either --heatmaps or --input_dir")

    # Collect heatmap paths
    heatmap_paths = []
    if args.heatmaps:
        heatmap_paths.append(Path(args.heatmaps))
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            raise ValueError(f"--input_dir is not a directory: {input_dir}")
        heatmap_paths.extend(sorted(input_dir.rglob("axial_heatmaps.npz")))

    if len(heatmap_paths) == 0:
        raise ValueError("No axial_heatmaps.npz files found.")

    # Load volume
    nii = nib.load(args.input_volume)
    volume = nii.get_fdata()
    print(f"Loaded volume: {args.input_volume} with shape {volume.shape}")

    # Define landmark colors
    landmark_colors = ["cyan", "lime", "orange", "magenta", "yellow", "white"]

    for heatmap_path in heatmap_paths:
        print(f"Processing: {heatmap_path}")
        data = np.load(heatmap_path, allow_pickle=True)
        heatmaps = data["heatmaps"]  # (num_slices, num_landmarks, H, W)
        z_indices = data["z_indices"].astype(int)
        image_size = tuple(data["image_size"].tolist()) if "image_size" in data else heatmaps.shape[2:4]

        if heatmaps.size == 0:
            print(f"  Skipping (no heatmaps)")
            continue

        num_slices, num_landmarks, h, w = heatmaps.shape
        print(f"  Heatmaps shape: {heatmaps.shape}, z_indices: {z_indices[0]} to {z_indices[-1]}")

        # Parse channels
        if args.channels is None:
            channels = list(range(num_landmarks))
        else:
            channels = [int(c.strip()) for c in args.channels.split(",") if c.strip() != ""]
        
        for ch in channels:
            if ch < 0 or ch >= num_landmarks:
                raise ValueError(f"Invalid channel {ch}; valid range is [0, {num_landmarks - 1}]")

        case_name = heatmap_path.parent.name
        case_output_dir = output_dir / case_name
        case_output_dir.mkdir(parents=True, exist_ok=True)

        # Select slices to visualize
        slice_indices = list(range(0, num_slices, args.stride))

        if args.grid:
            # Create grid visualization
            num_images = len(slice_indices)
            cols = min(args.grid_cols, num_images)
            rows = (num_images + cols - 1) // cols

            if args.summed:
                # Single grid with summed heatmaps
                fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
                if rows == 1 and cols == 1:
                    axes = np.array([[axes]])
                elif rows == 1:
                    axes = axes.reshape(1, -1)
                elif cols == 1:
                    axes = axes.reshape(-1, 1)

                for idx, si in enumerate(slice_indices):
                    row, col = idx // cols, idx % cols
                    ax = axes[row, col]

                    z = z_indices[si]
                    if z < 0 or z >= volume.shape[2]:
                        ax.axis("off")
                        continue

                    # Get transformed slice
                    img = _extract_and_transform_slice(volume, z, image_size)

                    # Sum heatmaps across selected channels
                    heatmap_sum = np.sum(heatmaps[si, channels], axis=0)

                    # Display based on style
                    if args.style == "blue_hot":
                        img_rgb = _create_blue_tinted_image(img)
                        result = _apply_hot_heatmap_overlay(img_rgb, heatmap_sum, alpha=args.alpha)
                        ax.imshow(result)
                    else:
                        heatmap_norm = heatmap_sum / (heatmap_sum.max() + 1e-8)
                        ax.imshow(img, cmap="gray")
                        ax.imshow(heatmap_norm, cmap=args.cmap, alpha=args.alpha * heatmap_norm)
                    ax.set_title(f"z={z}", fontsize=10)
                    ax.axis("off")

                    # Show keypoints if requested
                    if args.show_keypoints:
                        for ch_idx, ch in enumerate(channels):
                            x, y = _soft_argmax_2d(heatmaps[si, ch])
                            if not np.isnan(x) and not np.isnan(y):
                                color = landmark_colors[ch_idx % len(landmark_colors)]
                                ax.plot(y, x, "o", color=color, markersize=6, markeredgecolor="black", markeredgewidth=1)

                # Turn off unused axes
                for idx in range(num_images, rows * cols):
                    row, col = idx // cols, idx % cols
                    axes[row, col].axis("off")

                plt.suptitle(f"{case_name} - Summed heatmaps (ch: {channels})", fontsize=14)
                plt.tight_layout()
                out_path = case_output_dir / f"overlay_grid_summed.png"
                plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
                plt.close()
                print(f"  Saved: {out_path}")

            else:
                # Create grid for each channel
                for ch in channels:
                    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
                    if rows == 1 and cols == 1:
                        axes = np.array([[axes]])
                    elif rows == 1:
                        axes = axes.reshape(1, -1)
                    elif cols == 1:
                        axes = axes.reshape(-1, 1)

                    for idx, si in enumerate(slice_indices):
                        row, col = idx // cols, idx % cols
                        ax = axes[row, col]

                        z = z_indices[si]
                        if z < 0 or z >= volume.shape[2]:
                            ax.axis("off")
                            continue

                        # Get transformed slice
                        img = _extract_and_transform_slice(volume, z, image_size)

                        # Get heatmap for this channel
                        heatmap = heatmaps[si, ch]

                        # Display based on style
                        if args.style == "blue_hot":
                            img_rgb = _create_blue_tinted_image(img)
                            result = _apply_hot_heatmap_overlay(img_rgb, heatmap, alpha=args.alpha)
                            ax.imshow(result)
                        else:
                            heatmap_norm = heatmap / (heatmap.max() + 1e-8)
                            ax.imshow(img, cmap="gray")
                            ax.imshow(heatmap_norm, cmap=args.cmap, alpha=args.alpha * heatmap_norm)
                        ax.set_title(f"z={z}", fontsize=10)
                        ax.axis("off")

                        # Show keypoint if requested
                        if args.show_keypoints:
                            x, y = _soft_argmax_2d(heatmap)
                            if not np.isnan(x) and not np.isnan(y):
                                ax.plot(y, x, "o", color="cyan", markersize=8, markeredgecolor="black", markeredgewidth=1.5)

                    # Turn off unused axes
                    for idx in range(num_images, rows * cols):
                        row, col = idx // cols, idx % cols
                        axes[row, col].axis("off")

                    plt.suptitle(f"{case_name} - Channel {ch}", fontsize=14)
                    plt.tight_layout()
                    out_path = case_output_dir / f"overlay_grid_ch{ch}.png"
                    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
                    plt.close()
                    print(f"  Saved: {out_path}")

        else:
            # Create individual slice images
            for si in slice_indices:
                z = z_indices[si]
                if z < 0 or z >= volume.shape[2]:
                    continue

                # Get transformed slice
                img = _extract_and_transform_slice(volume, z, image_size)

                if args.summed:
                    # Sum all channels
                    heatmap_sum = np.sum(heatmaps[si, channels], axis=0)

                    fig, ax = plt.subplots(figsize=(8, 8))
                    
                    # Display based on style
                    if args.style == "blue_hot":
                        img_rgb = _create_blue_tinted_image(img)
                        result = _apply_hot_heatmap_overlay(img_rgb, heatmap_sum, alpha=args.alpha)
                        ax.imshow(result)
                    else:
                        heatmap_norm = heatmap_sum / (heatmap_sum.max() + 1e-8)
                        ax.imshow(img, cmap="gray")
                        ax.imshow(heatmap_norm, cmap=args.cmap, alpha=args.alpha * heatmap_norm)
                    
                    ax.set_title(f"{case_name} z={z} (summed ch: {channels})")
                    ax.axis("off")

                    if args.show_keypoints:
                        for ch_idx, ch in enumerate(channels):
                            x, y = _soft_argmax_2d(heatmaps[si, ch])
                            if not np.isnan(x) and not np.isnan(y):
                                color = landmark_colors[ch_idx % len(landmark_colors)]
                                ax.plot(y, x, "o", color=color, markersize=10, markeredgecolor="black", markeredgewidth=2)

                    plt.tight_layout()
                    out_path = case_output_dir / f"overlay_z{z:03d}_summed.png"
                    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
                    plt.close()

                else:
                    # Create image with subplots for each channel
                    ncols = min(len(channels) + 1, 4)  # +1 for original image
                    nrows = (len(channels) + 1 + ncols - 1) // ncols

                    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
                    if nrows == 1 and ncols == 1:
                        axes = np.array([[axes]])
                    elif nrows == 1:
                        axes = axes.reshape(1, -1)
                    elif ncols == 1:
                        axes = axes.reshape(-1, 1)

                    # First subplot: original image with all keypoints
                    ax = axes.flat[0]
                    if args.style == "blue_hot":
                        ax.imshow(_create_blue_tinted_image(img))
                    else:
                        ax.imshow(img, cmap="gray")
                    ax.set_title("MRI slice")
                    ax.axis("off")

                    if args.show_keypoints:
                        for ch_idx, ch in enumerate(channels):
                            x, y = _soft_argmax_2d(heatmaps[si, ch])
                            if not np.isnan(x) and not np.isnan(y):
                                color = landmark_colors[ch_idx % len(landmark_colors)]
                                ax.plot(y, x, "o", color=color, markersize=10, markeredgecolor="black", markeredgewidth=2, label=f"ch{ch}")
                        ax.legend(loc="upper right", fontsize=8)

                    # Remaining subplots: each channel overlay
                    for ch_idx, ch in enumerate(channels):
                        ax = axes.flat[ch_idx + 1]
                        heatmap = heatmaps[si, ch]

                        # Display based on style
                        if args.style == "blue_hot":
                            img_rgb = _create_blue_tinted_image(img)
                            result = _apply_hot_heatmap_overlay(img_rgb, heatmap, alpha=args.alpha)
                            ax.imshow(result)
                        else:
                            heatmap_norm = heatmap / (heatmap.max() + 1e-8)
                            ax.imshow(img, cmap="gray")
                            ax.imshow(heatmap_norm, cmap=args.cmap, alpha=args.alpha * heatmap_norm)
                        ax.set_title(f"Channel {ch}")
                        ax.axis("off")

                        if args.show_keypoints:
                            x, y = _soft_argmax_2d(heatmap)
                            if not np.isnan(x) and not np.isnan(y):
                                ax.plot(y, x, "o", color="cyan", markersize=10, markeredgecolor="black", markeredgewidth=2)

                    # Turn off unused axes
                    for idx in range(len(channels) + 1, nrows * ncols):
                        axes.flat[idx].axis("off")

                    plt.suptitle(f"{case_name} z={z}", fontsize=14)
                    plt.tight_layout()
                    out_path = case_output_dir / f"overlay_z{z:03d}.png"
                    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
                    plt.close()

            print(f"  Saved {len(slice_indices)} slice overlays to {case_output_dir}")

    print(f"\nDone! Output saved to {output_dir}")


if __name__ == "__main__":
    main()
