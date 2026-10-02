"""
Create 3D volume rendering from axial heatmaps for paper figures.
Outputs:
  1. NIfTI volume with proper spatial transform (for 3D Slicer viewing)
  2. 3D rendered figure (PNG) for paper inclusion
"""

from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
from scipy.ndimage import zoom


def _heatmap_to_voxel_affine(
    pre_rot_shape: Tuple[int, int],
    image_size: Tuple[int, int],
) -> np.ndarray:
    """
    Affine that maps heatmap voxel coords (row, col, z) to original voxel coords (i, j, k).
    Mirrors the inverse rotation/resize logic used for keypoints.
    Uses 'swap_alt' mode (default behavior matching inference).
    """
    pre_rot_height = pre_rot_shape[0]
    original_size = np.array(np.rot90(np.zeros(pre_rot_shape), k=1).shape, dtype=np.float64)
    scale = original_size / np.array(image_size, dtype=np.float64)
    
    # swap_alt mode (default)
    return np.array(
        [
            [0.0, scale[1], 0.0, 0.0],
            [-scale[0], 0.0, 0.0, pre_rot_height - 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _build_volume(
    heatmaps: np.ndarray, 
    z_indices: np.ndarray, 
    output_depth: int,
    interpolate: bool = False,
) -> np.ndarray:
    """
    Build a 3D volume from 2D heatmap slices.
    
    Args:
        heatmaps: (num_slices, H, W) array of 2D heatmaps
        z_indices: (num_slices,) array of z-indices
        output_depth: Total depth of output volume
        interpolate: If True, interpolate between slices for smoother rendering
    
    Returns:
        (H, W, output_depth) volume
    """
    num_slices, height, width = heatmaps.shape
    volume = np.zeros((height, width, output_depth), dtype=np.float32)
    
    for idx in range(num_slices):
        z = int(z_indices[idx])
        if 0 <= z < output_depth:
            volume[:, :, z] = heatmaps[idx]
    
    if interpolate and num_slices > 1:
        # Simple linear interpolation between populated slices
        z_min, z_max = int(z_indices.min()), int(z_indices.max())
        z_set = set(z_indices.astype(int))
        
        for z in range(z_min, z_max + 1):
            if z not in z_set:
                # Find nearest populated slices
                z_below = max([zz for zz in z_set if zz < z], default=None)
                z_above = min([zz for zz in z_set if zz > z], default=None)
                
                if z_below is not None and z_above is not None:
                    # Linear interpolation
                    t = (z - z_below) / (z_above - z_below)
                    volume[:, :, z] = (1 - t) * volume[:, :, z_below] + t * volume[:, :, z_above]
    
    return volume


def create_volume_rendering(
    volume: np.ndarray,
    output_path: Path,
    threshold: float = 0.1,
    cmap: str = "hot",
    elev: float = 20,
    azim: float = 135,
    figsize: Tuple[int, int] = (10, 10),
    dpi: int = 200,
    title: Optional[str] = None,
    alpha_scale: float = 1.0,
    show_axes: bool = True,
    bgcolor: str = "black",
    style: str = "hot",  # "hot" or "blue_hot"
) -> None:
    """
    Create a 3D volume rendering using matplotlib scatter plot of thresholded voxels.
    Z-axis (slice) is vertical.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.colors import LinearSegmentedColormap
    
    # Normalize volume
    vol_norm = volume / (volume.max() + 1e-8)
    
    # Get coordinates above threshold
    coords = np.argwhere(vol_norm > threshold)
    if len(coords) == 0:
        print(f"  Warning: No voxels above threshold {threshold}")
        return
    
    values = vol_norm[coords[:, 0], coords[:, 1], coords[:, 2]]
    
    # Create figure with appropriate background
    if style == "blue_hot":
        bgcolor = "#0a0a2e"  # Dark blue background
    
    fig = plt.figure(figsize=figsize, facecolor=bgcolor)
    ax = fig.add_subplot(111, projection="3d", facecolor=bgcolor)
    
    # Get colormap based on style
    if style == "blue_hot":
        # Custom hot colormap: dark red -> red -> orange -> yellow -> white
        colors_list = [
            (0.3, 0.0, 0.0),   # Dark red
            (0.8, 0.0, 0.0),   # Red
            (1.0, 0.3, 0.0),   # Orange-red
            (1.0, 0.6, 0.0),   # Orange
            (1.0, 0.9, 0.2),   # Yellow
            (1.0, 1.0, 0.8),   # Light yellow/white
        ]
        cmap_obj = LinearSegmentedColormap.from_list("blue_hot", colors_list)
    else:
        cmap_obj = plt.cm.get_cmap(cmap)
    
    colors = cmap_obj(values)
    colors[:, 3] = values * alpha_scale  # Alpha based on intensity
    
    # Scatter plot - Z (slice) is now the VERTICAL axis
    scatter = ax.scatter(
        coords[:, 1],  # y -> x axis in plot (horizontal)
        coords[:, 0],  # x -> y axis in plot (horizontal depth)
        coords[:, 2],  # z (slice) -> z axis in plot (VERTICAL)
        c=colors,
        s=3,
        alpha=values * alpha_scale,
    )
    
    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)
    
    # Styling - minimal: only z-axis label, no grid
    label_color = "white" if bgcolor in ["black", "#0a0a2e"] else "black"
    if show_axes:
        # Hide x and y axes
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("Slice (z)", color=label_color, fontsize=12, labelpad=10)
        
        # Hide ticks on x and y
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis='z', colors=label_color)
        
        # Hide panes and grid
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor(label_color)
        ax.zaxis.pane.set_alpha(0.1)
        ax.grid(False)
        
        # Only show z-axis line
        ax.xaxis.line.set_color('none')
        ax.yaxis.line.set_color('none')
        ax.zaxis.line.set_color(label_color)
        ax.zaxis.line.set_linewidth(1.5)
    else:
        ax.set_axis_off()
    
    if title:
        ax.set_title(title, color=label_color, fontsize=14, pad=20)
    
    # Set equal aspect ratio
    max_range = max(volume.shape) / 2
    mid_x = volume.shape[1] / 2
    mid_y = volume.shape[0] / 2
    mid_z = volume.shape[2] / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, facecolor=bgcolor, bbox_inches="tight")
    plt.close()


def create_isosurface_rendering(
    volume: np.ndarray,
    output_path: Path,
    threshold: float = 0.3,
    colors: Optional[List[str]] = None,
    elev: float = 20,
    azim: float = 45,
    figsize: Tuple[int, int] = (10, 10),
    dpi: int = 200,
    title: Optional[str] = None,
) -> None:
    """
    Create isosurface rendering using marching cubes (requires scikit-image).
    """
    try:
        from skimage import measure
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("  Warning: scikit-image not installed, skipping isosurface rendering")
        return
    
    # Normalize
    vol_norm = volume / (volume.max() + 1e-8)
    
    # Create figure
    fig = plt.figure(figsize=figsize, facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")
    
    # Extract isosurface using marching cubes
    try:
        verts, faces, normals, values = measure.marching_cubes(vol_norm, level=threshold)
    except ValueError:
        print(f"  Warning: Could not extract isosurface at threshold {threshold}")
        return
    
    # Create mesh
    mesh = Poly3DCollection(verts[faces], alpha=0.7)
    mesh.set_facecolor([1, 0.5, 0.1, 0.8])  # Orange color
    mesh.set_edgecolor([1, 0.7, 0.3, 0.3])
    ax.add_collection3d(mesh)
    
    # Set axis limits
    ax.set_xlim(0, volume.shape[0])
    ax.set_ylim(0, volume.shape[1])
    ax.set_zlim(0, volume.shape[2])
    
    ax.view_init(elev=elev, azim=azim)
    
    ax.set_xlabel("X", color="white")
    ax.set_ylabel("Y", color="white")
    ax.set_zlabel("Z (slice)", color="white")
    ax.tick_params(colors="white")
    
    if title:
        ax.set_title(title, color="white", fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, facecolor="black", bbox_inches="tight")
    plt.close()


def create_multi_view_rendering(
    volume: np.ndarray,
    output_path: Path,
    threshold: float = 0.1,
    cmap: str = "hot",
    dpi: int = 200,
    title: Optional[str] = None,
    style: str = "hot",  # "hot" or "blue_hot"
) -> None:
    """
    Create a figure with multiple viewing angles for the paper.
    Z-axis (slice) is vertical.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    # Background color based on style
    if style == "blue_hot":
        bgcolor = "#0a0a2e"
    else:
        bgcolor = "black"
    
    fig = plt.figure(figsize=(16, 5), facecolor=bgcolor)
    
    views = [
        (20, 135, "Oblique"),
        (0, 90, "Front"),
        (0, 180, "Side"),
        (90, 0, "Top"),
    ]
    
    vol_norm = volume / (volume.max() + 1e-8)
    coords = np.argwhere(vol_norm > threshold)
    
    if len(coords) == 0:
        print(f"  Warning: No voxels above threshold {threshold}")
        return
    
    values = vol_norm[coords[:, 0], coords[:, 1], coords[:, 2]]
    
    # Get colormap based on style
    if style == "blue_hot":
        colors_list = [
            (0.3, 0.0, 0.0),
            (0.8, 0.0, 0.0),
            (1.0, 0.3, 0.0),
            (1.0, 0.6, 0.0),
            (1.0, 0.9, 0.2),
            (1.0, 1.0, 0.8),
        ]
        cmap_obj = LinearSegmentedColormap.from_list("blue_hot", colors_list)
    else:
        cmap_obj = plt.cm.get_cmap(cmap)
    
    colors = cmap_obj(values)
    
    for idx, (elev, azim, view_name) in enumerate(views):
        ax = fig.add_subplot(1, 4, idx + 1, projection="3d", facecolor=bgcolor)
        
        # Z (slice) is now vertical
        ax.scatter(
            coords[:, 1],  # y -> x
            coords[:, 0],  # x -> y
            coords[:, 2],  # z (slice) -> z (vertical)
            c=colors,
            s=1,
            alpha=values * 0.8,
        )
        
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(view_name, color="white", fontsize=12)
        ax.set_axis_off()
        
        # Set equal aspect - Z (slice) is now vertical
        max_range = max(volume.shape) / 2
        mid_x = volume.shape[1] / 2
        mid_y = volume.shape[0] / 2
        mid_z = volume.shape[2] / 2
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if title:
        fig.suptitle(title, color="white", fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, facecolor=bgcolor, bbox_inches="tight")
    plt.close()


def main():
    import argparse
    import nibabel as nib

    parser = argparse.ArgumentParser(description="Create 3D volume rendering from heatmaps")
    parser.add_argument("--heatmaps", type=str, required=True, help="Path to axial_heatmaps.npz")
    parser.add_argument("--input_volume", type=str, required=True, help="Path to original NIfTI volume (for affine)")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--channels", type=str, default=None, help="Comma-separated channels (default: all, summed)")
    parser.add_argument("--threshold", type=float, default=0.1, help="Threshold for volume rendering (0-1)")
    parser.add_argument("--cmap", type=str, default="hot", help="Colormap for rendering")
    parser.add_argument("--elev", type=float, default=20, help="Elevation angle")
    parser.add_argument("--azim", type=float, default=135, help="Azimuth angle (default: 135 for good viewing)")
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI")
    parser.add_argument("--no_nifti", action="store_true", help="Skip NIfTI output")
    parser.add_argument("--no_render", action="store_true", help="Skip PNG rendering")
    parser.add_argument("--multi_view", action="store_true", help="Create multi-view rendering")
    parser.add_argument("--isosurface", action="store_true", help="Create isosurface rendering")
    parser.add_argument("--interpolate", action="store_true", help="Interpolate between slices")
    parser.add_argument("--per_channel", action="store_true", help="Create separate volumes per channel")
    parser.add_argument("--style", type=str, default="blue_hot", choices=["hot", "blue_hot"],
                        help="Color style: 'hot' (standard) or 'blue_hot' (blue background + hot spots)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load heatmaps
    print(f"Loading heatmaps: {args.heatmaps}")
    data = np.load(args.heatmaps, allow_pickle=True)
    heatmaps = data["heatmaps"]  # (num_slices, num_landmarks, H, W)
    z_indices = data["z_indices"].astype(int)
    image_size = tuple(data["image_size"].tolist()) if "image_size" in data else heatmaps.shape[2:4]
    
    if heatmaps.size == 0:
        raise ValueError("No heatmaps found in the provided .npz file.")
    
    num_slices, num_landmarks, h, w = heatmaps.shape
    print(f"  Shape: {heatmaps.shape}, z_indices: {z_indices[0]} to {z_indices[-1]}")

    # Parse channels
    if args.channels is None:
        channels = list(range(num_landmarks))
    else:
        channels = [int(c.strip()) for c in args.channels.split(",") if c.strip() != ""]

    # Load original NIfTI for affine
    print(f"Loading volume: {args.input_volume}")
    nii = nib.load(args.input_volume)
    nii_affine = nii.affine
    volume_shape = nii.shape
    print(f"  Shape: {volume_shape}")

    # Compute the heatmap-to-world affine
    pre_rot_shape = (volume_shape[0], volume_shape[1])  # Shape of axial slice before rotation
    heatmap_to_voxel = _heatmap_to_voxel_affine(pre_rot_shape, image_size)
    heatmap_to_world = nii_affine @ heatmap_to_voxel

    # Build output depth
    output_depth = int(z_indices.max()) + 1 if z_indices.size > 0 else volume_shape[2]

    case_name = Path(args.heatmaps).parent.name

    if args.per_channel:
        # Create separate volumes per channel
        for ch in channels:
            ch_heatmaps = heatmaps[:, ch]  # (num_slices, H, W)
            ch_volume = _build_volume(ch_heatmaps, z_indices, output_depth, interpolate=args.interpolate)
            
            if not args.no_nifti:
                nii_path = output_dir / f"{case_name}_heatmap_ch{ch}.nii.gz"
                nib.save(nib.Nifti1Image(ch_volume, heatmap_to_world), str(nii_path))
                print(f"  Saved NIfTI: {nii_path}")
            
            if not args.no_render:
                render_path = output_dir / f"{case_name}_render_ch{ch}.png"
                create_volume_rendering(
                    ch_volume, render_path,
                    threshold=args.threshold,
                    cmap=args.cmap,
                    elev=args.elev,
                    azim=args.azim,
                    dpi=args.dpi,
                    title=f"Channel {ch}",
                    style=args.style,
                )
                print(f"  Saved render: {render_path}")
    
    # Create summed volume
    summed_heatmaps = np.sum(heatmaps[:, channels], axis=1)  # (num_slices, H, W)
    summed_volume = _build_volume(summed_heatmaps, z_indices, output_depth, interpolate=args.interpolate)
    
    if not args.no_nifti:
        nii_path = output_dir / f"{case_name}_heatmap_summed.nii.gz"
        nib.save(nib.Nifti1Image(summed_volume, heatmap_to_world), str(nii_path))
        print(f"Saved NIfTI (summed): {nii_path}")
    
    if not args.no_render:
        render_path = output_dir / f"{case_name}_render_summed.png"
        create_volume_rendering(
            summed_volume, render_path,
            threshold=args.threshold,
            cmap=args.cmap,
            elev=args.elev,
            azim=args.azim,
            dpi=args.dpi,
            title=f"Heatmap Volume (channels: {channels})",
            style=args.style,
        )
        print(f"Saved render (summed): {render_path}")
    
    if args.multi_view:
        multi_path = output_dir / f"{case_name}_render_multiview.png"
        create_multi_view_rendering(
            summed_volume, multi_path,
            threshold=args.threshold,
            cmap=args.cmap,
            dpi=args.dpi,
            title=case_name,
            style=args.style,
        )
        print(f"Saved multi-view render: {multi_path}")
    
    if args.isosurface:
        iso_path = output_dir / f"{case_name}_render_isosurface.png"
        create_isosurface_rendering(
            summed_volume, iso_path,
            threshold=args.threshold * 2,  # Higher threshold for isosurface
            elev=args.elev,
            azim=args.azim,
            dpi=args.dpi,
            title=f"Isosurface (threshold={args.threshold * 2:.2f})",
        )
        print(f"Saved isosurface render: {iso_path}")

    print(f"\nDone! Output saved to {output_dir}")


if __name__ == "__main__":
    main()
