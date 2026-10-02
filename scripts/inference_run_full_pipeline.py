"""
End-to-end pipeline:
1) Run two-stage inference to get axial heatmaps.
2) Fit 3D lines from heatmaps.
3) Save raw and fitted Slicer markups + sulcus angle plot.
4) Save summed heatmap NIfTI aligned to Slicer visualization.

python scripts/run_full_pipeline.py ^
  --config config.yaml ^
  --sagittal_model .\tmp\checkpoints\sagittal_bounds\checkpoint_keypoint_model_10.pt ^
  --axial_model .\tmp\checkpoints\axial_landmarks\checkpoint_keypoint_model_9.pt ^
  --input_dir .\inference_data ^
  --output .\inference_output_all

"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib
import torch

# Add project root to path to enable imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.two_stage_inference_heatmaps import TwoStageInferenceHeatmaps  # noqa: E402
from scripts.fit_3d_lines_from_heatmaps import (  # noqa: E402
    _heatmap_to_voxel,
    _voxel_to_world,
    _write_slicer_markups,
)
from scripts.compute_sulcus_angles_spline_leo import (  # noqa: E402
    SimpleProbabilitySplineFitter,
    compute_angles_from_fitted_points,
)
from scripts.plot_sulcus_angles import _plot_case  # noqa: E402


def _apply_slicer_transform(affine: np.ndarray) -> np.ndarray:
    """Apply the same (-x, -y, z) world-space flip used for Slicer markups."""
    flip = np.diag([-1.0, -1.0, 1.0, 1.0])
    return flip @ affine


def _heatmap_to_voxel_affine(
    pre_rot_shape: Tuple[int, int],
    image_size: Tuple[int, int],
    transpose_mode: str = "swap_alt",
) -> np.ndarray:
    pre_rot_height = pre_rot_shape[0]
    original_size = np.array(np.rot90(np.zeros(pre_rot_shape), k=1).shape, dtype=np.float64)
    scale = original_size / np.array(image_size, dtype=np.float64)
    if transpose_mode == "swap":
        return np.array(
            [
                [0.0, -scale[1], 0.0, pre_rot_height - 1.0],
                [scale[0], 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    if transpose_mode == "swap_alt":
        return np.array(
            [
                [0.0, scale[1], 0.0, 0.0],
                [-scale[0], 0.0, 0.0, pre_rot_height - 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    return np.array(
        [
            [scale[0], 0.0, 0.0, 0.0],
            [0.0, -scale[1], 0.0, pre_rot_height - 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _build_volume(heatmaps: np.ndarray, z_indices: np.ndarray, output_depth: int) -> np.ndarray:
    num_slices, height, width = heatmaps.shape
    volume = np.zeros((height, width, output_depth), dtype=np.float32)
    for idx in range(num_slices):
        z = int(z_indices[idx])
        if 0 <= z < output_depth:
            volume[:, :, z] = heatmaps[idx]
    return volume


def _write_bounds_markups(points_world: List[List[float]], output_path: Path) -> None:
    slicer_points = [[-x, -y, z] for x, y, z in points_world]
    markup = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "coordinateUnits": "mm",
                "locked": True,
                "fixedNumberOfControlPoints": False,
                "labelFormat": "%N-%d",
                "lastUsedControlPointNumber": len(slicer_points),
                "controlPoints": [
                    {
                        "id": str(i + 1),
                        "label": f"SaggitalBounds-{i + 1}",
                        "description": "",
                        "associatedNodeID": "vtkMRMLScalarVolumeNode1",
                        "position": slicer_points[i],
                        "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                        "selected": True,
                        "locked": False,
                        "visibility": True,
                        "positionStatus": "defined",
                    }
                    for i in range(len(slicer_points))
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


def _fit_lines_spline(
    heatmaps: np.ndarray,
    z_indices: np.ndarray,
    volume: np.ndarray,
    affine: np.ndarray,
    image_size: Tuple[int, int],
    max_control: int,
    transpose_mode: str,
) -> Dict:
    """Fit splines through heatmap probability volumes using SimpleProbabilitySplineFitter.

    Returns raw/fitted world-space points grouped per slice, plus per-landmark
    fitted points in heatmap [col, row] format for angle computation.
    """
    num_slices, num_landmarks, h, w = heatmaps.shape
    channels = list(range(num_landmarks))

    # Store as per-slice lists so we can group 3 points per slice in order.
    raw_world_points: List[List[List[float]]] = [[] for _ in range(num_slices)]
    fitted_world_points: List[List[List[float]]] = [[] for _ in range(num_slices)]
    # Per-landmark fitted points in [col, row] heatmap space (for angle computation)
    per_landmark_points: Dict[str, np.ndarray] = {}

    for ch in channels:
        ch_heatmaps = heatmaps[:, ch, :, :]  # (num_slices, H, W)

        # --- raw argmax points ---
        flat = ch_heatmaps.reshape(num_slices, -1)
        max_idx = flat.argmax(axis=1)
        raw_rows = (max_idx // w).astype(np.float64)
        raw_cols = (max_idx % w).astype(np.float64)

        # --- spline-fitted points ---
        fitter = SimpleProbabilitySplineFitter(ch_heatmaps)
        res = fitter.fit(min_control=3, max_control=max_control, tol=0.01)
        # res["best_points"] is (num_slices, 2) in [col, row] format (Leo convention)
        fitted_pts = res["best_points"]  # (num_slices, 2) -> [col, row]
        per_landmark_points[str(ch)] = fitted_pts

        for idx, z in enumerate(z_indices):
            z = int(z)
            if z < 0 or z >= volume.shape[2]:
                continue
            pre_rot_shape = volume[:, :, z].shape

            # Raw argmax: already in (row, col) order for _heatmap_to_voxel
            i_raw, j_raw = _heatmap_to_voxel(
                (float(raw_rows[idx]), float(raw_cols[idx])),
                pre_rot_shape,
                image_size,
                transpose_mode=transpose_mode,
            )
            raw_voxel = np.array([i_raw, j_raw, float(z)], dtype=np.float64)
            raw_world = _voxel_to_world(raw_voxel.reshape(1, 3), affine)[0]
            raw_world_points[idx].append(raw_world.tolist())

            # Spline fitted: convert [col, row] -> (row, col) for _heatmap_to_voxel
            fit_col, fit_row = fitted_pts[idx]
            i_fit, j_fit = _heatmap_to_voxel(
                (float(fit_row), float(fit_col)),
                pre_rot_shape,
                image_size,
                transpose_mode=transpose_mode,
            )
            fit_voxel = np.array([i_fit, j_fit, float(z)], dtype=np.float64)
            fit_world = _voxel_to_world(fit_voxel.reshape(1, 3), affine)[0]
            fitted_world_points[idx].append(fit_world.tolist())

    return {
        "raw_world": raw_world_points,
        "fitted_world": fitted_world_points,
        "per_landmark_points": per_landmark_points,
        "channels": channels,
    }


def main() -> None:
    import argparse
    import json
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Run full knee pipeline on inference_data")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--sagittal_model", type=str, required=True, help="Path to sagittal model checkpoint")
    parser.add_argument("--axial_model", type=str, required=True, help="Path to axial model checkpoint")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing .nii/.nii.gz files")
    parser.add_argument("--output", type=str, default="inference_output_full", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_control", type=int, default=10, help="Max spline control points (default: 10)")
    parser.add_argument("--transpose_mode", type=str, default="swap_alt", choices=["swap_alt", "swap", "none"])
    parser.add_argument(
        "--slicer_transform",
        action="store_true",
        help="Apply (-x, -y, z) flip to heatmap NIfTI affine (leave off for default alignment)",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    image_size = cfg.program.image_size
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    elif isinstance(image_size, (list, tuple)):
        image_size = tuple(image_size)
    else:
        image_size = (256, 256)

    pipeline = TwoStageInferenceHeatmaps(
        sagittal_model_path=args.sagittal_model,
        axial_model_path=args.axial_model,
        device=args.device,
        image_size=image_size,
        z_margin=cfg.inference.get("z_margin", 5),
    )

    input_path_obj = Path(args.input_dir)
    if input_path_obj.is_file():
        input_paths = [input_path_obj]
    elif input_path_obj.is_dir():
        input_paths = sorted(input_path_obj.glob("*.nii.gz")) + sorted(input_path_obj.glob("*.nii"))
        if not input_paths:
            raise ValueError(f"No NIfTI files found in {input_path_obj}")
    else:
        raise ValueError(f"--input_dir path does not exist: {input_path_obj}")

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    for input_path in input_paths:
        case_id = input_path.name
        if case_id.endswith(".nii.gz"):
            case_id = case_id[:-7]
        elif case_id.endswith(".nii"):
            case_id = case_id[:-4]

        case_output_dir = output_root / case_id
        case_output_dir.mkdir(parents=True, exist_ok=True)

        nii = nib.load(str(input_path))
        volume = nii.get_fdata()
        nii_affine = nii.affine

        print(f"Running inference on {input_path}...")
        result = pipeline.predict_case(volume, nii_affine=nii_affine)

        # Save heatmaps npz and summed NIfTI
        heatmaps = result["axial_heatmaps"]["heatmaps"]
        z_indices = result["axial_heatmaps"]["z_indices"]
        npz_path = case_output_dir / "axial_heatmaps.npz"
        np.savez_compressed(
            npz_path,
            heatmaps=heatmaps,
            z_indices=z_indices,
            bounds=np.array([result["bounds"]["z_min"], result["bounds"]["z_max"]], dtype=np.int32),
            image_size=np.array(image_size, dtype=np.int32),
            affine=nii_affine,
        )

        if heatmaps.size == 0:
            print(f"Skipped case (no heatmaps): {case_id}")
            continue

        summed = np.sum(heatmaps, axis=1)
        output_depth = int(z_indices.max()) + 1 if z_indices.size > 0 else summed.shape[0]
        summed_volume = _build_volume(summed, z_indices, output_depth)
        heatmap_to_voxel = _heatmap_to_voxel_affine(
            volume[:, :, 0].shape,
            image_size,
            transpose_mode=args.transpose_mode,
        )
        heatmap_affine = nii_affine @ heatmap_to_voxel
        if args.slicer_transform:
            heatmap_affine = _apply_slicer_transform(heatmap_affine)
        heatmap_nii_path = case_output_dir / f"{case_id}_axial_heatmaps_sum_slicer.nii.gz"
        nib.save(nib.Nifti1Image(summed_volume, heatmap_affine), str(heatmap_nii_path))

        # Save sagittal bounds markups
        bounds_points_world = result["bounds"]["points_world"]
        bounds_path = case_output_dir / f"SaggitalBounds_{case_id}.mrk.json"
        _write_bounds_markups(bounds_points_world, bounds_path)

        # Fit lines + raw points
        points = _fit_lines_spline(
            heatmaps=heatmaps,
            z_indices=z_indices,
            volume=volume,
            affine=nii_affine,
            image_size=image_size,
            max_control=args.max_control,
            transpose_mode=args.transpose_mode,
        )

        # Save raw AxialLabels
        raw_path = case_output_dir / f"AxialLabels_{case_id}.mrk.json"
        raw_slicer_points = []
        for slice_points in points["raw_world"]:
            if not slice_points:
                continue
            for x, y, z in slice_points:
                raw_slicer_points.append([-x, -y, z])
        _write_slicer_markups(raw_slicer_points, raw_path)

        # Save fitted AxialLabels
        fitted_path = case_output_dir / f"AxialLabels_fitted_{case_id}.mrk.json"
        fitted_slicer_points = []
        for slice_points in points["fitted_world"]:
            if not slice_points:
                continue
            for x, y, z in slice_points:
                fitted_slicer_points.append([-x, -y, z])
        _write_slicer_markups(fitted_slicer_points, fitted_path)

        # Extract pixel spacing from NIfTI for physical angle computation
        pixdim = nii.header.get_zooms()
        pixel_spacing = (float(pixdim[1]), float(pixdim[0]))  # (row_spacing, col_spacing)

        # Compute and save sulcus angle curve
        angles = compute_angles_from_fitted_points(
            points["per_landmark_points"],
            points["channels"],
            pixel_spacing=pixel_spacing,
        )
        angles_json_path = case_output_dir / f"sulcus_angles_{case_id}.json"
        angles_data = {
            "case_id": case_id,
            "angles": angles,
            "num_slices": len(angles),
            "pixel_spacing": list(pixel_spacing),
            "fitted_points": {
                k: v.tolist() for k, v in points["per_landmark_points"].items()
            },
            "channels": points["channels"],
        }
        with open(angles_json_path, "w") as f:
            json.dump(angles_data, f, indent=2)

        # Plot sulcus angles from fitted markups
        sulcus_plot_path = case_output_dir / f"sulcus_angle_{case_id}.png"
        _plot_case(case_id, fitted_path, sulcus_plot_path)
        print(f"Saved outputs for {case_id} -> {case_output_dir}")


if __name__ == "__main__":
    main()
