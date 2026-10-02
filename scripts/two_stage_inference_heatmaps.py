"""
Two-stage inference pipeline that returns axial heatmaps instead of argmax points.
Stage 1: sagittal bounds prediction
Stage 2: axial heatmap prediction within bounded region
"""

import sys
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
from scipy.ndimage import zoom
import nibabel as nib

# Add project root to path to enable imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TwoStageInferenceHeatmaps:
    """
    Two-stage inference pipeline for knee landmark detection.
    Returns axial heatmaps for each slice in the bounded region.
    """

    def __init__(
        self,
        sagittal_model_path: str,
        axial_model_path: str,
        device: str = "cuda",
        image_size: Tuple[int, int] = (256, 256),
        z_margin: int = 5,
    ):
        self.device = torch.device(device)
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
        self.z_margin = z_margin

        from models import load_model
        self.sagittal_model = load_model(sagittal_model_path, self.device)
        self.axial_model = load_model(axial_model_path, self.device)

        self.sagittal_model.eval()
        self.axial_model.eval()

    def predict_case(self, volume: np.ndarray, nii_affine: np.ndarray = None) -> Dict:
        bounds = self._predict_bounds(volume, nii_affine)
        heatmaps = self._predict_axial_heatmaps(volume, bounds["z_min"], bounds["z_max"])
        return {
            "bounds": bounds,
            "axial_heatmaps": heatmaps,
        }

    def _predict_bounds(self, volume: np.ndarray, nii_affine: np.ndarray = None) -> Dict:
        sagittal_axis = 0
        sagittal_slice_idx = int(np.round((volume.shape[sagittal_axis] - 1) / 2.0))
        sagittal_slice_idx = np.clip(sagittal_slice_idx, 0, volume.shape[sagittal_axis] - 1)

        sagittal_slice = volume[sagittal_slice_idx, :, :]
        pre_rot_shape = sagittal_slice.shape

        sagittal_slice = np.rot90(sagittal_slice, k=1)
        original_size = np.array(sagittal_slice.shape)

        sagittal_slice = self._normalize_image(sagittal_slice)
        sagittal_slice_resized = self._resize_image(sagittal_slice, self.image_size)

        image_tensor = torch.from_numpy(sagittal_slice_resized).unsqueeze(0).unsqueeze(0).float()
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            result = self.sagittal_model.predict(image_tensor)

        keypoints_resized = result["keypoints"][0]

        scale = original_size / np.array(self.image_size)
        pre_rot_height, _ = pre_rot_shape

        keypoints_rotated_scaled = np.array(
            [[kp[0] / scale[1], kp[1] / scale[0]] for kp in keypoints_resized]
        )

        keypoints_unrot = np.array(
            [[kp[0], pre_rot_height - 1 - kp[1]] for kp in keypoints_rotated_scaled]
        )

        heatmaps = result["heatmaps"][0]
        confidence = float(heatmaps.max(dim=-1)[0].max(dim=-1)[0].mean())

        points_world = []
        world_z_coords = []
        if nii_affine is not None:
            for kp in keypoints_unrot:
                voxel_coords = np.array([sagittal_slice_idx, kp[0], kp[1], 1.0])
                world_coords = nii_affine @ voxel_coords
                world_coords_3d = world_coords[:3]
                points_world.append(world_coords_3d.tolist())
                world_z_coords.append(world_coords_3d[2])
        else:
            points_world = [[0, 0, 0], [0, 0, 0]]
            world_z_coords = [0, 0]

        if nii_affine is not None and len(world_z_coords) > 0:
            k_indices = np.arange(volume.shape[2])
            center_i, center_j = volume.shape[0] // 2, volume.shape[1] // 2
            world_z_at_k = (
                nii_affine[2, 0] * center_i
                + nii_affine[2, 1] * center_j
                + nii_affine[2, 2] * k_indices
                + nii_affine[2, 3]
            )
            world_z_min = np.min(world_z_coords)
            world_z_max = np.max(world_z_coords)
            z_idx_min = int(np.argmin(np.abs(world_z_at_k - world_z_min)))
            z_idx_max = int(np.argmin(np.abs(world_z_at_k - world_z_max)))
            z_min = int(np.min([z_idx_min, z_idx_max])) - self.z_margin
            z_max = int(np.max([z_idx_min, z_idx_max])) + self.z_margin
        else:
            z_coords = keypoints_unrot[:, 1]
            z_min = int(np.min(z_coords)) - self.z_margin
            z_max = int(np.max(z_coords)) + self.z_margin

        z_min = max(0, z_min)
        z_max = min(volume.shape[2] - 1, z_max)

        return {
            "z_min": z_min,
            "z_max": z_max,
            "points": keypoints_unrot.tolist(),
            "points_world": points_world,
            "confidence": confidence,
        }

    def _predict_axial_heatmaps(
        self,
        volume: np.ndarray,
        z_min: int,
        z_max: int,
        stride: int = 1,
    ) -> Dict:
        heatmaps_list = []
        z_indices = []

        for z in range(z_min, z_max + 1, stride):
            axial_slice = volume[:, :, z]

            axial_slice = np.rot90(axial_slice, k=1)
            axial_slice = self._normalize_image(axial_slice)
            axial_slice_resized = self._resize_image(axial_slice, self.image_size)

            image_tensor = torch.from_numpy(axial_slice_resized).unsqueeze(0).unsqueeze(0).float()
            image_tensor = image_tensor.to(self.device)

            with torch.no_grad():
                result = self.axial_model.predict(image_tensor)

            heatmaps = result["heatmaps"][0].detach().cpu().numpy()
            heatmaps_list.append(heatmaps)
            z_indices.append(z)

        heatmaps_arr = np.stack(heatmaps_list, axis=0) if heatmaps_list else np.zeros((0, 0, 0, 0))
        return {
            "heatmaps": heatmaps_arr,
            "z_indices": np.array(z_indices, dtype=np.int32),
        }

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        min_val, max_val = image.min(), image.max()
        if max_val > min_val:
            return (image - min_val) / (max_val - min_val)
        return image

    def _resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        scale_factors = np.array(target_size) / np.array(image.shape)
        return zoom(image, scale_factors, order=1)


def _apply_slicer_transform(affine: np.ndarray) -> np.ndarray:
    """Apply the same (-x, -y, z) world-space flip used for Slicer markups."""
    flip = np.diag([-1.0, -1.0, 1.0, 1.0])
    return flip @ affine


def _build_volume(heatmaps: np.ndarray, z_indices: np.ndarray, output_depth: int) -> np.ndarray:
    num_slices, height, width = heatmaps.shape
    volume = np.zeros((height, width, output_depth), dtype=np.float32)
    for idx in range(num_slices):
        z = int(z_indices[idx])
        if 0 <= z < output_depth:
            volume[:, :, z] = heatmaps[idx]
    return volume


def _heatmap_to_voxel_affine(
    pre_rot_shape: Tuple[int, int],
    image_size: Tuple[int, int],
    transpose_mode: str = "swap_alt",
) -> np.ndarray:
    """
    Affine that maps heatmap voxel coords (row, col, z) to original voxel coords (i, j, k).
    Mirrors the inverse rotation/resize logic used for keypoints.
    """
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


def main():
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Two-stage knee inference (axial heatmaps)")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--sagittal_model", type=str, required=True, help="Path to sagittal model checkpoint")
    parser.add_argument("--axial_model", type=str, required=True, help="Path to axial model checkpoint")
    parser.add_argument("--input_volume", type=str, help="Path to input NIfTI volume")
    parser.add_argument("--input_dir", type=str, help="Path to directory containing .nii.gz files")
    parser.add_argument("--output", type=str, default="inference_output", help="Output directory for heatmap .npz files")
    parser.add_argument(
        "--slicer_transform",
        action="store_true",
        help="Apply (-x, -y, z) world-space flip to match Slicer LPS markups",
    )
    parser.add_argument(
        "--transpose_xy",
        action="store_true",
        help="Swap in-plane axes across the main diagonal (non-default)",
    )
    parser.add_argument(
        "--transpose_xy_alt",
        action="store_true",
        help="Swap in-plane axes along the opposite diagonal (default behavior)",
    )
    args = parser.parse_args()
    if args.transpose_xy and args.transpose_xy_alt:
        raise ValueError("Use only one of --transpose_xy or --transpose_xy_alt.")

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
        device="cuda" if torch.cuda.is_available() else "cpu",
        image_size=image_size,
        z_margin=cfg.inference.get("z_margin", 5),
    )

    if not args.input_volume and not args.input_dir:
        raise ValueError("Provide either --input_volume or --input_dir")

    input_paths = []
    if args.input_volume:
        input_paths.append(Path(args.input_volume))
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            raise ValueError(f"--input_dir is not a directory: {input_dir}")
        input_paths.extend(sorted(input_dir.glob("*.nii.gz")))
        input_paths.extend(sorted(input_dir.glob("*.nii")))

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

        out_path = case_output_dir / "axial_heatmaps.npz"
        np.savez_compressed(
            out_path,
            heatmaps=result["axial_heatmaps"]["heatmaps"],
            z_indices=result["axial_heatmaps"]["z_indices"],
            bounds=np.array([result["bounds"]["z_min"], result["bounds"]["z_max"]], dtype=np.int32),
            image_size=np.array(pipeline.image_size, dtype=np.int32),
            affine=nii_affine,
        )
        print(f"Saved heatmaps: {out_path}")

        heatmaps = result["axial_heatmaps"]["heatmaps"]
        z_indices = result["axial_heatmaps"]["z_indices"]
        if heatmaps.size == 0:
            print(f"Skipped heatmap NIfTI save (no heatmaps) for {case_id}.")
            continue
        summed = np.sum(heatmaps, axis=1)
        output_depth = int(z_indices.max()) + 1 if z_indices.size > 0 else summed.shape[0]
        summed_volume = _build_volume(summed, z_indices, output_depth)
        transpose_mode = "swap_alt"
        if args.transpose_xy:
            transpose_mode = "swap"
        elif args.transpose_xy_alt:
            transpose_mode = "swap_alt"
        heatmap_to_voxel = _heatmap_to_voxel_affine(
            volume[:, :, 0].shape,
            pipeline.image_size,
            transpose_mode=transpose_mode,
        )
        slicer_affine = nii_affine @ heatmap_to_voxel
        if args.slicer_transform:
            slicer_affine = _apply_slicer_transform(slicer_affine)
        nii_path = case_output_dir / f"{case_id}_axial_heatmaps_sum_slicer.nii.gz"
        nib.save(nib.Nifti1Image(summed_volume, slicer_affine), str(nii_path))
        print(f"Saved heatmap NIfTI (Slicer space): {nii_path}")


if __name__ == "__main__":
    main()
