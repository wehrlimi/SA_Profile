"""
Two-stage inference pipeline:
1. Sagittal model predicts z-bounds
2. Axial model predicts landmarks within bounded region
"""

import os
import sys
from pathlib import Path

# Add project root to path to enable imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from typing import Tuple, List, Dict
from scipy.ndimage import zoom
import nibabel as nib


class TwoStageInferencePipeline:
    """
    Complete inference pipeline for knee landmark detection.
    """

    def __init__(self,
                 sagittal_model_path: str,
                 axial_model_path: str,
                 device: str = 'cuda',
                 image_size: Tuple[int, int] = (256, 256),
                 z_margin: int = 5):
        """
        Args:
            sagittal_model_path: Path to trained sagittal bounds model
            axial_model_path: Path to trained axial landmark model
            device: 'cuda' or 'cpu'
            image_size: Target image size for processing (tuple or int)
            z_margin: Additional margin (in slices) to add to predicted bounds
        """
        self.device = torch.device(device)
        # Handle image_size as int or tuple
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
        self.z_margin = z_margin

        # Load models
        from models import load_model
        self.sagittal_model = load_model(sagittal_model_path, self.device)
        self.axial_model = load_model(axial_model_path, self.device)

        self.sagittal_model.eval()
        self.axial_model.eval()

    def predict_case(self, volume: np.ndarray, nii_affine: np.ndarray = None) -> Dict:
        """
        Run full inference on a 3D volume.

        Args:
            volume: 3D numpy array (H, W, D)
            nii_affine: Optional affine matrix from NIfTI file for coordinate transformations

        Returns:
            dict: {
                'bounds': {
                    'z_min': int,
                    'z_max': int,
                    'points': [[y, z], [y, z]],  # sagittal coordinates in voxel space
                    'points_world': [[x, y, z], [x, y, z]]  # world coordinates (LPS)
                },
                'landmarks': {
                    z_coord: {
                        'points': [[x, y], [x, y], [x, y]],  # voxel coordinates
                        'points_world': [[x, y, z], [x, y, z], [x, y, z]]  # world coordinates (LPS)
                    },
                    ...
                }
            }
        """
        # Stage 1: Predict bounds from sagittal slice
        bounds = self._predict_bounds(volume, nii_affine)

        # Stage 2: Predict landmarks in bounded region
        landmarks = self._predict_landmarks(volume, bounds['z_min'], bounds['z_max'], nii_affine)

        return {
            'bounds': bounds,
            'landmarks': landmarks
        }

    def _predict_bounds(self, volume: np.ndarray, nii_affine: np.ndarray = None) -> Dict:
        """
        Stage 1: Predict z-bounds from middle sagittal slice.

        Args:
            volume: 3D numpy array (H, W, D)
            nii_affine: Optional affine matrix from NIfTI file

        Returns:
            dict: {
                'z_min': int,
                'z_max': int,
                'points': [[y, z], [y, z]],
                'confidence': float
            }
        """
        # Extract sagittal slice - always use axis 0 as sagittal axis
        sagittal_axis = 0
        # NOTE: During training, the sagittal slice is selected based on the average position
        # of the boundary points along the sagittal axis. At inference time, we don't have
        # boundary points yet, so we use the geometric center of the volume.
        sagittal_slice_idx = int(np.round((volume.shape[sagittal_axis] - 1) / 2.0))
        sagittal_slice_idx = np.clip(sagittal_slice_idx, 0, volume.shape[sagittal_axis] - 1)
        
        # Extract sagittal slice along axis 0: volume[sagittal_slice_idx, :, :] = (j, k)
        sagittal_slice = volume[sagittal_slice_idx, :, :]
        pre_rot_shape = sagittal_slice.shape

        # Store original unrotated slice for debug visualization
        sagittal_slice_original = sagittal_slice.copy()
        
        # Rotate image 90 degrees counter-clockwise to match training orientation
        sagittal_slice = np.rot90(sagittal_slice, k=1)  # Rotate 90° CCW
        original_size = np.array(sagittal_slice.shape)  # (H, W) after rotation

        # Normalize image
        sagittal_slice = self._normalize_image(sagittal_slice)
        sagittal_slice_normalized = sagittal_slice.copy()  # Store for debug

        # Resize to target size
        sagittal_slice_resized = self._resize_image(sagittal_slice, self.image_size)

        # Convert to tensor
        image_tensor = torch.from_numpy(sagittal_slice_resized).unsqueeze(0).unsqueeze(0).float()
        image_tensor = image_tensor.to(self.device)

        # Predict
        with torch.no_grad():
            result = self.sagittal_model.predict(image_tensor)

        # Extract keypoints (2 points: upper and lower bounds)
        # Model outputs heatmaps with shape (batch, num_channels, H, W) where num_channels=2
        # extract_keypoints returns: for each batch item, a list of keypoints (one per channel)
        # So result['keypoints'] has shape (batch, num_channels, 2) = (1, 2, 2)
        # result['keypoints'][0] gives us (2, 2) = [[x1, y1], [x2, y2]]
        keypoints_resized = result['keypoints'][0]  # Shape: (2, 2) - [[x, y], [x, y]]
        
        # Debug: Check heatmaps to see if they're different
        heatmaps_raw = result['heatmaps'][0]  # Shape: (2, H, W) - 2 heatmaps, one per boundary point
        print(f"DEBUG: keypoints_resized from model = {keypoints_resized}")
        print(f"DEBUG: Heatmap shapes: {heatmaps_raw.shape}")
        print(f"DEBUG: Heatmap max locations:")
        for ch in range(2):
            max_idx = torch.argmax(heatmaps_raw[ch]).item()
            size = heatmaps_raw[ch].shape[-1]
            max_x = max_idx % size
            max_y = max_idx // size
            max_val = heatmaps_raw[ch].max().item()
            print(f"  Channel {ch}: max at ({max_x}, {max_y}) with value {max_val:.4f}")
        print(f"DEBUG: Extracted keypoints: {keypoints_resized}")
        print(f"DEBUG: Are keypoints different? {not np.allclose(keypoints_resized[0], keypoints_resized[1], atol=0.1)}")
        print(f"DEBUG: original_size = {original_size}, image_size = {self.image_size}")

        # Scale back to original rotated image coordinates
        # The transformation depends on which sagittal axis was used
        scale = original_size / np.array(self.image_size)  # Scale factors after rotation
        pre_rot_height, pre_rot_width = pre_rot_shape
        print(f"DEBUG: sagittal_axis = {sagittal_axis}, sagittal_slice_idx = {sagittal_slice_idx}")
        print(f"DEBUG: scale = {scale}, pre_rot_height = {pre_rot_height}, pre_rot_width = {pre_rot_width}")
        print(f"DEBUG: original_size after rotation = {original_size}")
        
        # Transform back based on sagittal axis (matching training logic)
        # Training: keypoints_pixels = [kp[0], pre_rot_height - 1 - kp[1]] where kp is from keypoints_pixels_unrot
        # So inverse: keypoints_pixels_unrot = [kp[0], pre_rot_height - 1 - kp[1]]
        # Then scale back: divide by scale factors
        # Model predicts [x, y] where x=column, y=row in resized rotated space
        # After scaling back: [x/scale[1], y/scale[0]] gives us keypoints_pixels
        # Then inverse y-inversion: [x/scale[1], pre_rot_height - 1 - (y/scale[0])] gives us keypoints_pixels_unrot
        keypoints_rotated_scaled = np.array([
            [kp[0] / scale[1], kp[1] / scale[0]]  # Scale back from resized to rotated
            for kp in keypoints_resized
        ])
        
        # Inverse y-inversion to get back to unrotated coordinates
        keypoints_unrot = np.array([
            [kp[0], pre_rot_height - 1 - kp[1]]  # Inverse y-inversion: [row, col] in unrotated slice
            for kp in keypoints_rotated_scaled
        ])
        
        print(f"DEBUG: keypoints_resized = {keypoints_resized}")
        print(f"DEBUG: keypoints_rotated_scaled = {keypoints_rotated_scaled}")
        print(f"DEBUG: keypoints_unrot after transformation = {keypoints_unrot}")

        # Calculate confidence (could use heatmap max values)
        heatmaps = result['heatmaps'][0]  # Shape: (2, H, W)
        confidence = float(heatmaps.max(dim=-1)[0].max(dim=-1)[0].mean())

        # Convert bounds points to world coordinates (LPS)
        # Sagittal slice is extracted as volume[sagittal_slice_idx, :, :] = (j, k)
        # kp is [j, k] in the unrotated 2D slice
        points_world = []
        world_z_coords = []
        if nii_affine is not None:
            for kp in keypoints_unrot:
                # Reconstruct full 3D voxel coordinates: [i, j, k] where i = sagittal_slice_idx
                voxel_coords = np.array([sagittal_slice_idx, kp[0], kp[1], 1.0])  # [i, j, k]
                
                # Transform to world coordinates: [x, y, z, 1] = affine @ [i, j, k, 1]
                world_coords = nii_affine @ voxel_coords
                world_coords_3d = world_coords[:3]  # [x, y, z] in LPS
                points_world.append(world_coords_3d.tolist())
                world_z_coords.append(world_coords_3d[2])  # z-coordinate in world space (LPS) = Superior
        else:
            points_world = [[0, 0, 0], [0, 0, 0]]  # Fallback if no affine
            world_z_coords = [0, 0]

        # Extract z-coordinates in world space and convert back to voxel k indices
        # The boundary points mark superior-inferior extent (z in LPS = Superior)
        # We need to find which axial slices (k indices) correspond to the world z-coordinate range
        if nii_affine is not None and len(world_z_coords) > 0:
            # The boundary points have different i coordinates (superior-inferior in sagittal slice)
            # but we need to find the k indices that correspond to their world z-coordinates
            # For axial slices, we sample along the k-axis at the center of each axial slice
            k_indices = np.arange(volume.shape[2])
            center_i, center_j = volume.shape[0] // 2, volume.shape[1] // 2
            
            # Calculate world z for each k index at the center of axial slices
            # world_z = affine[2,0]*i + affine[2,1]*j + affine[2,2]*k + affine[2,3]
            world_z_at_k = nii_affine[2, 0] * center_i + nii_affine[2, 1] * center_j + nii_affine[2, 2] * k_indices + nii_affine[2, 3]
            
            # Find k indices that correspond to the min and max world z-coordinates
            world_z_min = np.min(world_z_coords)
            world_z_max = np.max(world_z_coords)
            
            # Find k indices closest to the min and max world z-coordinates
            z_idx_min = int(np.argmin(np.abs(world_z_at_k - world_z_min)))
            z_idx_max = int(np.argmin(np.abs(world_z_at_k - world_z_max)))
            
            # Debug output
            print(f"DEBUG: world_z_coords = {world_z_coords}")
            print(f"DEBUG: world_z_min = {world_z_min}, world_z_max = {world_z_max}")
            print(f"DEBUG: z_idx_min = {z_idx_min}, z_idx_max = {z_idx_max}")
            print(f"DEBUG: keypoints_unrot = {keypoints_unrot}")
            
            # Use the range between the two boundary points
            z_min = int(np.min([z_idx_min, z_idx_max])) - self.z_margin
            z_max = int(np.max([z_idx_min, z_idx_max])) + self.z_margin
        else:
            # Fallback: use k coordinates directly (incorrect but better than nothing)
            z_coords = keypoints_unrot[:, 1]  # k coordinates
            z_min = int(np.min(z_coords)) - self.z_margin
            z_max = int(np.max(z_coords)) + self.z_margin

        # Clamp to volume bounds
        z_min = max(0, z_min)
        z_max = min(volume.shape[2] - 1, z_max)

        return {
            'z_min': z_min,
            'z_max': z_max,
            'points': keypoints_unrot.tolist(),  # Return in unrotated space (voxel)
            'points_world': points_world,  # World coordinates (LPS)
            'confidence': confidence
        }

    def _predict_landmarks(self,
                           volume: np.ndarray,
                           z_min: int,
                           z_max: int,
                           nii_affine: np.ndarray = None,
                           stride: int = 1) -> Dict[int, List[List[float]]]:
        """
        Stage 2: Predict landmarks for axial slices in bounded region.

        Args:
            volume: 3D volume
            z_min: Lower bound (superior)
            z_max: Upper bound (superior)
            nii_affine: Optional affine matrix from NIfTI file
            stride: Process every Nth slice (default: 1, process all)

        Returns:
            dict: {z_coord: [[x, y], [x, y], [x, y]], ...} in original unrotated space
        """
        landmarks = {}

        # Process each slice in the bounded region
        for z in range(z_min, z_max + 1, stride):
            # Extract axial slice
            axial_slice = volume[:, :, z]  # (H, W)
            pre_rot_shape = axial_slice.shape  # (H, W) before rotation

            # Rotate image 90 degrees counter-clockwise to match training orientation
            axial_slice = np.rot90(axial_slice, k=1)  # Rotate 90° CCW
            original_size = np.array(axial_slice.shape)  # (H, W) after rotation

            # Normalize image
            axial_slice = self._normalize_image(axial_slice)

            # Resize to target size
            axial_slice_resized = self._resize_image(axial_slice, self.image_size)

            # Convert to tensor
            image_tensor = torch.from_numpy(axial_slice_resized).unsqueeze(0).unsqueeze(0).float()
            image_tensor = image_tensor.to(self.device)

            # Predict
            with torch.no_grad():
                result = self.axial_model.predict(image_tensor)

            # Extract keypoints (3 points per slice)
            # Keypoints are in resized rotated image space [x, y]
            keypoints_resized = result['keypoints'][0]  # Shape: (3, 2) - [[x, y], ...]

            # Scale back to original rotated image coordinates
            # In training (axial):
            # - keypoints_pixels = [i, pre_rot_height - 1 - j] where i=axis0, j=axis1
            # - Scaling: keypoints_resized = keypoints_pixels * scale where scale = [scale_H, scale_W]
            #   = [i * scale_H, (pre_rot_height - 1 - j) * scale_W]
            # To scale back: [x / scale_H, y / scale_W] = [i, pre_rot_height - 1 - j]
            scale = original_size / np.array(self.image_size)  # [scale_H, scale_W] after rotation
            keypoints_rotated = keypoints_resized * scale  # Element-wise multiplication

            # Transform keypoints back to unrotated space
            # keypoints_rotated is [i, pre_rot_height - 1 - j] in rotated image space
            # To get [i, j]: invert the y-inversion
            # j = pre_rot_height - 1 - (pre_rot_height - 1 - j) = j
            # So: [i, j] = [kp[0], pre_rot_height - 1 - kp[1]]
            pre_rot_height, pre_rot_width = pre_rot_shape
            keypoints_unrot = np.array([
                [kp[0], pre_rot_height - 1 - kp[1]]  # Get back to [i, j]
                for kp in keypoints_rotated
            ])

            # Convert to world coordinates (LPS)
            # For axial slice extracted as volume[:, :, z]:
            # - Slice dimensions are (i, j) where i is axis 0, j is axis 1
            # - keypoints_unrot is [i, j] after inverse transformation
            # - Full voxel coordinates: [i, j, k] = [kp[0], kp[1], z]
            keypoints_world = []
            if nii_affine is not None:
                for kp in keypoints_unrot:
                    # Reconstruct full 3D voxel coordinates
                    # kp is [i, j] where i=axis0, j=axis1, and k=z (axis2)
                    voxel_coords = np.array([kp[0], kp[1], z, 1.0])
                    # Transform to world coordinates: [x, y, z, 1] = affine @ [i, j, k, 1]
                    world_coords = nii_affine @ voxel_coords
                    keypoints_world.append(world_coords[:3].tolist())  # [x, y, z] in LPS
            else:
                keypoints_world = [[0, 0, 0]] * len(keypoints_unrot)  # Fallback

            landmarks[z] = {
                'points': keypoints_unrot.tolist(),  # Voxel coordinates
                'points_world': keypoints_world  # World coordinates (LPS)
            }

        return landmarks

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range"""
        min_val, max_val = image.min(), image.max()
        if max_val > min_val:
            return (image - min_val) / (max_val - min_val)
        return image

    def _resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size"""
        scale_factors = np.array(target_size) / np.array(image.shape)
        return zoom(image, scale_factors, order=1)

    def batch_predict(self, volumes: List[np.ndarray]) -> List[Dict]:
        """
        Run inference on multiple volumes.

        Args:
            volumes: List of 3D numpy arrays

        Returns:
            List of prediction dictionaries
        """
        results = []
        for i, volume in enumerate(volumes):
            print(f"Processing volume {i + 1}/{len(volumes)}...")
            result = self.predict_case(volume)
            results.append(result)
        return results

    def convert_to_slicer_format(self, result: Dict, output_dir: str = ".", case_id: str = None) -> Dict[str, str]:
        """
        Convert prediction results to Slicer markup format and save as JSON files.

        Args:
            result: Prediction result dictionary from predict_case()
            output_dir: Directory to save output files

        Returns:
            dict with paths to saved files: {'bounds': path, 'landmarks': path}
        """
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create TrochleaBounds.mrk.json
        bounds_points = result['bounds']['points_world']
        # Apply rotation matrix (-1, -1, 1) to world coordinates for Slicer display
        # This transforms (x, y, z) -> (-x, -y, z) in LPS coordinate system
        bounds_points_slicer = [
            [-x, -y, z] for x, y, z in bounds_points
        ]
        bounds_markup = {
            "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
            "markups": [
                {
                    "type": "Fiducial",
                    "coordinateSystem": "LPS",
                    "coordinateUnits": "mm",
                    "locked": True,
                    "fixedNumberOfControlPoints": False,
                    "labelFormat": "%N-%d",
                    "lastUsedControlPointNumber": len(bounds_points),
                    "controlPoints": [
                        {
                            "id": str(i + 1),
                            "label": f"TrochleaBounds-{i + 1}",
                            "description": "",
                            "associatedNodeID": "vtkMRMLScalarVolumeNode1",
                            "position": bounds_points_slicer[i],  # [x, y, z] in LPS (transformed for Slicer)
                            "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                            "selected": True,
                            "locked": False,
                            "visibility": True,
                            "positionStatus": "defined"
                        }
                        for i in range(len(bounds_points))
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
                        "snapMode": "toVisibleSurface"
                    }
                }
            ]
        }

        bounds_suffix = f"_{case_id}" if case_id else ""
        bounds_path = output_dir / f"TrochleaBounds{bounds_suffix}.mrk.json"
        with open(bounds_path, 'w') as f:
            json.dump(bounds_markup, f, indent=4)

        # Create AxialLabels.mrk.json
        # Collect all landmark points from all slices
        all_landmark_points = []
        for z_coord, landmark_data in sorted(result['landmarks'].items()):
            points_world = landmark_data['points_world']
            # Apply rotation matrix (-1, -1, 1) to world coordinates for Slicer display
            # This transforms (x, y, z) -> (-x, -y, z) in LPS coordinate system
            points_world_slicer = [
                [-x, -y, z] for x, y, z in points_world
            ]
            all_landmark_points.extend(points_world_slicer)

        landmarks_markup = {
            "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
            "markups": [
                {
                    "type": "Fiducial",
                    "coordinateSystem": "LPS",
                    "coordinateUnits": "mm",
                    "locked": False,
                    "fixedNumberOfControlPoints": False,
                    "labelFormat": "%N-%d",
                    "lastUsedControlPointNumber": len(all_landmark_points),
                    "controlPoints": [
                        {
                            "id": str(i + 1),
                            "label": f"AxialLabels-{i + 1}",
                            "description": "",
                            "associatedNodeID": "vtkMRMLScalarVolumeNode1",
                            "position": all_landmark_points[i],  # [x, y, z] in LPS
                            "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                            "selected": True,
                            "locked": True,
                            "visibility": True,
                            "positionStatus": "defined"
                        }
                        for i in range(len(all_landmark_points))
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
                        "snapMode": "toVisibleSurface"
                    }
                }
            ]
        }

        landmarks_suffix = f"_{case_id}" if case_id else ""
        landmarks_path = output_dir / f"AxialLabels{landmarks_suffix}.mrk.json"
        with open(landmarks_path, 'w') as f:
            json.dump(landmarks_markup, f, indent=4)

        return {
            'bounds': str(bounds_path),
            'landmarks': str(landmarks_path)
        }


def main():
    """Example usage"""
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Two-stage knee landmark inference")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--sagittal_model', type=str, required=True, help='Path to sagittal model checkpoint')
    parser.add_argument('--axial_model', type=str, required=True, help='Path to axial model checkpoint')
    parser.add_argument('--input_volume', type=str, help='Path to input NIfTI volume')
    parser.add_argument('--input_dir', type=str, help='Path to directory containing .nii.gz files')
    parser.add_argument('--output', type=str, default='inference_output', help='Output directory for Slicer markup files')
    parser.add_argument('--output_format', type=str, choices=['slicer', 'json'], default='slicer', help='Output format: slicer (default) or json (legacy, deprecated)')
    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)

    # Initialize pipeline
    # Handle image_size as int or tuple
    image_size = cfg.program.image_size
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    elif isinstance(image_size, (list, tuple)):
        image_size = tuple(image_size)
    else:
        image_size = (256, 256)  # Default fallback

    pipeline = TwoStageInferencePipeline(
        sagittal_model_path=args.sagittal_model,
        axial_model_path=args.axial_model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        image_size=image_size,
        z_margin=cfg.inference.get('z_margin', 5)
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
        # Determine case ID from filename (strip .nii.gz or .nii)
        case_id = input_path.name
        if case_id.endswith(".nii.gz"):
            case_id = case_id[:-7]
        elif case_id.endswith(".nii"):
            case_id = case_id[:-4]

        case_output_dir = output_root / case_id
        case_output_dir.mkdir(parents=True, exist_ok=True)

        # Load volume
        nii = nib.load(str(input_path))
        volume = nii.get_fdata()
        nii_affine = nii.affine  # Get affine matrix for coordinate transformations

        # Run inference
        print(f"Running inference on {input_path}...")
        result = pipeline.predict_case(volume, nii_affine=nii_affine)

        # Save results
        if args.output_format == 'slicer':
            saved_files = pipeline.convert_to_slicer_format(result, case_output_dir, case_id=case_id)
            print(f"Results saved in Slicer format:")
            print(f"  TrochleaBounds: {saved_files['bounds']}")
            print(f"  AxialLabels: {saved_files['landmarks']}")
        else:
            # Save in legacy JSON format (deprecated)
            import json
            output_path = case_output_dir / 'predictions.json'
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"WARNING: Legacy JSON format is deprecated. Use --output_format slicer (default)")
            print(f"Results saved to {output_path}")

        print(f"Predicted bounds: z={result['bounds']['z_min']} to {result['bounds']['z_max']}")
        print(f"Detected landmarks on {len(result['landmarks'])} slices")


if __name__ == "__main__":
    main()