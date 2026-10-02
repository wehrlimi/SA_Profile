import json
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from models import softmax2d


class SagittalBoundsDataset(Dataset):
    """
    Dataset for training the sagittal boundary detection model.

    Returns sagittal slices with 2 boundary points marking trochlea extent.
    """

    def __init__(self, data_path_file, lakefs_loader, image_size, augmentation=None, keypoint_std=3.0):
        """
        Args:
            data_path_file: Path to JSON file with data splits
            lakefs_loader: LakeFSLoader instance for loading images
            image_size: Target image size (H, W) or int (square)
            augmentation: Optional augmentation transform
            keypoint_std: Standard deviation for heatmap generation
        """
        with open(data_path_file, 'r') as f:
            self.data = json.load(f)

        self.lakefs_loader = lakefs_loader
        # Handle image_size as int or tuple
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
        self.augmentation = augmentation
        self.keypoint_std = keypoint_std
        self.num_landmarks = 2  # Two boundary points
        self.num_keypoints = self.num_landmarks  # Alias for evaluation compatibility
        self.keypoint_names = ['boundary_point_0', 'boundary_point_1']  # For evaluation compatibility

    def __len__(self):
        return len(self.data)

    def get_num_landmarks(self):
        return self.num_landmarks

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load full 3D volume
        volume = self.lakefs_loader.load_volume(item['image'])
        
        # Load NIfTI to get spacing, orientation, and affine info
        local_path = self.lakefs_loader.get_file(item['image'])
        nii = nib.load(local_path)
        spacing = np.array(nii.header.get_zooms())
        affine = nii.affine  # Transform matrix from voxel to world coordinates

        # Extract sagittal slice using x-coordinates from boundary points
        bounds = item['bounds']
        inv_affine = np.linalg.inv(affine)
        
        # Get boundary points in world coordinates (x, y, z in LPS)
        if 'xyz' in bounds:
            # New format: use full 3D coordinates
            world_points_3d = np.array(bounds['xyz'])  # [[x, y, z], [x, y, z]] in world space (LPS)
        else:
            # Backward compatibility: approximate using middle slice
            keypoints_list = bounds['points']  # [[y, z], [y, z]]
            # Estimate x-coordinate (rough approximation - ideally should have full 3D)
            center_points = [volume.shape[i] // 2 for i in range(3)]
            # Use middle of axis 1 as rough estimate (may be wrong for rotated volumes)
            estimated_x = affine[0, 0] * center_points[0] + affine[0, 1] * center_points[1] + affine[0, 2] * center_points[2] + affine[0, 3]
            world_points_3d = np.array([
                [estimated_x, kp[0], kp[1]]  # [x, y, z] - approximate x
                for kp in keypoints_list
            ])
        
        # Convert world coordinates to voxel coordinates
        voxel_points = []
        for world_point in world_points_3d:
            world_point_homogeneous = np.array([world_point[0], world_point[1], world_point[2], 1.0])
            voxel_point_homogeneous = inv_affine @ world_point_homogeneous
            voxel_point = voxel_point_homogeneous[:3]  # [i, j, k]
            voxel_points.append(voxel_point)
        
        voxel_points = np.array(voxel_points)  # Shape: (2, 3) - [[i, j, k], [i, j, k]]
        
        # Find which axis has the least variation between the two boundary points
        # The sagittal axis is the one where both points are roughly the same (they're on the same sagittal slice)
        axis_variations = np.std(voxel_points, axis=0)  # Variation along each axis
        sagittal_axis = int(np.argmin(axis_variations))  # Axis with smallest variation
        
        # Use average position along sagittal axis for slice extraction
        sagittal_slice_idx = int(np.round(np.mean(voxel_points[:, sagittal_axis])))
        sagittal_slice_idx = np.clip(sagittal_slice_idx, 0, volume.shape[sagittal_axis] - 1)
        
        # Extract sagittal slice along the determined axis
        if sagittal_axis == 0:
            sagittal_slice = volume[sagittal_slice_idx, :, :]  # Extract along axis 0
            pre_rot_shape = sagittal_slice.shape  # (j, k) = (height, width)
        elif sagittal_axis == 1:
            sagittal_slice = volume[:, sagittal_slice_idx, :]  # Extract along axis 1
            pre_rot_shape = sagittal_slice.shape  # (i, k) = (height, width)
        else:  # sagittal_axis == 2
            sagittal_slice = volume[:, :, sagittal_slice_idx]  # Extract along axis 2
            pre_rot_shape = sagittal_slice.shape  # (i, j) = (height, width)
        
        # Rotate image 90 degrees counter-clockwise to match annotation orientation
        # 90° CCW: (x, y) -> (-y, x) which for numpy means transpose and flip one axis
        # np.rot90(array, k=1) rotates 90° CCW
        sagittal_slice = np.rot90(sagittal_slice, k=1)  # Rotate 90° counter-clockwise
        original_size = np.array(sagittal_slice.shape)  # (H, D) after rotation

        # Normalize image
        sagittal_slice = self._normalize_image(sagittal_slice)

        # Use the voxel points we already computed above for coordinate conversion
        # voxel_points is already computed: Shape: (2, 3) - [[i, j, k], [i, j, k]]
        keypoints_voxel = voxel_points  # Reuse the already-computed voxel coordinates
        
        # Extract 2D coordinates for sagittal slice based on which axis was used
        # Map voxel coordinates to 2D pixel coordinates in the unrotated slice
        # The annotations were made on the rotated view, so keypoints should map to rotated image
        # without applying rotation transformation (they're already in rotated coordinate space)
        
        # Get the pre-rotation shape to understand coordinate mapping
        pre_rot_height, pre_rot_width = pre_rot_shape
        
        # Map keypoints to 2D coordinates in the unrotated slice
        # Get the pre-rotation shape for coordinate inversion
        pre_rot_height, pre_rot_width = pre_rot_shape
        
        if sagittal_axis == 0:
            # Extracted as volume[sagittal_slice_idx, :, :], so 2D slice is (j, k)
            # Map voxel coords to 2D: [j (y), k (z)]
            keypoints_pixels_unrot = np.array([
                [kp_voxel[1], kp_voxel[2]]  # [j (y), k (z)] = [row, col] in unrotated slice
                for kp_voxel in keypoints_voxel
            ])
        elif sagittal_axis == 1:
            # Extracted as volume[:, sagittal_slice_idx, :], so 2D slice is (i, k)
            # Map voxel coords to 2D: [i (z), k (y)]
            keypoints_pixels_unrot = np.array([
                [kp_voxel[0], kp_voxel[2]]  # [i (z), k (y)] = [row, col] in unrotated slice
                for kp_voxel in keypoints_voxel
            ])
        else:  # sagittal_axis == 2
            # Extracted as volume[:, :, sagittal_slice_idx], so 2D slice is (i, j)
            # Map voxel coords to 2D: [i (z), j (y)]
            keypoints_pixels_unrot = np.array([
                [kp_voxel[0], kp_voxel[1]]  # [i (z), j (y)] = [row, col] in unrotated slice
                for kp_voxel in keypoints_voxel
            ])
        
        # Keypoints should map directly to the rotated image
        # Note: The image is rotated 90° CCW, but keypoints from annotations
        # are already in the coordinate system that matches the rotated view
        # So we map them directly with y-inversion to match image display coordinates
        # In image coordinates, y increases downward, so we invert: y_new = height - 1 - y_old
        keypoints_pixels = np.array([
            [kp[0], pre_rot_height - 1 - kp[1]]  # Keep x, invert y for image coordinates
            for kp in keypoints_pixels_unrot
        ])

        # Resize image to target size
        sagittal_slice_resized = self._resize_image(sagittal_slice, self.image_size)

        # Scale keypoints to resized image coordinates
        # original_size is (H, D) where H=superior (i), D=anterior (k)
        # keypoints_pixels is [[k, i], [k, i]] = [[D, H], [D, H]]
        # scale = [scale_H, scale_D]
        # So we need: keypoints[:, 0] (k/D) * scale[1], keypoints[:, 1] (i/H) * scale[0]
        scale = np.array(self.image_size) / original_size  # [scale_H, scale_D]
        keypoints_resized = np.array([
            [kp[0] * scale[1], kp[1] * scale[0]]  # [k * scale_D, i * scale_H]
            for kp in keypoints_pixels
        ])

        # Apply augmentation if provided
        if self.augmentation:
            # Convert normalized image (0-1) to uint8 (0-255) for albumentations
            image_uint8 = (sagittal_slice_resized * 255).astype(np.uint8)
            
            # Augmentation expects keypoints as list of (x, y) tuples and class_labels as list
            keypoints_for_aug = [(float(kp[0]), float(kp[1])) for kp in keypoints_resized]
            class_labels = [0, 1]  # Two boundary points
            image_numpy, keypoints_aug, classes_aug = self.augmentation.augment(
                image_uint8, keypoints_for_aug, class_labels
            )
            # Convert back to float (0-1) range
            image_numpy = image_numpy.astype(np.float32) / 255.0
            # Map augmented keypoints back to fixed-size array using class_labels
            # This ensures we always have self.num_landmarks keypoints (some may be None if removed by augmentation)
            keypoints_fixed = [None] * self.num_landmarks
            for c, p in zip(classes_aug, keypoints_aug):
                keypoints_fixed[c] = np.array(p)
            keypoints_resized = np.array([kp if kp is not None else [np.nan, np.nan] for kp in keypoints_fixed])
            sagittal_slice_resized = image_numpy
        else:
            image_numpy = sagittal_slice_resized

        # Generate heatmaps using torch softmax (consistent with loss function)
        # Always generate self.num_landmarks heatmaps
        heatmaps = self._generate_heatmaps(keypoints_resized, self.image_size, self.keypoint_std)

        # Convert to tensors
        image = torch.from_numpy(image_numpy).unsqueeze(0).float()  # (1, H, W)

        # Calculate spacing for the resized and rotated image (mm per pixel)
        # spacing is [dx, dy, dz] corresponding to [i, j, k]
        if sagittal_axis == 0:
            unrot_spacing = np.array([spacing[1], spacing[2]])  # [j, k]
        elif sagittal_axis == 1:
            unrot_spacing = np.array([spacing[0], spacing[2]])  # [i, k]
        else:
            unrot_spacing = np.array([spacing[0], spacing[1]])  # [i, j]
        
        # After rot90(k=1), axes are (unrot_axis1, unrot_axis0)
        rotated_spacing = np.array([unrot_spacing[1], unrot_spacing[0]])
        # After resizing to image_size (H, W)
        spacing_mm = rotated_spacing * (original_size / np.array(self.image_size))

        return {
            'image': image,
            'heatmaps': heatmaps,
            'keypoints': keypoints_resized,
            'case_id': item.get('metadata', idx),
            'spacing_mm': spacing_mm
        }

    def _normalize_image(self, image):
        """Normalize image to [0, 1] range"""
        min_val, max_val = image.min(), image.max()
        if max_val > min_val:
            return (image - min_val) / (max_val - min_val)
        return image

    def _resize_image(self, image, target_size):
        """Resize image to target size using numpy/scipy"""
        from scipy.ndimage import zoom

        scale_factors = np.array(target_size) / np.array(image.shape)
        resized = zoom(image, scale_factors, order=1)
        return resized

    def _generate_heatmaps(self, keypoints, image_size, std):
        """
        Generate heatmaps using torch softmax (consistent with CELoss).
        Returns torch tensors of shape (num_landmarks, H, W)
        Keypoints with NaN values will generate uniform heatmaps.
        """
        num_landmarks = len(keypoints)
        heatmaps = []
        
        for i, kp in enumerate(keypoints):
            # Check if keypoint is valid (not NaN)
            if np.any(np.isnan(kp)):
                # Generate uniform heatmap for missing keypoints
                heatmap = torch.ones(image_size[0], image_size[1], dtype=torch.float32) / (image_size[0] * image_size[1])
            else:
                x, y = kp[0], kp[1]  # keypoint coordinates in image space: kp is [x, y] = [col, row]

                # Create coordinate grid
                y_coords = torch.arange(0, image_size[0], dtype=torch.float32)  # rows (height)
                x_coords = torch.arange(0, image_size[1], dtype=torch.float32)  # cols (width)
                yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
                
                # Create position tensor
                pos = torch.stack([yy, xx], dim=-1)  # (H, W, 2) where pos[:,:,0] is y, pos[:,:,1] is x
                mean = torch.tensor([y, x], dtype=torch.float32)  # [y, x] to match pos[:,:,0] (y) and pos[:,:,1] (x)
                
                # Compute log probability (Gaussian-like)
                log_prob = -torch.sum((pos - mean) ** 2, dim=-1) / (2 * std ** 2)
                
                # Apply softmax2d (consistent with loss function)
                heatmap = softmax2d(log_prob.unsqueeze(0).unsqueeze(0), log=True).squeeze()
            heatmaps.append(heatmap)

        return torch.stack(heatmaps)  # (num_landmarks, H, W)


class AxialLandmarkDataset(Dataset):
    """
    Dataset for training the axial landmark detection model.

    Returns axial slices with 3 landmark points per slice.
    Can be used with ground truth bounds or predicted bounds.
    """

    def __init__(self, data_path_file, lakefs_loader, image_size,
                 augmentation=None, keypoint_std=3.0,
                 use_ground_truth_bounds=True):
        """
        Args:
            data_path_file: Path to JSON file with data splits
            lakefs_loader: LakeFSLoader instance for loading images
            image_size: Target image size (H, W) or int (square)
            augmentation: Optional augmentation transform
            keypoint_std: Standard deviation for heatmap generation
            use_ground_truth_bounds: If True, use ground truth z-ranges.
                                     If False, expect bounds to be predicted externally.
        """
        with open(data_path_file, 'r') as f:
            self.data = json.load(f)

        self.lakefs_loader = lakefs_loader
        # Handle image_size as int or tuple
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
        self.augmentation = augmentation
        self.keypoint_std = keypoint_std
        self.use_ground_truth_bounds = use_ground_truth_bounds
        self.num_landmarks = 3  # Three points per slice
        self.num_keypoints = self.num_landmarks  # Alias for evaluation compatibility
        self.keypoint_names = ['landmark_0', 'landmark_1', 'landmark_2']  # For evaluation compatibility

        # Flatten data: create one sample per labeled slice
        self.samples = []
        for item in self.data:
            for z_coord_str, label_data in item['labels'].items():
                # Handle both old format (direct list) and new format (dict with keypoints and xyz_coords)
                if isinstance(label_data, dict):
                    keypoints = label_data['keypoints']  # [[x, y], [x, y], [x, y]]
                    xyz_coords = label_data['xyz_coords']  # [[x, y, z], ...] full 3D
                else:
                    # Backward compatibility: old format where label_data is directly [[x, y], ...]
                    keypoints = label_data
                    xyz_coords = None
                
                # z_coord_str is now a world coordinate (z in LPS), not a slice index
                # We'll convert it to slice index later using affine transform
                z_coord_world = float(z_coord_str)
                
                self.samples.append({
                    'image': item['image'],
                    'z_coord_world': z_coord_world,  # World z-coordinate from annotation
                    'keypoints': keypoints,  # [[x, y], [x, y], [x, y]]
                    'xyz_coords': xyz_coords,  # [[x, y, z], ...] full 3D or None
                    'bounds': item['bounds'],
                    'metadata': item.get('metadata')
                })

    def __len__(self):
        return len(self.samples)

    def get_num_landmarks(self):
        return self.num_landmarks

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load full 3D volume
        volume = self.lakefs_loader.load_volume(sample['image'])

        # Load NIfTI to get spacing and affine for coordinate conversion
        local_path = self.lakefs_loader.get_file(sample['image'])
        nii = nib.load(local_path)
        spacing = np.array(nii.header.get_zooms())
        affine = nii.affine  # Transform matrix from voxel to world coordinates

        # Extract axial slice using z-coordinate from annotation
        z_coord_world = sample['z_coord_world']
        
        # Convert world z-coordinate (in LPS) to voxel slice index
        # The affine matrix converts (i, j, k) voxel to (x, y, z) world
        # affine transforms: [x, y, z, 1] = affine @ [i, j, k, 1]
        # z = affine[2, 0]*i + affine[2, 1]*j + affine[2, 2]*k + affine[2, 3]
        # For a point at center of slice: use i=H/2, j=W/2
        # Sample points along k-axis and find the closest to our target z-coordinate
        k_indices = np.arange(volume.shape[2])
        center_i, center_j = volume.shape[0] // 2, volume.shape[1] // 2
        world_z = affine[2, 0] * center_i + affine[2, 1] * center_j + affine[2, 2] * k_indices + affine[2, 3]
        
        # Find closest k index to target z-coordinate
        z_idx = int(np.argmin(np.abs(world_z - z_coord_world)))
        z_idx = np.clip(z_idx, 0, volume.shape[2] - 1)
        
        axial_slice = volume[:, :, z_idx]  # Shape: (H, W)
        
        # Rotate image 90 degrees counter-clockwise to match annotation orientation
        # np.rot90(array, k=1) rotates 90° CCW
        axial_slice = np.rot90(axial_slice, k=1)  # Rotate 90° counter-clockwise
        original_size = np.array(axial_slice.shape)  # (H, W) after rotation

        # Normalize image
        axial_slice = self._normalize_image(axial_slice)

        # Get keypoints - they are in world coordinates (LPS) from annotations
        # We need to convert them to voxel coordinates, then to 2D pixel coordinates for the axial slice
        keypoints_world = sample.get('xyz_coords', None)  # Full 3D coordinates if available
        keypoints_xy_world = np.array(sample['keypoints'])  # Shape: (3, 2) - [[x, y], [x, y], [x, y]] in world space (LPS)
        
        if keypoints_world is not None:
            # Use full 3D coordinates to convert to voxel coordinates
            inv_affine = np.linalg.inv(affine)
            keypoints_voxel = []
            for kp_world in keypoints_world:
                # kp_world is [x, y, z] in LPS world space
                world_point_homogeneous = np.array([kp_world[0], kp_world[1], kp_world[2], 1.0])
                voxel_point_homogeneous = inv_affine @ world_point_homogeneous
                voxel_point = voxel_point_homogeneous[:3]  # [i, j, k]
                keypoints_voxel.append(voxel_point)
            
            keypoints_voxel = np.array(keypoints_voxel)  # Shape: (3, 3) - [[i, j, k], ...]
            
            # Extract 2D pixel coordinates for axial slice
            # Axial slice is volume[:, :, z_idx], so 2D coordinates are (i, j) = (rows, cols)
            # Get pre-rotation shape for coordinate transformation
            pre_rot_shape = volume[:, :, z_idx].shape  # (H, W) before rotation
            pre_rot_height, pre_rot_width = pre_rot_shape
            
            # Apply transformations: first swap x and y (mirror on diagonal), then rotate 90° CCW
            # Step 1: Swap (i, j) -> (j, i) where i is row (y), j is col (x)
            # Step 2: Rotate 90° CCW: (x, y) -> (y, height - 1 - x)
            #        After swap we have (j, i), which becomes (i, pre_rot_height - 1 - j) after CCW rotation
            # Combined transformation: (i, j) -> (i, pre_rot_height - 1 - j)
            keypoints_pixels = np.array([
                [kp_voxel[0], pre_rot_height - 1 - kp_voxel[1]]  # Swap then rotate CCW: (i,j)->(j,i)->(i, h-1-j)
                for kp_voxel in keypoints_voxel
            ])
        else:
            # Fallback: assume keypoints are already in approximate pixel coordinates
            # This should not happen with the current data format, but keeping for backward compatibility
            keypoints_pixels = keypoints_xy_world

        # Resize image to target size
        axial_slice_resized = self._resize_image(axial_slice, self.image_size)

        # Scale keypoints to resized image coordinates
        scale = np.array(self.image_size) / original_size  # [scale_H, scale_W]
        keypoints_resized = keypoints_pixels * scale

        # Apply augmentation if provided
        if self.augmentation:
            # Convert normalized image (0-1) to uint8 (0-255) for albumentations
            image_uint8 = (axial_slice_resized * 255).astype(np.uint8)
            
            # Augmentation expects keypoints as list of (x, y) tuples and class_labels as list
            keypoints_for_aug = [(float(kp[0]), float(kp[1])) for kp in keypoints_resized]
            class_labels = [0, 1, 2]  # Three landmark points
            image_numpy, keypoints_aug, classes_aug = self.augmentation.augment(
                image_uint8, keypoints_for_aug, class_labels
            )
            # Convert back to float (0-1) range
            image_numpy = image_numpy.astype(np.float32) / 255.0
            # Map augmented keypoints back to fixed-size array using class_labels
            # This ensures we always have self.num_landmarks keypoints (some may be None if removed by augmentation)
            keypoints_fixed = [None] * self.num_landmarks
            for c, p in zip(classes_aug, keypoints_aug):
                keypoints_fixed[c] = np.array(p)
            keypoints_resized = np.array([kp if kp is not None else [np.nan, np.nan] for kp in keypoints_fixed])
            axial_slice_resized = image_numpy
        else:
            image_numpy = axial_slice_resized

        # Generate heatmaps using torch softmax (consistent with loss function)
        # Always generate self.num_landmarks heatmaps
        heatmaps = self._generate_heatmaps(keypoints_resized, self.image_size, self.keypoint_std)

        # Convert to tensors
        image = torch.from_numpy(image_numpy).unsqueeze(0).float()  # (1, H, W)

        # Calculate spacing for the resized and rotated image (mm per pixel)
        # Axial slice is volume[:, :, z_idx], so unrotated axes are [i, j]
        unrot_spacing = np.array([spacing[0], spacing[1]])
        # After rot90(k=1), axes are [j, i]
        rotated_spacing = np.array([unrot_spacing[1], unrot_spacing[0]])
        # After resizing to image_size (H, W)
        spacing_mm = rotated_spacing * (original_size / np.array(self.image_size))

        return {
            'image': image,
            'heatmaps': heatmaps,
            'keypoints': keypoints_resized,
            'z_coord': z_idx,  # Voxel slice index
            'z_coord_world': z_coord_world,  # World z-coordinate from annotation
            'bounds_z': sample['bounds']['z_coords'],  # For reference
            'case_id': sample.get('metadata', idx),
            'spacing_mm': spacing_mm
        }

    def _normalize_image(self, image):
        """Normalize image to [0, 1] range"""
        min_val, max_val = image.min(), image.max()
        if max_val > min_val:
            return (image - min_val) / (max_val - min_val)
        return image

    def _resize_image(self, image, target_size):
        """Resize image to target size"""
        from scipy.ndimage import zoom

        scale_factors = np.array(target_size) / np.array(image.shape)
        resized = zoom(image, scale_factors, order=1)
        return resized

    def _generate_heatmaps(self, keypoints, image_size, std):
        """
        Generate heatmaps using torch softmax (consistent with CELoss).
        Returns torch tensors of shape (num_landmarks, H, W)
        Keypoints with NaN values will generate uniform heatmaps.
        """
        num_landmarks = len(keypoints)
        heatmaps = []
        
        for i, kp in enumerate(keypoints):
            # Check if keypoint is valid (not NaN)
            if np.any(np.isnan(kp)):
                # Generate uniform heatmap for missing keypoints
                heatmap = torch.ones(image_size[0], image_size[1], dtype=torch.float32) / (image_size[0] * image_size[1])
            else:
                x, y = kp[0], kp[1]  # keypoint coordinates in image space (x, y)

                # Create coordinate grid
                y_coords = torch.arange(0, image_size[0], dtype=torch.float32)
                x_coords = torch.arange(0, image_size[1], dtype=torch.float32)
                yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
                
                # Create position tensor
                pos = torch.stack([yy, xx], dim=-1)  # (H, W, 2)
                mean = torch.tensor([y, x], dtype=torch.float32)  # Note: (y, x) order for indexing
                
                # Compute log probability (Gaussian-like)
                log_prob = -torch.sum((pos - mean) ** 2, dim=-1) / (2 * std ** 2)
                
                # Apply softmax2d (consistent with loss function)
                heatmap = softmax2d(log_prob.unsqueeze(0).unsqueeze(0), log=True).squeeze()
            heatmaps.append(heatmap)

        return torch.stack(heatmaps)  # (num_landmarks, H, W)


class AxialVolumeDataset(Dataset):
    """
    Alternative dataset that processes entire volumes at once.
    Useful for inference with predicted bounds.
    """

    def __init__(self, data_path_file, lakefs_loader, image_size,
                 predicted_bounds=None):
        """
        Args:
            data_path_file: Path to JSON file with data splits
            lakefs_loader: LakeFSLoader instance
            image_size: Target image size (H, W) or int (square)
            predicted_bounds: Optional dict mapping case_ids to predicted z-bounds
        """
        with open(data_path_file, 'r') as f:
            self.data = json.load(f)

        self.lakefs_loader = lakefs_loader
        # Handle image_size as int or tuple
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
        self.predicted_bounds = predicted_bounds or {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load full volume
        volume = self.lakefs_loader.load_volume(item['image'])

        # Determine z-range
        if self.predicted_bounds and idx in self.predicted_bounds:
            z_min, z_max = self.predicted_bounds[idx]
        else:
            # Use ground truth bounds
            z_min, z_max = item['bounds']['z_coords']

        # Extract slices in the bounded region
        z_range = range(int(z_min), int(z_max) + 1)
        slices = [volume[:, :, z] for z in z_range]

        # Resize all slices
        slices_resized = [self._resize_image(s, self.image_size) for s in slices]

        # Stack into tensor
        images = torch.stack([
            torch.from_numpy(s).unsqueeze(0).float()
            for s in slices_resized
        ])  # Shape: (num_slices, 1, H, W)

        return {
            'images': images,
            'z_range': list(z_range),
            'case_id': item.get('metadata', idx)
        }

    def _resize_image(self, image, target_size):
        """Resize image to target size"""
        from scipy.ndimage import zoom
        scale_factors = np.array(target_size) / np.array(image.shape)
        return zoom(image, scale_factors, order=1)