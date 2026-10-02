import json

import numpy as np
import torch
from torchvision import transforms
import nibabel as nib
from models import softmax2d


class KneeSlice:
    """
    Class that represents a knee slice with image and landmarks
    
    Args:
        image_path (str): Path to the image of the knee slice.
        z_coordinate (float): Z-coordinate of the given knee slice
        keypoints (dict): Dictionary of (keypoint name, coordinates) pairs for all keypoints present in the slice
        keypoint_names (list[str]): List of all possible keypoint names
    """
    def __init__(self, image_path, z_coordinate, keypoints, keypoint_names):
        
        self.image_path = image_path
        self.keypoint_names = keypoint_names
        self.num_keypoints = len(keypoint_names)

        img = nib.load(self.image_path)
        self.spacing = np.array(img.header.get_zooms())
        self.slice_index = int(round(z_coordinate / self.spacing[2]))

        # read keypoints
        self.keypoints = [None] * self.num_keypoints
        for i, name in enumerate(keypoint_names):
            if name in keypoints:
                self.keypoints[i] = np.array(keypoints[name]) / self.spacing[:2]
        
    def load_image(self):
        img = nib.load(self.image_path)
        img_tensor = torch.tensor(img.dataobj[:, :, self.slice_index], dtype=torch.float32)
        min_pixel, max_pixel = img_tensor.min(), img_tensor.max()
        img_tensor = (img_tensor - min_pixel) / max(max_pixel - min_pixel, 1e-10) # normalize
        return img_tensor

    def get_annotated_indices_and_keypoints(self):
        return [i for i in range(self.num_keypoints) if self.keypoints[i] is not None], [p for p in self.keypoints if p is not None]

    def get_keypoints(self):
        return self.keypoints


class KneeDataset(torch.utils.data.Dataset):
    """
    Dataset for knee landmarks
    
    Args:
        data_path_file (str): Path to file containing the paths to the data.
        lakefs_loader (LakeFSLoader, optional): LakeFSLoader for downloading training data; if None, the data has to be stored locally (default: None).
        keypoint_names (list[str]): List of all possible keypoint names
        image_size (int): Image size of all images.
        augmentation (DataAugmentation, optional): Data augmentations; if None, no augmentation is performed (default: None).
        image_only (bool, optional): Whether only images and no labels are given (default: False).
        keypoint_std (float, optional): Standard deviation of Gaussian in target heatmaps (default: 3.).
    """
    def __init__(self, data_path_file, lakefs_loader, keypoint_names, image_size=256,
                 augmentation=None, image_only=False, keypoint_std=3.):
        
        self.lakefs_loader = lakefs_loader
        self.keypoint_names = keypoint_names
        self.num_keypoints = len(keypoint_names)
        self.image_size = image_size
        self.augmentation = augmentation
        self.image_only = image_only
        self.keypoint_std = keypoint_std
        
        self.knee_slices = []
        label_counter = 0
        with open(data_path_file) as json_file:
            path_files = json.load(json_file)
            for path_file in path_files:
                image_path = self.lakefs_loader.get_file(path_file["image"])

                nifti_image = nib.load(image_path)
                z_spacing = nifti_image.header.get_zooms()[2]
                z_coords = np.arange(nifti_image.get_fdata().shape[2]) * z_spacing

                for z_coord in z_coords:
                    keypoints = None
                    for z_coord_kp, kps in path_file["labels"].items():
                        if np.isclose(float(z_coord_kp), z_coord):
                            keypoints = kps
                            break
                    if keypoints is None:
                        if np.random.rand() < 0.1: # keep some slices without keypoints inside in case the model should detect that there are no keypoints (by predicting the uniform distribution)
                            self.knee_slices.append(KneeSlice(image_path, float(z_coord), [], keypoint_names))
                    else:
                        self.knee_slices.append(KneeSlice(image_path, float(z_coord), keypoints, keypoint_names))
                        label_counter += 1
        
        print(f"num slices with keypoints = {label_counter}")
        print(f"num slices without keypoints = {len(self.knee_slices) - label_counter}")

    def contains_only_images(self):
        return self.image_only

    def get_num_landmarks(self):
        return self.num_keypoints

    def get_class_labels(self):
        return self.keypoint_names

    def __len__(self):
        return len(self.knee_slices)

    def _generate_heatmap(self, point, image_size, invisible, std):
        """ Generates a heatmap (image_size x image_size) with a spherical Gaussian around point and std=self.keypoint_std, scaled so that max=1, min=0; returns NaN-tensor if point is None """
        if invisible:
            return torch.ones((image_size, image_size)) / (image_size ** 2)
        if point is None and not invisible:
            return torch.full((image_size, image_size), float('nan'))
        x = torch.arange(0, image_size)
        y = torch.arange(0, image_size)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        pos = torch.stack((xx, yy), dim=-1).float()
        mean = torch.tensor(point).float()
        log_prob = - torch.einsum('hwd,hwd->hw', *[(pos - mean.view(1, 1, *mean.shape))]*2) / (2 * std**2)
        return softmax2d(log_prob)

    def __getitem__(self, idx):
        """ returns image, idx, [ heatmap|NaN-tensor ], [ implant_label|-1 ] (sorted by class index) """
        knee = self.knee_slices[idx]
        image = knee.load_image()

        if self.image_only:
            image = transforms.ToTensor()(image)
            return {"image": image, "idx" : idx}
        else:
            if self.augmentation is not None:
                keypoints = [None] * self.num_keypoints
                classes_existing, keypoints_existing = knee.get_annotated_indices_and_keypoints()
                image_numpy, keypoints_aug, classes_aug = self.augmentation.augment(image, keypoints_existing, classes_existing)
                for c, p in zip(classes_aug, keypoints_aug):
                    keypoints[c] = np.array(p)
                invisible = [ keypoints[i] is None and (p is not None or i < 3) for i, p in enumerate(keypoints) ] # if keypoints are marked as invisible, the model is trained to predict the uniform distribution
                image = torch.from_numpy(image_numpy)
            else:
                keypoints = knee.get_keypoints()
                invisible = [False] * self.num_keypoints
            
            # resize global images & heatmaps
            heatmaps = torch.stack([self._generate_heatmap(p, self.image_size, invisible=invisible[i], std=self.keypoint_std) for i, p in enumerate(keypoints)])
            
            return {
                    "idx" : idx,
                    "image" : image.unsqueeze(0),
                    "heatmaps" : heatmaps,
                    "keypoints" : keypoints,
                    "path" : knee.image_path
                }
            
            
