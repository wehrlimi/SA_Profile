import nibabel as nib
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Optional, Tuple
from torch.nn.modules.loss import _Loss

from sklearn.preprocessing import MinMaxScaler


def norm_grid(grid, xmin, xmax, smin=-1, smax=1):
    def min_max_scale(X, x_min, x_max, s_min, s_max):
        return (X - x_min)/(x_max - x_min)*(s_max - s_min) + s_min

    return min_max_scale(X=grid, x_min=xmin, x_max=xmax, s_min=smin, s_max=smax)


def get_image_coordinate_grid_nib(image: nib.Nifti1Image):
    img_header = image.header
    img_data = image.get_fdata()
    img_affine = image.affine
    (x, y, z) = image.shape

    label = []
    coordinates = []

    X = np.linspace(0, x - 1, x)
    Y = np.linspace(0, y - 1, y)
    Z = np.linspace(0, z - 1, z)


    points = np.meshgrid(X, Y, Z, indexing='ij')
    points = np.stack(points).transpose(1, 2, 3, 0).reshape(-1, 3)
    coordinates = list(nib.affines.apply_affine(img_affine, points))
    label = list(img_data.flatten())

    # convert to numpy array
    coordinates_array = np.array(coordinates, dtype=np.float32)
    label_array = np.array(label, dtype=np.float32)
    label_array_unnorm = label_array

    label_array = np.clip(label_array, np.quantile(label_array, 0.001), np.quantile(label_array, 0.999))
    label_array = (label_array - np.min(label_array)) / (np.max(label_array) - np.min(label_array))

    def min_max_scale(X, s_min, s_max):
        x_min, x_max = X.min(), X.max()
        return (X - x_min) / (x_max - x_min) * (s_max - s_min) + s_min

    coordinates_arr_norm = min_max_scale(X=coordinates_array, s_min=-1, s_max=1)

    image_dict = {
        'affine': torch.tensor(img_affine),
        'origin': torch.tensor(np.array([0])),
        'spacing': torch.tensor(np.array(img_header["pixdim"][1:4])),
        'dim': torch.tensor(np.array([x, y, z])),
        'intensity': torch.tensor(label_array_unnorm, dtype=torch.float32).view(-1, 1),
        'intensity_norm': torch.tensor(label_array, dtype=torch.float32).view(-1, 1),
        'coordinates': torch.tensor(coordinates_array, dtype=torch.float32),
        'coordinates_norm': torch.tensor(coordinates_arr_norm, dtype=torch.float32),
    }

    return image_dict
