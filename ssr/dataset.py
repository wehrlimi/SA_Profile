import torch
import os
import nibabel as nib
import numpy as np
from utils import get_image_coordinate_grid_nib, norm_grid
from torch.utils.data import Dataset
#import open3d as o3d
from sklearn.neighbors import NearestNeighbors


class MultiDataset(Dataset):
    def __init__(self, ax_dir: str='', cor_dir:str='', sag_dir:str=''):
        super().__init__()

        self.ax_dir = ax_dir
        self.cor_dir = cor_dir
        self.sag_dir = sag_dir

        self.data = []
        self.label = []
        self.len = 0
        self._process()
        #self._remove_close_points()

        self.min_values, _ = torch.min(self.data, dim=0)
        self.max_values, _ = torch.max(self.data, dim=0)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]

        return data, label

    def get_coordinates(self):
        return self.coordinates

    def get_affine(self):
        return self.affine

    def get_dim(self):
        return self.dim

    def get_minmax_values_original(self):
        ax_dict = get_image_coordinate_grid_nib(nib.load(self.ax_dir))
        cor_dict = get_image_coordinate_grid_nib(nib.load(self.cor_dir))
        sag_dir = get_image_coordinate_grid_nib(nib.load(self.sag_dir))

        labels_ax_or = ax_dict['intensity']
        labels_cor_or = cor_dict['intensity']
        labels_sag_or = sag_dir['intensity']

        min_values = [min(labels_ax_or), min(labels_cor_or), min(labels_sag_or)]
        max_values = [max(labels_ax_or), max(labels_cor_or), max(labels_sag_or)]

        return min_values, max_values
    
    def get_axial_coordinate_bounds(self):
        """Get coordinate bounds from axial image only (world coordinates)."""
        ax_dict = get_image_coordinate_grid_nib(nib.load(self.ax_dir))
        ax_coords = ax_dict['coordinates'].numpy()
        return ax_coords.min(axis=0), ax_coords.max(axis=0)
    
    def get_axial_affine(self):
        """Get affine matrix from axial image."""
        ax_dict = get_image_coordinate_grid_nib(nib.load(self.ax_dir))
        return ax_dict['affine']
    
    def get_axial_shape(self):
        """Get shape (dimensions) from axial image."""
        ax_dict = get_image_coordinate_grid_nib(nib.load(self.ax_dir))
        return ax_dict['dim'].numpy()
    
    def get_axial_intensity_bounds(self):
        """Get intensity min/max from axial image only (original, unnormalized values)."""
        ax_dict = get_image_coordinate_grid_nib(nib.load(self.ax_dir))
        labels_ax_or = ax_dict['intensity']
        return float(labels_ax_or.min()), float(labels_ax_or.max())
    
    def get_original_world_coordinate_bounds(self):
        """Get the original world coordinate bounds used for normalization during training.
        These are the min/max values across all three views (ax, cor, sag) before normalization.
        Must match exactly how _process() computes min_c and max_c (scalar values, not per-axis).
        """
        ax_dict = get_image_coordinate_grid_nib(nib.load(self.ax_dir))
        cor_dict = get_image_coordinate_grid_nib(nib.load(self.cor_dir))
        sag_dict = get_image_coordinate_grid_nib(nib.load(self.sag_dir))
        
        data_ax = ax_dict['coordinates']
        data_cor = cor_dict['coordinates']
        data_sag = sag_dict['coordinates']
        
        # Match exactly how _process() computes bounds (scalar min/max, not per-axis)
        min1, max1 = data_ax.min(), data_ax.max()
        min2, max2 = data_cor.min(), data_cor.max()
        min3, max3 = data_sag.min(), data_sag.max()
        
        # Get the combined min/max across all three views (same as in _process)
        # min_c is the minimum of all minimums, max_c is the maximum of all maximums
        min_c = np.min(np.array([min1, min2, min3]))
        max_c = np.max(np.array([max1, max2, max3]))
        
        return min_c, max_c
    
    def _process(self):
        ax_dict = get_image_coordinate_grid_nib(nib.load(self.ax_dir))
        cor_dict = get_image_coordinate_grid_nib(nib.load(self.cor_dir))
        sag_dict = get_image_coordinate_grid_nib(nib.load(self.sag_dir))

        data_ax = ax_dict['coordinates']
        data_cor = cor_dict['coordinates']
        data_sag = sag_dict['coordinates']

        min1, max1 = data_ax.min(), data_ax.max()
        min2, max2 = data_cor.min(), data_cor.max()
        min3, max3 = data_sag.min(), data_sag.max()


        min_c, max_c = np.min(np.array([min1, min2, min3])), np.max(np.array([max1, max2, max3]))

        data_ax = norm_grid(data_ax, xmin=min_c, xmax=max_c)
        data_cor = norm_grid(data_cor, xmin=min_c, xmax=max_c)
        data_sag = norm_grid(data_sag, xmin=min_c, xmax=max_c)

        labels_ax = ax_dict['intensity_norm']
        labels_cor = cor_dict['intensity_norm']
        labels_sag = sag_dict['intensity_norm']

        labels_ax_stack = torch.cat((torch.ones(labels_ax.shape) * -1, torch.ones(labels_ax.shape) * -1, labels_ax), dim=1)
        labels_cor_stack = torch.cat((torch.ones(labels_cor.shape) * -1, labels_cor, torch.ones(labels_cor.shape) * -1), dim=1)
        labels_sag_stack = torch.cat((labels_sag, torch.ones(labels_sag.shape) * -1, torch.ones(labels_sag.shape) * -1), dim=1)

        # Inference grid (from axial!)
        self.coordinates = ax_dict['coordinates_norm']
        self.affine = ax_dict['affine']
        self.dim = ax_dict['dim']

        self.data = torch.cat((data_ax, data_cor, data_sag), dim=0)
        self.label = torch.cat((labels_ax_stack, labels_cor_stack, labels_sag_stack), dim=0)
        self.len = len(self.label)


class InferDataset(Dataset):
    def __init__(self, grid):
        super(InferDataset, self,).__init__()
        self.grid = grid

    def __len__(self):
        return len(self.grid)

    def __getitem__(self, idx):
        data = self.grid[idx]
        return data
