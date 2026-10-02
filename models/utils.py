import os
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def kaiming_weight_init(m):
    classname = m.__class__.__name__
    if 'Conv2d' in classname or 'ConvTranspose2d' in classname:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif 'Linear' in classname:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()


def avg_coordinate(heatmap, center, radius):
    size = heatmap.shape[-1]
    upper_left, lower_right = center - radius, center + radius + 1
    upper_left_cropped, lower_right_cropped = np.maximum(upper_left, 0), np.minimum(lower_right, size)
    
    window = heatmap[upper_left_cropped[1]:lower_right_cropped[1], upper_left_cropped[0]:lower_right_cropped[0]]
    padding_lower_right, padding_upper_left = lower_right - lower_right_cropped, upper_left_cropped - upper_left
    
    window = F.pad(window.unsqueeze(0),
                   (padding_upper_left[0], padding_lower_right[0], padding_upper_left[1], padding_lower_right[1]),
                   mode='constant', value=0).squeeze(0)
    window = window / max(window.sum(), 1e-10)
    
    x_grid, y_grid = torch.meshgrid(torch.arange(0, window.shape[0]), torch.arange(0, window.shape[1]), indexing='xy')
    return upper_left + np.array([torch.sum(x_grid * window), torch.sum(y_grid * window)])


def extract_keypoints(heatmap_batch, method="argmax", weight_iterations=1):
    size = heatmap_batch.size()[-1]
    keypoint_batch = []
    if method == "argmax":
        for heatmaps in heatmap_batch:
            keypoints = []
            for heatmap in heatmaps:
                max_index = torch.argmax(heatmap).item()
                keypoints.append(np.array([max_index % size, max_index // size]))
            keypoint_batch.append(np.array(keypoints))
    elif method == "weighted":
        window_radius = int(round(size * 0.25))
        for heatmaps in heatmap_batch:
            keypoints = []
            for heatmap in heatmaps:
                max_index = torch.argmax(heatmap).item()
                point = np.array([max_index % size, max_index // size])

                for _ in range(weight_iterations):
                    point = avg_coordinate(heatmap, point.round().astype(int), window_radius)
                    point = np.maximum(0, np.minimum(size, point))
                
                keypoints.append(point)
                assert not np.any(np.isnan(keypoints[-1]))

            keypoint_batch.append(np.array(keypoints))
    else:
        raise ValueError(
            f"Invalid Value for arg 'method': '{method} \n Supported methods: 'argmax', 'weighted'"
        )
    
    return np.array(keypoint_batch)


def softmax2d(x, log=False):
    return (F.log_softmax if log else F.softmax)(x.view(*x.shape[:-2], -1), dim=-1).view(*x.shape)


def load_model(path, device):
    # PyTorch 2.6+ defaults to weights_only=True, but our checkpoints contain OmegaConf objects
    # Since these are our own checkpoints, we trust them and set weights_only=False
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_class = getattr(importlib.import_module(checkpoint["module_name"]), checkpoint["class_name"])

    # Convert OmegaConf objects to regular Python types if needed
    init_args = checkpoint["init_args"]
    try:
        from omegaconf import OmegaConf
        # Check if init_args contains OmegaConf objects
        if OmegaConf.is_config(init_args):
            init_args = OmegaConf.to_container(init_args, resolve=True)
    except ImportError:
        # OmegaConf not available, assume init_args is already in correct format
        pass
    
    model = model_class(**init_args)
    model.load_state_dict(checkpoint["model_state_dict"])
    # Move model to the specified device
    model = model.to(device)
    return model


def save_model(path, model, name, epoch_idx):
    torch.save({'epoch' : epoch_idx,
                'module_name' : model.__class__.__module__,
                'class_name' : model.__class__.__name__,
                'name' : name,
                'init_args' : model.init_args,
                'model_state_dict' : model.state_dict()},
                os.path.join(path, f"checkpoint_{name}_{epoch_idx}.pt"))

