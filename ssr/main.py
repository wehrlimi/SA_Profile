import argparse
import yaml
import nibabel as nib
import torch
import numpy as np
from nibabel import Nifti1Image
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import MultiDataset
from tqdm import tqdm
import time
import os
from typing import Tuple, List, Dict, Any

from network.inr import WIREMLP, ReLUMLP
from network.pos_encoders import FourierFeatPosEncoder, NeRFPosEncoder

# Try to import Muon optimizer
try:
    from muon import SingleDeviceMuonWithAuxAdam
    MUON_AVAILABLE = True
except ImportError:
    print("Warning: Muon optimizer not available. Install with: pip install git+https://github.com/KellerJordan/Muon")
    MUON_AVAILABLE = False
    SingleDeviceMuonWithAuxAdam = None


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_muon_optimizer(model, muon_lr: float, aux_lr: float, 
                        muon_weight_decay: float = 0.0, 
                        aux_weight_decay: float = 0.0):
    """
    Setup Muon optimizer for ssr models.
    Muon is used for hidden layer weight matrices (2D tensors).
    Adam is used for biases and other parameters.
    """
    if not MUON_AVAILABLE:
        raise ImportError("Muon optimizer is not available. Install with: pip install git+https://github.com/KellerJordan/Muon")
    
    muon_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        is_muon_target = False
        
        # Check if it's a weight matrix (2D tensor) from a hidden layer
        if param.ndim == 2:
            # Parse layer index from name
            # Examples: 'layers.1.freqs.weight', 'layers.1.scale.weight', 'layers.1.linear.weight'
            parts = name.split('.')
            if len(parts) >= 2 and parts[0] == 'layers':
                try:
                    layer_idx = int(parts[1])
                    num_layers = len(model.layers)
                    
                    # Hidden layers are those between first and last
                    # (layer_idx 0 is first, layer_idx num_layers-1 is last)
                    if 0 < layer_idx < num_layers - 1:
                        is_muon_target = True
                except (ValueError, IndexError):
                    pass
        
        if is_muon_target:
            muon_params.append(param)
        else:
            other_params.append(param)
    
    if muon_params:
        # Ensure learning rates are floats, not lists
        muon_lr = float(muon_lr) if not isinstance(muon_lr, (int, float)) else muon_lr
        aux_lr = float(aux_lr) if not isinstance(aux_lr, (int, float)) else aux_lr
        
        param_groups = [
            dict(params=muon_params, use_muon=True, lr=muon_lr, weight_decay=muon_weight_decay),
            dict(params=other_params, use_muon=False, lr=aux_lr, betas=(0.9, 0.999), weight_decay=aux_weight_decay)
        ]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
        print(f"Muon optimizer configured. Muon params: {len(muon_params)}, Other params: {len(other_params)}")
        return optimizer
    else:
        # Fallback to Adam if no Muon params found
        print("Warning: No Muon params found, using Adam")
        return torch.optim.Adam(model.parameters(), lr=aux_lr)


def create_model(config: Dict[str, Any], device: torch.device):
    """Create the INR model based on configuration."""
    model_cfg = config['model']
    model_type = model_cfg['type']
    
    if model_type == 'WIREMLP':
        model = WIREMLP(
            in_size=model_cfg['in_size'],
            out_size=model_cfg['out_size'],
            hidden_size=model_cfg['hidden_size'],
            num_layers=model_cfg['num_layers'],
            wire_omega=model_cfg['wire_omega'],
            wire_sigma=model_cfg.get('wire_sigma', 20.0)
        )
    elif model_type == 'ReLUMLP':
        model = ReLUMLP(
            in_size=model_cfg['in_size'],
            out_size=model_cfg['out_size'],
            hidden_size=model_cfg['hidden_size'],
            num_layers=model_cfg['num_layers']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    print(f"Created {model_type} model with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Model: {model}")
    return model


def create_optimizer(model, config: Dict[str, Any]):
    """Create optimizer based on configuration."""
    opt_cfg = config['optimizer']
    opt_type = opt_cfg['type']
    
    if opt_type == 'muon':
        if not MUON_AVAILABLE:
            print("Warning: Muon not available, falling back to Adam")
            opt_type = 'adam'
        else:
            # Ensure learning rates are floats
            muon_lr = opt_cfg['muon_lr']
            aux_lr = opt_cfg['lr']
            
            if isinstance(muon_lr, str):
                muon_lr = float(muon_lr)
            elif not isinstance(muon_lr, (int, float)):
                muon_lr = float(muon_lr)
            
            if isinstance(aux_lr, str):
                aux_lr = float(aux_lr)
            elif not isinstance(aux_lr, (int, float)):
                aux_lr = float(aux_lr)
            
            return setup_muon_optimizer(
                model,
                muon_lr=muon_lr,
                aux_lr=aux_lr,
                muon_weight_decay=opt_cfg.get('muon_weight_decay', 0.0),
                aux_weight_decay=opt_cfg.get('muon_aux_weight_decay', 0.0)
            )
    
    if opt_type == 'adam':
        # Ensure learning rate and other parameters are proper types
        lr = opt_cfg['lr']
        if isinstance(lr, str):
            lr = float(lr)
        elif not isinstance(lr, (int, float)):
            lr = float(lr)
        
        weight_decay = opt_cfg.get('weight_decay', 0.0)
        if isinstance(weight_decay, str):
            weight_decay = float(weight_decay)
        elif not isinstance(weight_decay, (int, float)):
            weight_decay = float(weight_decay)
        
        betas = opt_cfg.get('betas', [0.9, 0.999])
        if isinstance(betas, list):
            betas = tuple(float(b) if isinstance(b, str) else float(b) for b in betas)
        
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")


def create_scheduler(optimizer, config: Dict[str, Any]):
    """Create learning rate scheduler based on configuration."""
    sched_cfg = config.get('scheduler', {})
    sched_type = sched_cfg.get('type')
    
    if sched_type is None or sched_type == 'none':
        return None
    elif sched_type == 'cosine':
        # Ensure T_max and eta_min are numeric types
        T_max = sched_cfg.get('T_max', 100)
        eta_min = sched_cfg.get('eta_min', 1e-6)
        
        # Convert to proper types if they're strings
        if isinstance(T_max, str):
            T_max = int(float(T_max))
        elif not isinstance(T_max, (int, float)):
            T_max = int(T_max)
        
        if isinstance(eta_min, str):
            eta_min = float(eta_min)
        elif not isinstance(eta_min, (int, float)):
            eta_min = float(eta_min)
        
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )
    elif sched_type == 'step':
        step_size = sched_cfg.get('step_size', 50)
        gamma = sched_cfg.get('gamma', 0.9)
        
        # Convert to proper types if they're strings
        if isinstance(step_size, str):
            step_size = int(float(step_size))
        elif not isinstance(step_size, (int, float)):
            step_size = int(step_size)
        
        if isinstance(gamma, str):
            gamma = float(gamma)
        elif not isinstance(gamma, (int, float)):
            gamma = float(gamma)
        
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")


def create_pos_encoder(config: Dict[str, Any], coord_size: int):
    """Create positional encoder if specified in config."""
    pos_enc_type = config['model'].get('pos_encoder')
    
    if pos_enc_type is None:
        return None
    elif pos_enc_type == 'FourierFeat':
        # You may want to add these to config
        return FourierFeatPosEncoder(coord_size=coord_size, freq_num=128, freq_scale=1.0)
    elif pos_enc_type == 'NeRF':
        # You may want to add these to config
        return NeRFPosEncoder(coord_size=coord_size, freq_num=10, freq_scale=1.0)
    else:
        raise ValueError(f"Unknown positional encoder type: {pos_enc_type}")


@torch.no_grad()
def sample_at_resolution_matching_axial(model, dataset, resolution: Tuple[int, ...], 
                                       batch_size: int, device: torch.device, pos_encoder=None):
    """Sample matching the axial image's orientation exactly using axial-specific coordinate bounds."""
    # Get axial-specific information for voxel grid and affine
    original_affine = dataset.get_axial_affine()
    original_shape = dataset.get_axial_shape()  # (x, y, z) as numpy array
    
    # Convert to numpy if needed
    if isinstance(original_affine, torch.Tensor):
        original_affine = original_affine.numpy()
    if isinstance(original_shape, torch.Tensor):
        original_shape = original_shape.numpy()
    
    # Create sampling grid in voxel space matching axial image
    x = np.linspace(0, original_shape[0] - 1, resolution[0])
    y = np.linspace(0, original_shape[1] - 1, resolution[1])
    z = np.linspace(0, original_shape[2] - 1, resolution[2])
    
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], axis=1)
    
    # Apply affine to get world coordinates (same as axial image uses)
    world_coords = nib.affines.apply_affine(original_affine, points)
    
    # CRITICAL: Normalize using the SAME bounds as training (combined from all views)
    # Get the ORIGINAL world coordinate bounds used for normalization during training
    # These are the min/max values across all three views BEFORE normalization
    # Note: These are SCALAR values (not per-axis), matching how _process() normalizes
    coords_min, coords_max = dataset.get_original_world_coordinate_bounds()
    
    # Check if axial world coordinates are within the normalization bounds
    ax_world_min = world_coords.min(axis=0)
    ax_world_max = world_coords.max(axis=0)
    print(f"Sampling world coord range: min={ax_world_min}, max={ax_world_max}")
    print(f"Normalization bounds: min={coords_min}, max={coords_max}")
    
    # Normalize to [-1, 1] range using the SAME bounds as training
    # norm_grid uses: (X - x_min)/(x_max - x_min) * (s_max - s_min) + s_min
    # where s_min=-1, s_max=1, so: (X - x_min)/(x_max - x_min) * 2 - 1
    coords_norm = (world_coords - coords_min) / (coords_max - coords_min) * 2.0 - 1.0
    
    # Check if normalized coordinates are within expected range
    coords_norm_min = coords_norm.min(axis=0)
    coords_norm_max = coords_norm.max(axis=0)
    print(f"Normalized coord range: min={coords_norm_min}, max={coords_norm_max}")
    if np.any(coords_norm < -1.1) or np.any(coords_norm > 1.1):
        print(f"WARNING: Some normalized coordinates are outside [-1, 1] range!")
        print(f"  This may cause noise at boundaries.")
    
    coords_norm = torch.tensor(coords_norm, dtype=torch.float32).to(device)
    
    # Predict
    predictions_list = []
    for i in range(0, len(coords_norm), batch_size):
        batch = coords_norm[i:i + batch_size]
        if pos_encoder:
            batch = pos_encoder.forward(batch)
        pred = model.forward(batch)
        predictions_list.append(pred)
    
    predictions = torch.cat(predictions_list, dim=0)
    
    # Extract axial channel (channel 0 in output corresponds to axial)
    pred_axial = predictions[:, 0].reshape(resolution)
    
    return pred_axial, original_affine, original_shape


def train_and_save(ax_path: str, cor_path: str, sag_path: str, 
                   output_path: str, config: Dict[str, Any] = None,
                   config_path: str = 'config.yaml', cuda_device: str = None,
                   patient_id: str = None, study_id: str = None):
    """
    Train an INR model on the given three views and save the result.
    
    Args:
        ax_path: Path to axial .nii.gz file
        cor_path: Path to coronal .nii.gz file
        sag_path: Path to sagittal .nii.gz file
        output_path: Path where the final result should be saved
        config: Optional config dictionary (if None, will load from config_path)
        config_path: Path to config YAML file (used if config is None)
        cuda_device: Optional CUDA device override
        patient_id: Optional patient ID for naming (uses old_main.py convention if provided)
        study_id: Optional study ID for naming (uses old_main.py convention if provided)
    
    Returns:
        output_path: Path to saved file
    """
    # Load configuration if not provided
    if config is None:
        config = load_config(config_path)
    
    # Override CUDA device if specified
    if cuda_device is not None:
        config['cuda_device'] = cuda_device
    
    # Set CUDA device
    if config.get('cuda_device') and config['cuda_device'] != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config['cuda_device'])
    
    # Setup device
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Load dataset with provided paths
    dataset = MultiDataset(ax_path, cor_path, sag_path)
    print(f"Dataset loaded. Training with {config['training']['batch_size']} points per step.")
    
    # Create model
    inr = create_model(config, device)
    
    # Create positional encoder if specified
    pos_encoder = create_pos_encoder(config, coord_size=config['model']['in_size'])
    
    # Create optimizer
    optimizer = create_optimizer(inr, config)
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config)
    if scheduler:
        print(f"Using {config['scheduler']['type']} scheduler")
    
    # Setup logging
    log_cfg = config.get('logging', {})
    run_name = time.strftime("run_%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_cfg.get('log_dir', 'tb_logs'), run_name)
    os.makedirs(log_dir, exist_ok=True)
    logger = SummaryWriter(log_dir=log_dir)
    
    # Create output directory
    save_dir = log_cfg.get('save_images_dir', './images_dir')
    os.makedirs(save_dir, exist_ok=True)
    
    # Create dataloader
    train_cfg = config['training']
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=train_cfg.get('num_workers', 16)
    )
    
    # Training loop
    num_epochs = train_cfg['num_epochs']
    log_interval = log_cfg.get('log_interval', 10)
    output_resolution = config['output']['resolution']
    
    for epoch in tqdm(range(num_epochs + 1), desc="Epochs"):
        for i, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)):
            inr.train()
            optimizer.zero_grad()
            
            coords, values = data
            coords = coords.to(device)
            values = values.to(device)
            
            # Separate by view
            ax_mask = (values[:, 2] != -1.0)
            values_ax = values[ax_mask, 2].reshape(-1, 1)
            coords_ax = coords[ax_mask, :]
            
            cor_mask = (values[:, 1] != -1.0)
            values_cor = values[cor_mask, 1].reshape(-1, 1)
            coords_cor = coords[cor_mask, :]
            
            sag_mask = (values[:, 0] != -1.0)
            values_sag = values[sag_mask, 0].reshape(-1, 1)
            coords_sag = coords[sag_mask, :]
            
            # Concatenate all views
            coords = torch.cat((coords_ax, coords_cor, coords_sag), dim=0)

            # Apply positional encoding if used
            if pos_encoder:
                coords = pos_encoder.forward(coords)
            
            # Forward pass
            out = inr.forward(coords)
            out_ax = out[:len(coords_ax), 0:1]
            out_cor = out[len(coords_ax):len(coords_ax)+len(coords_cor), 1:2]
            out_sag = out[len(coords_ax)+len(coords_cor):, 2:3]
            
            # Compute loss
            loss = (nn.functional.mse_loss(out_ax, values_ax) +
                   nn.functional.mse_loss(out_cor, values_cor) +
                   nn.functional.mse_loss(out_sag, values_sag))

            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Logging
            global_step = epoch * len(dataloader) + i
            logger.add_scalar('loss', loss.item(), global_step)
            
            # Log learning rates
            if isinstance(optimizer, SingleDeviceMuonWithAuxAdam) and MUON_AVAILABLE:
                logger.add_scalar('lr/muon', optimizer.param_groups[0]['lr'], global_step)
                logger.add_scalar('lr/aux', optimizer.param_groups[1]['lr'], global_step)
            else:
                logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)
        
        # Update scheduler
        if scheduler:
            scheduler.step()
            # Fix: Ensure learning rates are floats for Muon optimizer
            if MUON_AVAILABLE and isinstance(optimizer, SingleDeviceMuonWithAuxAdam):
                for group in optimizer.param_groups:
                    if isinstance(group['lr'], (list, tuple)):
                        group['lr'] = float(group['lr'][0]) if len(group['lr']) > 0 else group['lr']
                    elif not isinstance(group['lr'], (int, float)):
                        group['lr'] = float(group['lr'])
        
        # Periodic evaluation and saving
        if epoch % log_interval == 0 or epoch == num_epochs:
            print(f"\nSampling at epoch {epoch}")
            # Sample using axial-specific coordinate bounds to match orientation
            pred_axial, original_affine, original_shape = sample_at_resolution_matching_axial(
                inr, dataset, resolution=output_resolution,
                batch_size=train_cfg['batch_size'], device=device, pos_encoder=pos_encoder
            )
            
            print(f"Original axial shape: {original_shape}")
            
            # Compute target affine transformation for new resolution
            # Scale the affine to account for resolution change
            if isinstance(original_shape, torch.Tensor):
                original_shape = original_shape.numpy()
            if isinstance(original_affine, torch.Tensor):
                original_affine = original_affine.numpy()
            
            # The affine matrix maps voxel coordinates to world coordinates
            # When we change resolution, we need to scale the rotation/scaling part
            # (first 3x3) to account for the new voxel size
            # The translation part (last column) should remain the same
            
            # Calculate scaling factors: new_voxel_size / old_voxel_size
            # Since we're sampling the same physical space with different resolution,
            # voxel size scales inversely with resolution
            scale_factors = original_shape.astype(float) / np.array(output_resolution, dtype=float)
            
            # Scale each column of the rotation/scaling part (first 3x3) by the corresponding scale factor
            # This properly accounts for the change in voxel spacing
            target_affine = original_affine.copy()
            for i in range(3):
                target_affine[:3, i] = original_affine[:3, i] * scale_factors[i]
            
            # Validate affine matrix (check for NaN, Inf, or invalid values)
            if not np.isfinite(target_affine).all():
                print(f"Warning: Invalid values in target_affine! Using original affine instead.")
                target_affine = original_affine.copy()
            
            # Ensure affine is 4x4
            if target_affine.shape != (4, 4):
                print(f"Warning: target_affine has wrong shape {target_affine.shape}! Using original affine instead.")
                target_affine = original_affine.copy()
            
            # Denormalization - get original intensity range from axial image
            axial_min, axial_max = dataset.get_axial_intensity_bounds()
            
            # Convert to tensors on the same device as prediction
            axial_min = torch.tensor(axial_min, dtype=torch.float32, device=device)
            axial_max = torch.tensor(axial_max, dtype=torch.float32, device=device)
            
            # Denormalize the prediction
            # The model outputs normalized [0, 1] values (from intensity_norm in utils.py)
            pred_axial_denorm = pred_axial * (axial_max - axial_min) + axial_min
            
            # Convert to numpy and ensure proper shape and dtype
            pred_data = pred_axial_denorm.detach().cpu().numpy()
            pred_data = pred_data.astype(np.float32)
            
            # Ensure data shape matches output resolution
            if pred_data.shape != tuple(output_resolution):
                print(f"Warning: Data shape {pred_data.shape} doesn't match resolution {output_resolution}. Reshaping...")
                pred_data = pred_data.reshape(output_resolution)
            
            # Validate data (check for NaN, Inf)
            if not np.isfinite(pred_data).all():
                print(f"Warning: Invalid values (NaN/Inf) in prediction data! Replacing with zeros.")
                pred_data = np.nan_to_num(pred_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Save with original axial affine (scaled for resolution)
            # No flipping needed - orientation already matches axial image
            nib_img = Nifti1Image(pred_data, target_affine)
            
            # Only save final epoch result when called programmatically
            if epoch == num_epochs:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                nib.save(nib_img, output_path)
                print(f"Saved prediction to {output_path}")
                print(f"Sampled at resolution: {output_resolution}")
    
    logger.close()
    print("Training completed!")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Train SSR INR model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration YAML file')
    parser.add_argument('--cuda_device', type=str, default=None,
                       help='Override CUDA device (e.g., "0", "1", "2")')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override CUDA device if specified
    if args.cuda_device is not None:
        config['cuda_device'] = args.cuda_device
    
    # Set CUDA device
    if config.get('cuda_device') and config['cuda_device'] != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config['cuda_device'])
    
    # Setup device
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Load dataset
    data_cfg = config['data']
    dataset = MultiDataset(data_cfg['ax_path'], data_cfg['cor_path'], data_cfg['sag_path'])
    print(f"Dataset loaded. Training with {config['training']['batch_size']} points per step.")
    
    # Create model
    inr = create_model(config, device)
    
    # Create positional encoder if specified
    pos_encoder = create_pos_encoder(config, coord_size=config['model']['in_size'])
    
    # Create optimizer
    optimizer = create_optimizer(inr, config)
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config)
    if scheduler:
        print(f"Using {config['scheduler']['type']} scheduler")
    
    # Setup logging
    log_cfg = config.get('logging', {})
    run_name = time.strftime("run_%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_cfg.get('log_dir', 'tb_logs'), run_name)
    os.makedirs(log_dir, exist_ok=True)
    logger = SummaryWriter(log_dir=log_dir)
    
    # Create output directory
    save_dir = log_cfg.get('save_images_dir', './images_dir')
    os.makedirs(save_dir, exist_ok=True)
    
    # Create dataloader
    train_cfg = config['training']
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=train_cfg.get('num_workers', 16)
    )
    
    # Training loop
    num_epochs = train_cfg['num_epochs']
    log_interval = log_cfg.get('log_interval', 10)
    output_resolution = config['output']['resolution']
    
    for epoch in tqdm(range(num_epochs + 1), desc="Epochs"):
        for i, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)):
            inr.train()
            optimizer.zero_grad()
            
            coords, values = data
            coords = coords.to(device)
            values = values.to(device)
            
            # Separate by view
            ax_mask = (values[:, 2] != -1.0)
            values_ax = values[ax_mask, 2].reshape(-1, 1)
            coords_ax = coords[ax_mask, :]
            
            cor_mask = (values[:, 1] != -1.0)
            values_cor = values[cor_mask, 1].reshape(-1, 1)
            coords_cor = coords[cor_mask, :]
            
            sag_mask = (values[:, 0] != -1.0)
            values_sag = values[sag_mask, 0].reshape(-1, 1)
            coords_sag = coords[sag_mask, :]
            
            # Concatenate all views
            coords = torch.cat((coords_ax, coords_cor, coords_sag), dim=0)

            # Apply positional encoding if used
            if pos_encoder:
                coords = pos_encoder.forward(coords)
            
            # Forward pass
            out = inr.forward(coords)
            out_ax = out[:len(coords_ax), 0:1]
            out_cor = out[len(coords_ax):len(coords_ax)+len(coords_cor), 1:2]
            out_sag = out[len(coords_ax)+len(coords_cor):, 2:3]
            
            # Compute loss
            loss = (nn.functional.mse_loss(out_ax, values_ax) +
                   nn.functional.mse_loss(out_cor, values_cor) +
                   nn.functional.mse_loss(out_sag, values_sag))

            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Logging
            global_step = epoch * len(dataloader) + i
            logger.add_scalar('loss', loss.item(), global_step)
            
            # Log learning rates
            if isinstance(optimizer, SingleDeviceMuonWithAuxAdam) and MUON_AVAILABLE:
                logger.add_scalar('lr/muon', optimizer.param_groups[0]['lr'], global_step)
                logger.add_scalar('lr/aux', optimizer.param_groups[1]['lr'], global_step)
            else:
                logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)
        
        # Update scheduler
        if scheduler:
            scheduler.step()
            # Fix: Ensure learning rates are floats for Muon optimizer
            if MUON_AVAILABLE and isinstance(optimizer, SingleDeviceMuonWithAuxAdam):
                for group in optimizer.param_groups:
                    if isinstance(group['lr'], (list, tuple)):
                        group['lr'] = float(group['lr'][0]) if len(group['lr']) > 0 else group['lr']
                    elif not isinstance(group['lr'], (int, float)):
                        group['lr'] = float(group['lr'])
        
        # Periodic evaluation and saving
        if epoch % log_interval == 0 or epoch == num_epochs:
            print(f"\nSampling at epoch {epoch}")
            # Sample using axial-specific coordinate bounds to match orientation
            pred_axial, original_affine, original_shape = sample_at_resolution_matching_axial(
                inr, dataset, resolution=output_resolution,
                batch_size=train_cfg['batch_size'], device=device, pos_encoder=pos_encoder
            )
            
            print(f"Original axial shape: {original_shape}")
            
            # Compute target affine transformation for new resolution
            # Scale the affine to account for resolution change
            if isinstance(original_shape, torch.Tensor):
                original_shape = original_shape.numpy()
            if isinstance(original_affine, torch.Tensor):
                original_affine = original_affine.numpy()
            
            # The affine matrix maps voxel coordinates to world coordinates
            # When we change resolution, we need to scale the rotation/scaling part
            # (first 3x3) to account for the new voxel size
            # The translation part (last column) should remain the same
            
            # Calculate scaling factors: new_voxel_size / old_voxel_size
            # Since we're sampling the same physical space with different resolution,
            # voxel size scales inversely with resolution
            scale_factors = original_shape.astype(float) / np.array(output_resolution, dtype=float)
            
            # Scale each column of the rotation/scaling part (first 3x3) by the corresponding scale factor
            # This properly accounts for the change in voxel spacing
            target_affine = original_affine.copy()
            for i in range(3):
                target_affine[:3, i] = original_affine[:3, i] * scale_factors[i]
            
            # Validate affine matrix (check for NaN, Inf, or invalid values)
            if not np.isfinite(target_affine).all():
                print(f"Warning: Invalid values in target_affine! Using original affine instead.")
                target_affine = original_affine.copy()
            
            # Ensure affine is 4x4
            if target_affine.shape != (4, 4):
                print(f"Warning: target_affine has wrong shape {target_affine.shape}! Using original affine instead.")
                target_affine = original_affine.copy()
            
            # Denormalization - get original intensity range from axial image
            axial_min, axial_max = dataset.get_axial_intensity_bounds()
            
            # Convert to tensors on the same device as prediction
            axial_min = torch.tensor(axial_min, dtype=torch.float32, device=device)
            axial_max = torch.tensor(axial_max, dtype=torch.float32, device=device)
            
            # Denormalize the prediction
            # The model outputs normalized [0, 1] values (from intensity_norm in utils.py)
            pred_axial_denorm = pred_axial * (axial_max - axial_min) + axial_min
            
            # Convert to numpy and ensure proper shape and dtype
            pred_data = pred_axial_denorm.detach().cpu().numpy()
            pred_data = pred_data.astype(np.float32)
            
            # Ensure data shape matches output resolution
            if pred_data.shape != tuple(output_resolution):
                print(f"Warning: Data shape {pred_data.shape} doesn't match resolution {output_resolution}. Reshaping...")
                pred_data = pred_data.reshape(output_resolution)
            
            # Validate data (check for NaN, Inf)
            if not np.isfinite(pred_data).all():
                print(f"Warning: Invalid values (NaN/Inf) in prediction data! Replacing with zeros.")
                pred_data = np.nan_to_num(pred_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Save with original axial affine (scaled for resolution)
            # No flipping needed - orientation already matches axial image
            nib_img = Nifti1Image(pred_data, target_affine)
            save_path = os.path.join(
                save_dir,
                f'res_{output_resolution[0]}_{output_resolution[1]}_{output_resolution[2]}_epoch_{epoch}_contr_0.nii.gz'
            )
            nib.save(nib_img, save_path)
            print(f"Saved prediction to {save_path}")
            print(f"Sampled at resolution: {output_resolution}")
    
    logger.close()
    print("Training completed!")


if __name__ == '__main__':
    main()
