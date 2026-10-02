"""
Training script for both sagittal and axial models.
Can train them separately or sequentially.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import numpy as np
import torch
import torchvision
from torchvision.utils import make_grid

# Add project root to path to enable imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf
from data.dataset_classes import SagittalBoundsDataset, AxialLandmarkDataset


def train_sagittal_model(cfg):
    """Train the sagittal boundary detection model"""
    logging.info("=" * 80)
    logging.info("TRAINING SAGITTAL BOUNDS MODEL")
    logging.info("=" * 80)

    # Update config for sagittal model
    cfg_sagittal = cfg.copy()
    cfg_sagittal.program.result_path = os.path.join(cfg.program.result_path, "sagittal_bounds")
    cfg_sagittal.program.checkpoint_path = os.path.join(cfg.program.checkpoint_path, "sagittal_bounds")
    # out_channels will be determined from dataset.get_num_landmarks()

    # Note: dataset class is passed directly to train_model_with_dataset

    os.makedirs(cfg_sagittal.program.result_path, exist_ok=True)
    os.makedirs(cfg_sagittal.program.checkpoint_path, exist_ok=True)

    # Train using existing train_model function
    # You'll need to modify train_model to accept dataset_class parameter
    best_model_path = train_model_with_dataset(cfg_sagittal, SagittalBoundsDataset)

    logging.info("Sagittal model training complete!")
    if best_model_path:
        return best_model_path
    else:
        # Fallback if no checkpoint was saved
        return os.path.join(cfg_sagittal.program.checkpoint_path, "best_model.pt")


def train_axial_model(cfg):
    """Train the axial landmark detection model"""
    logging.info("=" * 80)
    logging.info("TRAINING AXIAL LANDMARK MODEL")
    logging.info("=" * 80)

    # Update config for axial model
    cfg_axial = cfg.copy()
    cfg_axial.program.result_path = os.path.join(cfg.program.result_path, "axial_landmarks")
    cfg_axial.program.checkpoint_path = os.path.join(cfg.program.checkpoint_path, "axial_landmarks")
    # out_channels will be determined from dataset.get_num_landmarks()
    cfg_axial.dataset_class = 'AxialLandmarkDataset'

    os.makedirs(cfg_axial.program.result_path, exist_ok=True)
    os.makedirs(cfg_axial.program.checkpoint_path, exist_ok=True)

    # Train using existing train_model function
    best_model_path = train_model_with_dataset(cfg_axial, AxialLandmarkDataset)

    logging.info("Axial model training complete!")
    if best_model_path:
        return best_model_path
    else:
        # Fallback if no checkpoint was saved
        return os.path.join(cfg_axial.program.checkpoint_path, "best_model.pt")


def train_model_with_dataset(cfg, dataset_class):
    """
    Modified version of your train_model that accepts a dataset class.

    This is a wrapper around your existing train.py::train_model function.
    You'll need to modify your train.py to accept dataset_class as parameter.
    
    Returns:
        str: Path to the best model checkpoint, or None if no checkpoint was saved.
    """
    from torch.utils.tensorboard import SummaryWriter
    from torch.utils.data import DataLoader
    import torch
    from data import LakeFSLoader, DataAugmentation
    from models import UNet, CELoss, save_model
    from scripts import run_evaluation, define_device
    import time

    writer = SummaryWriter(cfg.program.result_path)
    OmegaConf.save(cfg, os.path.join(cfg.program.result_path, "parameter_config.yaml"))

    device = define_device(cfg)

    data_loader_lakefs = LakeFSLoader(
        endpoint=cfg.lakefs.s3_endpoint,
        repo_name=cfg.lakefs.repository,
        local_cache_path=cfg.data.local_cache_path,
        ca_path=cfg.lakefs.ca_path
    )

    augmentation = None
    if cfg.training.use_augmentation:
        augmentation = DataAugmentation(cfg)

    # Use the provided dataset class instead of hardcoded KneeDataset
    data_set_training = dataset_class(
        data_path_file=os.path.join(cfg.data.data_split_path, "train.json"),
        lakefs_loader=data_loader_lakefs,
        image_size=cfg.program.image_size,
        augmentation=augmentation,
        keypoint_std=3.0
    )

    data_set_validation = dataset_class(
        data_path_file=os.path.join(cfg.data.data_split_path, "val.json"),
        lakefs_loader=data_loader_lakefs,
        image_size=cfg.program.image_size,
        augmentation=None
    )

    num_landmarks = data_set_training.get_num_landmarks()

    # Define model
    model = UNet(
        in_channels=1,
        out_channels=num_landmarks,
        down_channels=cfg.model.down_channels,
        down_layers=cfg.model.down_layers,
        up_channels=cfg.model.up_channels,
        up_layers=cfg.model.up_layers,
        bottleneck_channels=cfg.model.bottleneck_channels,
        bottleneck_layers=cfg.model.bottleneck_layers,
        dropout_p=cfg.model.dropout_p,
        residual=cfg.model.residual,
        keypoint_extraction=cfg.model.keypoint_extraction
    )

    model.train()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: (1 - cfg.optimizer.lr_min) * cfg.optimizer.lr_decay ** epoch + cfg.optimizer.lr_min
    )

    loss_function = CELoss()

    logging.info(f"Number of samples in training data set: {len(data_set_training)}")
    logging.info(f"Number of samples in validation data set: {len(data_set_validation)}")
    logging.info(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")

    # Save debug training images (5 samples)
    debug_output_dir = os.path.join(cfg.program.result_path, "debug_training_images")
    os.makedirs(debug_output_dir, exist_ok=True)
    _save_training_debug_images(data_set_training, debug_output_dir, model_name=dataset_class.__name__)
    logging.info(f"Saved debug training images to {debug_output_dir}")

    data_loader = DataLoader(
        data_set_training,
        batch_size=cfg.program.data_loader.batch_size,
        num_workers=cfg.program.data_loader.num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=cfg.program.data_loader.persistent_workers,
        collate_fn=lambda x: x
    )

    if not os.path.exists(cfg.program.checkpoint_path):
        os.makedirs(cfg.program.checkpoint_path)

    last_eval_loss_value = float("inf")
    best_checkpoint_path = None  # Track the path to the best checkpoint

    logging.info("Starting training")
    for epoch_idx in range(1, cfg.training.num_epoch + 1):
        t1 = time.time()
        num_batches = len(data_loader)

        for run_idx, batch in enumerate(data_loader):
            iteration_idx = (epoch_idx - 1) * len(data_loader) + run_idx

            # Train model
            images = torch.stack([item['image'].to(device) for item in batch])
            heatmaps = torch.stack([item['heatmaps'].to(device) for item in batch])
            images.requires_grad_()

            pred_heatmaps = model(images)

            heatmaps_mask = ~(torch.isnan(heatmaps).view(*heatmaps.shape[:-2], -1).any(dim=-1))

            loss = loss_function(pred_heatmaps[heatmaps_mask], heatmaps[heatmaps_mask])
            loss.backward()
            optimizer.step()

            for param in model.parameters():
                param.grad = None

            writer.add_scalar('loss/cross-entropy', loss.item(), iteration_idx)

            logging.info(" ".join([
                "Epoch", f"{epoch_idx}/{cfg.training.num_epoch}",
                "iter", str(iteration_idx + 1), f"{(iteration_idx + 1) % num_batches}/{num_batches}",
                "loss", f"{loss.item():.2E}"
            ]))

        logging.info(f"lr = {scheduler.get_last_lr()}")
        scheduler.step()

        torch.cuda.empty_cache()

        t2 = time.time()
        logging.info(f'Epoch {epoch_idx} took {t2 - t1:.2f} seconds.')

        if epoch_idx % cfg.training.eval_each_epoch == 0:
            eval_result = run_evaluation(
                cfg, data_set_validation, device, model,
                show_heatmaps=(epoch_idx in cfg.evaluation.create_images_epochs)
            )
            logging.info(eval_result)
            writer.add_scalar('eval/keypoint', eval_result["pos_loss"], epoch_idx)
            writer.add_scalar('eval/keypoint large deviation', eval_result["pos_dev_loss"], epoch_idx)

            # Log images with predictions to TensorBoard
            _log_predictions_to_tensorboard(
                writer, data_set_validation, device, model, epoch_idx, max_images=4
            )

            # Save best model
            if last_eval_loss_value > eval_result["pos_loss"]:
                last_eval_loss_value = eval_result["pos_loss"]
                save_model(cfg.program.checkpoint_path, model, "keypoint_model", epoch_idx)
                # Update the best checkpoint path
                best_checkpoint_path = os.path.join(cfg.program.checkpoint_path, f"checkpoint_keypoint_model_{epoch_idx}.pt")
                
                # Also create a symlink/copy as best_model.pt for convenience
                best_model_path = os.path.join(cfg.program.checkpoint_path, "best_model.pt")
                if os.path.exists(best_model_path):
                    os.remove(best_model_path)
                try:
                    # Try to create a symlink (works on Unix/Linux/Mac)
                    os.symlink(os.path.basename(best_checkpoint_path), best_model_path)
                except (OSError, AttributeError):
                    # If symlink fails (Windows or other issues), copy the file instead
                    import shutil
                    shutil.copy2(best_checkpoint_path, best_model_path)

    writer.close()
    
    # Return the best checkpoint path (or best_model.pt if it exists)
    best_model_path = os.path.join(cfg.program.checkpoint_path, "best_model.pt")
    if os.path.exists(best_model_path):
        return best_model_path
    elif best_checkpoint_path and os.path.exists(best_checkpoint_path):
        return best_checkpoint_path
    else:
        # Fallback: find the most recent checkpoint
        import glob
        checkpoint_files = glob.glob(os.path.join(cfg.program.checkpoint_path, "checkpoint_keypoint_model_*.pt"))
        if checkpoint_files:
            # Sort by epoch number (extract from filename)
            checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
            return checkpoint_files[-1]
        else:
            return None


def _save_training_debug_images(dataset, output_dir, model_name="Unknown", num_samples=5):
    """
    Save debug images from training dataset with ground truth keypoints overlaid.
    Similar to inference debug visualization but for training data.
    
    Args:
        dataset: Training dataset
        output_dir: Directory to save debug images
        model_name: Name of the model (for filename prefix)
        num_samples: Number of samples to save
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except ImportError:
        logging.warning("matplotlib not available, skipping debug image saving")
        return

    # Sample indices evenly distributed
    num_samples = min(num_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    
    for idx_idx, dataset_idx in enumerate(indices):
        try:
            sample = dataset[dataset_idx]
            
            # Get image and keypoints
            image = sample['image'].squeeze(0).cpu().numpy()  # (H, W)
            keypoints = sample['keypoints']  # (num_landmarks, 2) or list of [x, y] or None
            
            # Extract patient/case name from case_id, image path, or dataset structure
            import os
            case_name = None
            case_id = sample.get('case_id', None)
            
            # First try to get case name from the dataset's data structure
            # The dataset.data is a list of items, and each item has an 'image' path
            if hasattr(dataset, 'data') and dataset_idx < len(dataset.data):
                item = dataset.data[dataset_idx]
                # Extract case name from image path (most reliable)
                if 'image' in item:
                    image_path = item['image']
                    # Extract case directory name from path
                    # Path format is usually like "path/to/case_dir/image.nii.gz"
                    path_parts = image_path.replace('\\', '/').strip('/').split('/')
                    # The case name is typically the directory containing the image
                    if len(path_parts) >= 2:
                        # Use the parent directory name as case name
                        case_name = path_parts[-2]  # Second to last part is usually the case directory
                    elif len(path_parts) == 1:
                        # If only one part, try to extract from filename
                        filename = os.path.splitext(os.path.splitext(path_parts[0])[0])[0]  # Remove .nii.gz
                        case_name = filename
            
            # If still no case name, try from case_id (metadata path)
            if not case_name and case_id is not None:
                if isinstance(case_id, str):
                    # If case_id is a path (metadata path), extract the case directory name
                    if '/' in case_id or '\\' in case_id:
                        case_name = os.path.basename(os.path.dirname(case_id))
                        if not case_name or case_name == '.':
                            # Fallback: try to extract from the full path
                            parts = case_id.replace('\\', '/').split('/')
                            # Find the part that looks like a case name
                            for part in reversed(parts):
                                if part and part != '.' and not part.endswith('.json'):
                                    case_name = part
                                    break
                    else:
                        # It's already a name
                        case_name = case_id
                else:
                    case_name = f'case_{case_id}'
            
            # Final fallback
            if not case_name:
                case_name = f'unknown_case_{dataset_idx}'
            
            # Sanitize case name for filename (remove invalid characters)
            case_name_safe = "".join(c for c in case_name if c.isalnum() or c in ('_', '-', '.')).strip()
            if not case_name_safe:
                case_name_safe = f'case_{dataset_idx}'
            
            # Normalize image for display
            if image.max() > image.min():
                image_normalized = (image - image.min()) / (image.max() - image.min())
            else:
                image_normalized = image
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(image_normalized, cmap='gray')
            ax.set_title(f'{model_name} - Training Sample {idx_idx + 1}/{num_samples}\n'
                        f'Patient: {case_name}\nDataset Index: {dataset_idx}, Shape: {image.shape}')
            ax.set_xlabel('Column (x)')
            ax.set_ylabel('Row (y)')
            
            # Plot ground truth keypoints
            if isinstance(keypoints, np.ndarray):
                keypoints_list = keypoints
            elif isinstance(keypoints, list):
                keypoints_list = np.array([kp for kp in keypoints if kp is not None])
            else:
                keypoints_list = np.array([])
            
            for i, kp in enumerate(keypoints_list):
                if len(kp) >= 2 and not np.any(np.isnan(kp[:2])):
                    x, y = float(kp[0]), float(kp[1])
                    # Draw circle and cross
                    circle = Circle((x, y), radius=5, color='lime', fill=False, linewidth=2)
                    ax.add_patch(circle)
                    ax.plot(x, y, 'g+', markersize=15, markeredgewidth=3)
                    ax.text(x + 10, y, f'GT-{i+1}\n({x:.1f}, {y:.1f})', 
                           color='yellow', fontsize=9, weight='bold',
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save figure with patient name in filename
            output_path = os.path.join(output_dir, f'{model_name}_{case_name_safe}_sample_{idx_idx + 1:02d}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Saved debug image: {output_path}")
            
        except Exception as e:
            logging.warning(f"Failed to save debug image for sample {dataset_idx}: {e}")


def _log_predictions_to_tensorboard(writer, dataset, device, model, epoch, max_images=4):
    """
    Log validation images with ground truth and predicted keypoints to TensorBoard.
    
    For sagittal bounds: Should show a side view of the knee with 2 boundary points.
    For axial landmarks: Should show a top-down view with 3 landmarks.
    
    Args:
        writer: TensorBoard SummaryWriter
        dataset: Validation dataset
        device: torch device
        model: Trained model
        epoch: Current epoch number
        max_images: Maximum number of images to log
    """
    model.eval()
    
    # Sample a few images from validation set
    num_samples = min(max_images, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    
    images_to_log = []
    
    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            
            # Get image and ground truth keypoints
            image = sample['image'].unsqueeze(0).to(device)  # (1, 1, H, W)
            gt_keypoints = sample['keypoints']  # (num_landmarks, 2)
            
            # Predict
            pred_result = model.predict(image)
            pred_keypoints = pred_result['keypoints'][0]  # (num_landmarks, 2)
            pred_heatmaps = pred_result['heatmaps'][0]  # (num_landmarks, H, W)
            
            # Create visualization
            # Convert image to numpy for visualization
            img_np = image[0, 0].cpu().numpy()
            
            # Check if image is mostly black/noise (indicates wrong slice or orientation issue)
            img_mean = img_np.mean()
            img_std = img_np.std()
            
            # Normalize to [0, 1] range
            if img_std > 1e-6:
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-10)
            else:
                # Image is uniform/black - this might indicate wrong slice
                logging.warning(f"Image {idx} appears uniform (mean={img_mean:.3f}, std={img_std:.3f}) - may indicate wrong slice extraction")
                img_np = np.zeros_like(img_np)
            
            # Create RGB image for visualization (3 channels)
            img_rgb = np.stack([img_np, img_np, img_np], axis=0).astype(np.float32)  # (3, H, W)
            
            # Overlay ground truth keypoints (green circles)
            for kp in gt_keypoints:
                if not np.any(np.isnan(kp)):
                    x, y = int(kp[0]), int(kp[1])  # Note: kp[0] is x, kp[1] is y in image coordinates
                    if 0 <= y < img_rgb.shape[1] and 0 <= x < img_rgb.shape[2]:
                        # Draw green circle for ground truth
                        for dy in range(-4, 5):
                            for dx in range(-4, 5):
                                if dy*dy + dx*dx <= 16 and 0 <= y+dy < img_rgb.shape[1] and 0 <= x+dx < img_rgb.shape[2]:
                                    img_rgb[1, y+dy, x+dx] = 1.0  # Green channel
                                    img_rgb[0, y+dy, x+dx] = 0.0
                                    img_rgb[2, y+dy, x+dx] = 0.0
            
            # Overlay predicted keypoints (red crosses)
            for kp in pred_keypoints:
                if not np.any(np.isnan(kp)):
                    x, y = int(kp[0]), int(kp[1])  # Note: kp[0] is x, kp[1] is y
                    if 0 <= y < img_rgb.shape[1] and 0 <= x < img_rgb.shape[2]:
                        # Draw red cross for predictions (thicker)
                        for i in range(-5, 6):
                            if 0 <= y+i < img_rgb.shape[1]:
                                img_rgb[0, y+i, x] = 1.0  # Red channel
                                img_rgb[1, y+i, x] = 0.0
                                img_rgb[2, y+i, x] = 0.0
                            if 0 <= x+i < img_rgb.shape[2]:
                                img_rgb[0, y, x+i] = 1.0  # Red channel
                                img_rgb[1, y, x+i] = 0.0
                                img_rgb[2, y, x+i] = 0.0
            
            images_to_log.append(torch.from_numpy(img_rgb))
            
            # Also log predicted heatmaps (max over all landmarks)
            max_heatmap = pred_heatmaps.max(dim=0)[0].cpu().numpy()  # (H, W)
            if max_heatmap.max() > max_heatmap.min() + 1e-6:
                max_heatmap = (max_heatmap - max_heatmap.min()) / (max_heatmap.max() - max_heatmap.min() + 1e-10)
            # Convert to RGB (heat colormap: red for high values)
            heatmap_rgb = np.stack([max_heatmap, max_heatmap * 0.3, max_heatmap * 0.1], axis=0).astype(np.float32)  # (3, H, W)
            images_to_log.append(torch.from_numpy(heatmap_rgb))
    
    model.train()
    
    if images_to_log:
        # Create grid of images (2 columns: image with keypoints, heatmap)
        grid = make_grid(images_to_log, nrow=2, padding=2, normalize=False)
        writer.add_image('validation/predictions', grid, epoch)


def main():
    parser = argparse.ArgumentParser(description="Train both sagittal and axial models")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model', type=str, choices=['sagittal', 'axial', 'both'], default='both',
                        help='Which model(s) to train')
    parser.add_argument('--sagittal_first', action='store_true',
                        help='Train sagittal model first, then axial')
    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if args.model == 'both' or args.sagittal_first:
        # Train both models sequentially
        sagittal_model_path = train_sagittal_model(cfg)
        logging.info(f"Sagittal model saved at: {sagittal_model_path}")

        axial_model_path = train_axial_model(cfg)
        logging.info(f"Axial model saved at: {axial_model_path}")

        logging.info("=" * 80)
        logging.info("BOTH MODELS TRAINED SUCCESSFULLY!")
        logging.info(f"Sagittal: {sagittal_model_path}")
        logging.info(f"Axial: {axial_model_path}")
        logging.info("=" * 80)

    elif args.model == 'sagittal':
        sagittal_model_path = train_sagittal_model(cfg)
        logging.info(f"Sagittal model saved at: {sagittal_model_path}")

    elif args.model == 'axial':
        axial_model_path = train_axial_model(cfg)
        logging.info(f"Axial model saved at: {axial_model_path}")


if __name__ == "__main__":
    main()