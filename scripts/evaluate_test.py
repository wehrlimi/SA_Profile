import os
import sys
import logging
import argparse
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf
from data import LakeFSLoader
from data.dataset_classes import SagittalBoundsDataset, AxialLandmarkDataset
from models import load_model

def evaluate_model(cfg, dataset, model, device, landmark_names_offset=0):
    """
    Evaluate a model on a dataset and return errors for each landmark.
    """
    data_loader = DataLoader(dataset, batch_size=cfg.program.data_loader.batch_size,
                             num_workers=0,
                             shuffle=False, pin_memory=True, collate_fn=lambda x: x)
    
    model.eval()
    num_landmarks = dataset.num_keypoints
    all_errors_pixel = [[] for _ in range(num_landmarks)]
    all_errors_mm = [[] for _ in range(num_landmarks)]

    with torch.no_grad():
        for batch in data_loader:
            images = torch.stack([item['image'].to(device) for item in batch])
            gt_keypoints = np.array([[p if p is not None else np.array([np.nan, np.nan]) for p in item['keypoints']] for item in batch])
            spacing_mm = np.array([item['spacing_mm'] for item in batch]) # Shape: (batch, 2) [dy, dx]

            # Predict
            model_pred = model.predict(images)
            pred_keypoints = model_pred["keypoints"] # Shape: (batch, num_landmarks, 2)

            for b in range(len(batch)):
                for k in range(num_landmarks):
                    if not np.any(np.isnan(gt_keypoints[b, k])):
                        # Pixel distance
                        diff_pixel = pred_keypoints[b, k] - gt_keypoints[b, k]
                        dist_pixel = np.sqrt(np.sum(diff_pixel**2))
                        all_errors_pixel[k].append(dist_pixel)

                        # MM distance
                        # spacing_mm[b] is [dy, dx] (row, col spacing)
                        # diff_pixel is [x, y] (col, row diff)
                        # We need to multiply x by spacing_mm[1] and y by spacing_mm[0]
                        diff_mm = diff_pixel * np.array([spacing_mm[b, 1], spacing_mm[b, 0]])
                        dist_mm = np.sqrt(np.sum(diff_mm**2))
                        all_errors_mm[k].append(dist_mm)

    return all_errors_pixel, all_errors_mm

def main():
    parser = argparse.ArgumentParser(description="Evaluate knee landmark detection models on test data")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load config
    cfg = OmegaConf.load(args.config)
    device = torch.device(cfg.device.type if torch.cuda.is_available() else "cpu")

    # Initialize LakeFS Loader
    lakefs_loader = LakeFSLoader(
        endpoint=cfg.lakefs.s3_endpoint,
        repo_name=cfg.lakefs.repository,
        local_cache_path=cfg.data.local_cache_path,
        ca_path=cfg.lakefs.ca_path
    )

    # Specific checkpoints from user
    sagittal_checkpoint = os.path.join("tmp", "checkpoints", "sagittal_bounds", "checkpoint_keypoint_model_34.pt")
    axial_checkpoint = os.path.join("tmp", "checkpoints", "axial_landmarks", "checkpoint_keypoint_model_24.pt")
    test_json = os.path.join("tmp", "test.json")

    # Load models
    logging.info(f"Loading sagittal model from {sagittal_checkpoint}")
    sagittal_model = load_model(sagittal_checkpoint, device)
    
    logging.info(f"Loading axial model from {axial_checkpoint}")
    axial_model = load_model(axial_checkpoint, device)

    # Initialize datasets
    sagittal_test_set = SagittalBoundsDataset(
        data_path_file=test_json,
        lakefs_loader=lakefs_loader,
        image_size=cfg.program.image_size,
        augmentation=None
    )

    axial_test_set = AxialLandmarkDataset(
        data_path_file=test_json,
        lakefs_loader=lakefs_loader,
        image_size=cfg.program.image_size,
        augmentation=None
    )

    # Evaluate sagittal model
    logging.info("Evaluating Sagittal Model...")
    sag_pixels, sag_mm = evaluate_model(cfg, sagittal_test_set, sagittal_model, device)

    # Evaluate axial model
    logging.info("Evaluating Axial Model...")
    ax_pixels, ax_mm = evaluate_model(cfg, axial_test_set, axial_model, device)

    # Combine results
    all_pixels = sag_pixels + ax_pixels
    all_mm = sag_mm + ax_mm
    landmark_names = sagittal_test_set.keypoint_names + axial_test_set.keypoint_names
    
    # Print Summary
    print("\nEvaluation Summary (Mean Absolute Error):")
    print(f"{'Landmark':<20} | {'Pixel Error':<12} | {'MM Error':<12}")
    print("-" * 50)
    for i, name in enumerate(landmark_names):
        m_pixel = np.mean(all_pixels[i]) if all_pixels[i] else 0
        m_mm = np.mean(all_mm[i]) if all_mm[i] else 0
        print(f"{name:<20} | {m_pixel:>12.4f} | {m_mm:>12.4f}")

    # Rename landmarks for display
    display_names = []
    for name in landmark_names:
        if name == 'boundary_point_0':
            display_names.append('S1')
        elif name == 'boundary_point_1':
            display_names.append('S2')
        elif name.startswith('landmark_'):
            idx = int(name.split('_')[-1])
            display_names.append(f'T{idx + 1}')
        else:
            display_names.append(name)

    # Create Boxplots
    logging.info("Generating Boxplots...")
    os.makedirs(cfg.evaluation.result_path, exist_ok=True)

    num_landmarks = len(display_names)
    # Five distinct green shades (dark to light)
    landmark_colors = ['#1B5E20', '#388E3C', '#4CAF50', '#81C784', '#C8E6C9'][:num_landmarks]

    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 15,
        'xtick.labelsize': 14,
        'ytick.labelsize': 13,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Helper to compute y-limits from data without outliers (whisker range)
    def whisker_ylim(data_list, pad_frac=0.1):
        lo, hi = float('inf'), float('-inf')
        for data in data_list:
            if len(data) == 0:
                continue
            q1, q3 = np.percentile(data, 25), np.percentile(data, 75)
            iqr = q3 - q1
            whisker_lo = max(np.min(data), q1 - 1.5 * iqr)
            whisker_hi = min(np.max(data), q3 + 1.5 * iqr)
            lo = min(lo, whisker_lo)
            hi = max(hi, whisker_hi)
        pad = (hi - lo) * pad_frac
        return max(0, lo - pad), hi + pad

    # --- Pixel Boxplot (left) ---
    bp1 = ax1.boxplot(all_pixels, labels=display_names, patch_artist=True,
                       widths=0.55, showfliers=False,
                       medianprops=dict(color='black', linewidth=1.5),
                       whiskerprops=dict(linewidth=1.2),
                       capprops=dict(linewidth=1.2))
    for patch, color in zip(bp1['boxes'], landmark_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)

    ax1.set_ylabel('L2 Distance [pixels]', fontsize=15, fontweight='bold')
    ax1.set_xlabel('')
    ylo, yhi = whisker_ylim(all_pixels)
    ax1.set_ylim(ylo, yhi)
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- MM Boxplot (right) ---
    bp2 = ax2.boxplot(all_mm, labels=display_names, patch_artist=True,
                       widths=0.55, showfliers=False,
                       medianprops=dict(color='black', linewidth=1.5),
                       whiskerprops=dict(linewidth=1.2),
                       capprops=dict(linewidth=1.2))
    for patch, color in zip(bp2['boxes'], landmark_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)

    ax2.set_ylabel('L2 Distance [mm]', fontsize=15, fontweight='bold')
    ax2.set_xlabel('')
    ylo, yhi = whisker_ylim(all_mm)
    ax2.set_ylim(ylo, yhi)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plot_path = os.path.join(cfg.evaluation.result_path, "landmark_errors_boxplot.pdf")
    plt.savefig(plot_path, bbox_inches='tight')
    logging.info(f"Boxplots saved to {plot_path}")

    # Save numeric results
    json_results = {
        "landmark_names": landmark_names,
        "pixel_errors": [list(e) for e in all_pixels],
        "mm_errors": [list(e) for e in all_mm]
    }
    json_path = os.path.join(cfg.evaluation.result_path, "evaluation_results.json")
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=4)
    logging.info(f"Detailed results saved to {json_path}")

if __name__ == "__main__":
    main()
