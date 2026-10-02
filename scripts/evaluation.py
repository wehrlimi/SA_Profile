import os
import io
import logging

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_test(cfg):
    raise NotImplementedError()


def run_evaluation(cfg, data_set, device, model, show_heatmaps=False):
    logging.info("Start Evaluation:")

    data_loader = DataLoader(data_set, batch_size=cfg.program.data_loader.batch_size,
                             num_workers=cfg.program.data_loader.num_workers,
                             shuffle=False, pin_memory=True, persistent_workers=cfg.program.data_loader.persistent_workers,
                             collate_fn=lambda x : x)
    result_path_images = os.path.join(cfg.evaluation.result_path, "img")
    
    model.eval()

    point_position_error_total = 0
    point_deviation_error_total = 0

    num_keypoints = data_set.num_keypoints
    point_position_errors_by_landmark = [[] for _ in range(num_keypoints)]

    with torch.no_grad():
        for run_idx, batch in enumerate(data_loader):
            logging.info(f"Batch {run_idx+1}/{len(data_loader)}")

            images = torch.stack([item['image'].to(device) for item in batch])
            keypoints = np.array( [ [ p if p is not None else np.array([np.nan, np.nan]) for p in item['keypoints'] ] for item in batch] )

            keypoint_mask = ~np.isnan(keypoints)
            
            # predict
            images = images.detach()
            model_pred = model.predict(images)

            pred_keypoints = model_pred["keypoints"]

            # compute errors
            distances = np.sqrt(np.sum((np.where(keypoint_mask, pred_keypoints, 0) - np.where(keypoint_mask, keypoints, 0))**2, axis=-1))
            num_points = np.sum(np.sum(keypoint_mask, axis=-1) > 0)
            batch_point_error = np.sum(distances) / num_points

            batch_point_deviation_error = np.sum(distances > 20) / num_points

            point_position_error_total += batch_point_error
            point_deviation_error_total += batch_point_deviation_error

            # distances by landmark
            all_distances = np.full((len(batch), num_keypoints), -np.inf)
            
            for i in range(len(batch)):
                for k in range(num_keypoints):
                    if not np.any(np.isnan(keypoints[i, k])):
                        point_position_errors_by_landmark[k].append(distances[i, k])
                        all_distances[i, k] = distances[i, k]

            if show_heatmaps:
                if not os.path.isdir(result_path_images):
                    os.makedirs(result_path_images)
                
                create_image_overlay(result_path_images, images.cpu(), pred_keypoints, keypoints, run_idx, data_set.keypoint_names, model_pred["heatmaps"],
                                     model_pred["visible"] if "visible" in model_pred else None)

    if show_heatmaps:
        cols = 5
        rows = (num_keypoints + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        axes = axes.flatten()

        for i in range(num_keypoints):
            axes[i].hist(point_position_errors_by_landmark[i], bins=30, edgecolor='black')
            axes[i].set_title(data_set.keypoint_names[i])
            axes[i].set_xlabel("l2-distance")

        for j in range(num_keypoints, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(result_path_images, "distances.png"), bbox_inches="tight")
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        plt.close(fig)

    return {"pos_loss": point_position_error_total / len(data_loader),
            "pos_dev_loss": point_deviation_error_total / len(data_loader)
            }

def create_image_overlay(result_path_images, images, pos_est_batches, keypoints_batches, run_idx, landmarks_shown, heatmaps, visible):
    num_landmarks_shown = len(landmarks_shown)
    for batch_idx in range(images.shape[0]):
        # Calculate rows needed
        # Row 0: overview (col 0) + landmark 0 (cols 2-3)
        # Row 1+: landmarks 1+ (2 per row, each needs 2 cols: cols 0-1 and 2-3)
        # Formula: 1 row for overview + landmark 0, then ceil((num_landmarks_shown - 1) / 2) rows for remaining landmarks
        # Total: 1 + ceil((num_landmarks_shown - 1) / 2) = 1 + (num_landmarks_shown - 1 + 1) // 2 = 1 + num_landmarks_shown // 2
        # But for num_landmarks_shown=0: need 1 row, for num_landmarks_shown=1: need 1 row, for num_landmarks_shown>=2: need at least 2 rows
        if num_landmarks_shown <= 1:
            rows = 1
        else:
            rows = 1 + (num_landmarks_shown - 1 + 1) // 2  # = 1 + num_landmarks_shown // 2
        cols = 4
        fig, axs = plt.subplots(rows, cols, figsize=(20, 4*rows))
        
        # Handle case where axs is 1D (when rows=1)
        if rows == 1:
            axs = axs.reshape(1, -1)
        elif axs.ndim == 1:
            axs = axs.reshape(rows, -1)
        
        image = images[batch_idx].detach().to("cpu").numpy()[0]

        pos_est = pos_est_batches[batch_idx]
        pos_gt = keypoints_batches[batch_idx].squeeze()

        axs[0, 0].imshow(image, cmap="Greys")
        for i, landmark_name in enumerate(landmarks_shown):
            axs[0, 0].plot(pos_gt[i, 0], pos_gt[i, 1], "go", mew=3, ms=8)
            axs[0, 0].plot(pos_est[i, 0], pos_est[i, 1], "r+", mew=3, ms=8)
            axs[0, 0].plot([pos_gt[i, 0], pos_est[i, 0]], [pos_gt[i, 1], pos_est[i, 1]], 'b-')

        for landmark_idx, landmark_name in enumerate(landmarks_shown):
            # Calculate row and column for this landmark
            # Original formula: i, j = (landmark_idx + 1) // 2, 2 * ((landmark_idx + 1) % 2)
            # This gives: landmark 0 -> row 0, col 2; landmark 1 -> row 1, col 0; landmark 2 -> row 1, col 2
            i = (landmark_idx + 1) // 2
            j = 2 * ((landmark_idx + 1) % 2)
            
            # Ensure we don't go out of bounds
            if i >= rows or j + 1 >= cols:
                continue
                
            axs[i, j].imshow(image, cmap="Greys")
            axs[i, j].plot(pos_gt[landmark_idx, 0], pos_gt[landmark_idx, 1], "go", mew=3, ms=8)
            axs[i, j].plot(pos_est[landmark_idx, 0], pos_est[landmark_idx, 1], "r+", mew=3, ms=8)
            axs[i, j].plot([pos_gt[landmark_idx, 0], pos_est[landmark_idx, 0]], [pos_gt[landmark_idx, 1], pos_est[landmark_idx, 1]], 'b-')
            axs[i, j].set_title(f"{landmark_name}, p = " + ("{:.5f}".format(visible[batch_idx, landmark_idx]) if visible is not None else "?"))

            axs[i, j+1].imshow(heatmaps[batch_idx][landmark_idx], cmap='jet')
            #axs[i, j+1].plot(pos_est[landmark_idx, 0], pos_est[landmark_idx, 1], "w+", mew=3, ms=8)

        plt.tight_layout()
        plt.savefig(os.path.join(result_path_images, str(run_idx*images.shape[0]+batch_idx).zfill(5) + ".png"), bbox_inches="tight")

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        plt.close(fig)
