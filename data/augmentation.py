import albumentations as A
import cv2
import numpy as np


class DataAugmentation():
    def __init__(self, cfg):
        # order matters!!!
        transformation_list = [
            A.Rotate(limit=cfg.augmentation.rotate_limit, border_mode=cv2.BORDER_CONSTANT, p=cfg.augmentation.rotate_prob),
            A.ElasticTransform(alpha=cfg.augmentation.elastic_deform_alpha, sigma=cfg.augmentation.elastic_deform_sigma, p=cfg.augmentation.elastic_deform_prob),
            A.RandomBrightnessContrast(brightness_limit=cfg.augmentation.brightness_limit, contrast_limit=cfg.augmentation.contrast_limit, brightness_by_max=True, p=cfg.augmentation.brightness_prob),
            A.MultiplicativeNoise(multiplier=cfg.augmentation.mult_noise_factor, elementwise=True, p=cfg.augmentation.mult_noise_prob),
            A.GaussNoise(std_range=cfg.augmentation.gauss_noise_std_range, p=cfg.augmentation.gauss_noise_prob)
        ]

        self.transformation = A.Compose(transformation_list, keypoint_params=A.KeypointParams(format='xy', remove_invisible=True, label_fields=['class_labels']))

    def augment(self, image, keypoints, class_labels):
        image = np.array(image)
        
        transformed = self.transformation(image=image, keypoints=keypoints, class_labels=class_labels)
        transformed_image = transformed['image']
        transformed_keypoints = transformed['keypoints']
        transformed_class_labels = map(int, transformed['class_labels'])
        
        return transformed_image, transformed_keypoints, transformed_class_labels
