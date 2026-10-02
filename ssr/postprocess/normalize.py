import nibabel as nib
import numpy as np
import os

def normalize_image(image_data):
    image_data = image_data.astype(np.float32)
    min_val = np.min(image_data)
    max_val = np.max(image_data)
    return (((image_data - min_val) / (max_val - min_val)) * 2.0 - 1.0).astype(np.float32)

def normalize_shift_to_zero_one(img_data):
    img_data = img_data.astype(np.float32)
    return ((img_data + 1.0) / 2.0).astype(np.float32)

def clip(image_data, lower=1.0, upper=99.0):
    image_data = image_data.astype(np.float32)
    lower_bound = np.percentile(image_data, lower)
    upper_bound = np.percentile(image_data, upper)
    return np.clip(image_data, lower_bound, upper_bound).astype(np.float32)

def normalize_and_save(input_path):
    # Load image
    img = nib.load(input_path)
    data = img.get_fdata().astype(np.float32)  # Ensure input data is float32


    # Create output filenames
    base, ext = os.path.splitext(os.path.basename(input_path))
    if ext == ".gz":  # handle .nii.gz
        base, _ = os.path.splitext(base)
        ext = ".nii.gz"
    else:
        ext = ".nii"

    path_clipped = os.path.join(os.path.dirname(input_path), f"{base}_clipped{ext}")
    path_clipped_norm = os.path.join(os.path.dirname(input_path), f"{base}_clipped_norm{ext}")
    path_clipped_norm_shifted = os.path.join(os.path.dirname(input_path), f"{base}_clipped_norm_shifted{ext}")

    # Clip
    clipped_data = clip(data)
    clipped_img = nib.Nifti1Image(clipped_data, affine=np.eye(4), header=img.header)
    nib.save(clipped_img, path_clipped)

    # Normalize to [-1, 1]
    normalized_data = normalize_image(clipped_data)
    normalized_img = nib.Nifti1Image(normalized_data, affine=np.eye(4), header=img.header)
    nib.save(normalized_img, path_clipped_norm)

    # Shift to [0, 1]
    shifted_data = normalize_shift_to_zero_one(normalized_data)
    shifted_img = nib.Nifti1Image(shifted_data, affine=np.eye(4), header=img.header)
    nib.save(shifted_img, path_clipped_norm_shifted)

    print(f"Saved normalized image to {input_path}")

# Input path
target_path = r"C:\Users\michael\Documents\Data\Gyozo_HR_MRI\Fesh_Paul_V2\7 pd_space_fs_sag_p2_iso_320_FOV_160_0.5x0.5_zerooffset.nii.gz"

# Normalize and save
normalize_and_save(target_path)
