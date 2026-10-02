import numpy as np
import nibabel as nib
import os

def normalize_shift_to_zero_one(img_data):
    return (img_data + 1)/2
# Input paths
input_path = r"C:\Users\michael\Documents\Data\Gyozo_HR_MRI\Fesh_Paul_V2\res_224_320_320_epoch_100_contr_0.nii.gz"
# Load image
img = nib.load(input_path)
data = img.get_fdata()

base, ext = os.path.splitext(os.path.basename(input_path))
if ext == ".gz":  # handle .nii.gz
    base, _ = os.path.splitext(base)
    ext = ".nii.gz"
else:
    ext = ".nii"

output_path = os.path.join(os.path.dirname(input_path), f"{base}_zeroone{ext}")

zeroone_data = normalize_shift_to_zero_one(data)

# Save normalized image
zeroone_img = nib.Nifti1Image(zeroone_data, affine=np.eye(4), header=img.header)
nib.save(zeroone_img, output_path)
print(f"Saved normalized image to {output_path}")


