from pytorch_msssim import ssim
import nibabel as nib
import torch

# Define the paths to the files
gt_file = r"C:\Users\michael\Documents\Data\Gyozo_HR_MRI\Fresh_Paul\7 pd_space_fs_sag_p2_iso_320_FOV_160_0.5x0.5_zerooffset_normalized_zeroone.nii.gz"
pred_file = r"C:\Users\michael\Documents\Data\Gyozo_HR_MRI\Fresh_Paul\res_224_320_320_epoch_100_contr_0.nii.gz"
def load_and_normalize_nii(file_path):
    """Load a NIfTI file and normalize it to [0, 1], then convert to PyTorch tensor."""
    nii_img = nib.load(file_path)
    data = nii_img.get_fdata()
    data = torch.tensor(data, dtype=torch.float32)
    # data_min = data.min()
    # data_max = data.max()
    # if (data_max - data_min) > 0:
    #     data = (data - data_min) / (data_max - data_min)
    # else:
    #     data = torch.zeros_like(data)  # If flat image, just use zeros
    return data.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims => (N=1, C=1, H, W, D)

target = load_and_normalize_nii(gt_file)
pred = load_and_normalize_nii(pred_file)
pred = torch.clip(pred, 0, 1)
print(f'Target shape: {target.shape}, Pred shape: {pred.shape}')

#SSIM

#target = ...  # (N, C, H, W, D)
#pred = ...    # (N, C, H, W, D)
# Check dimensions and value ranges
assert target.shape == pred.shape, f"Shape mismatch: {target.shape} vs {pred.shape}"
assert torch.all((0 <= target) & (target <= 1)), "Target not in [0, 1]"
assert torch.all((0 <= pred) & (pred <= 1)), "Prediction not in [0, 1]"

ssim_score = ssim(pred, target, data_range=1.).item()
print(f'SSIM: {ssim_score:.4f}')
#PSNR
#target = ...  # (N, C, H, W, D)
#pred = ...    # (N, C, H, W, D)

diff = pred - target
squared_error = diff ** 2

#mse = squared_error.view(pred.shape[0], -1).mean(dim=-1)
#mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3, 4])  # Batch-wise
mse = torch.nn.functional.mse_loss(pred, target)
eps = 1e-8  # Avoid division by zero
psnr = 10 * torch.log10((1.0 ** 2) / (mse + eps))
psnr_score = psnr.mean().item()
print(f'PSNR: {psnr_score:.2f}')