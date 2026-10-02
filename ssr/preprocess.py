import nibabel as nib
import numpy as np


def clip_and_normalize(img):
    data = img.get_fdata()
    clipped = np.clip(data, 0, np.quantile(data, 0.999))
    normalized = (clipped - np.min(clipped))/(np.max(clipped) - np.min(clipped))

    return normalized



ax = nib.load('/home/paulfriedrich/Desktop/single_subject_sr/2_pd_axial_fat_sat.nii.gz')
ax_new = clip_and_normalize(ax)
ax_new_nifti = nib.Nifti1Image(ax_new, np.eye(4))
nib.save(ax_new_nifti, '/home/paulfriedrich/Desktop/single_subject_sr/axial.nii.gz')

cor = nib.load('/home/paulfriedrich/Desktop/single_subject_sr/3_pd_fs_cor.nii.gz')
cor_new = clip_and_normalize(cor)
cor_new_nifti = nib.Nifti1Image(cor_new, np.eye(4))
nib.save(cor_new_nifti, '/home/paulfriedrich/Desktop/single_subject_sr/coronal.nii.gz')

sag = nib.load('/home/paulfriedrich/Desktop/single_subject_sr/5_pd_fs_sag.nii.gz')
sag_new = clip_and_normalize(sag)
sag_new_nifti = nib.Nifti1Image(sag_new, np.eye(4))
nib.save(sag_new_nifti, '/home/paulfriedrich/Desktop/single_subject_sr/sagittal.nii.gz')
