import numpy as np
from dipy.io.image import load_nifti
import nibabel as nib


def main():
    peaks, affine = load_nifti('ground-truth-peaks.nii.gz')
    mask = np.mean(peaks, axis=3)
    mask = np.abs(mask)
    mask[mask > 0] = 1
    mask[mask == 0] = 0
    mask_img = nib.Nifti1Image(mask, None)
    nib.save(mask_img, "hardi_mask.nii.gz")


if __name__ == '__main__':
    main()