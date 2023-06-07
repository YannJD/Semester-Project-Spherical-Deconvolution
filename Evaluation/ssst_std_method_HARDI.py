import pathlib

import numpy as np
import dipy.direction.peaks as dp
import nibabel as nib
from dipy.core.gradients import unique_bvals_tolerance, gradient_table
from dipy.data import get_sphere, default_sphere
from dipy.denoise.localpca import mppca
from dipy.io import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.reconst import shm
from dipy.reconst.csdeconv import auto_response_ssst, ConstrainedSphericalDeconvModel
from dipy.reconst.mcsd import MultiShellDeconvModel, multi_shell_fiber_response, response_from_mask_msmt, \
    mask_for_response_msmt
from dipy.segment.mask import median_otsu
from dipy.segment.tissue import TissueClassifierHMRF

from dipy_functions import get_ss_calibration_response
from mrtrix_functions import save_to_mrtrix_format
from dipy_peak_extraction import peak_extraction


def main():
    # data, affine = load_nifti('HARDI_volumes/SNR_10/data_1.nii')
    # bvals, bvecs = read_bvals_bvecs('Hardi_dataset/hardi-scheme.bval', 'Hardi_dataset/hardi-scheme.bvec')
    data, affine = load_nifti('DISCO_volumes/SNR_10/data_1.nii')
    bvals, bvecs = read_bvals_bvecs('DISCO_dataset/DiSCo_1_shell.bvals', 'DISCO_dataset/DiSCo_1_shell.bvecs')
    gtab = gradient_table(bvals, bvecs)

    l_max = 8

    # mask, affine = load_nifti('mask.nii')
    mask, affine = load_nifti('DISCO_dataset/DiSCo1_mask.nii.gz')
    mask = mask.astype(bool)
    data[~mask] = 0
    data = mppca(data, mask=mask, patch_radius=2)

    # response_fun = get_ss_calibration_response(data, gtab, l_max)
    # response_fun, ratio = auto_response_ssst(gtab, data, roi_center=[44, 24, 25], roi_radii=3, fa_thr=0.7)
    response_fun, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)

    sphere = get_sphere('symmetric724')

    csd_model = ConstrainedSphericalDeconvModel(gtab, response_fun, sh_order=l_max)
    csd_fit = csd_model.fit(data)
    csd_odf = csd_fit.odf(sphere)

    path = 'test_ssst'

    save_nifti(path + '/odfs.nii.gz', csd_odf, affine)

    save_to_mrtrix_format(shm.sf_to_sh(csd_odf, sphere, l_max), l_max, sphere, affine, path)

    peak_extraction(
        path + '/odfs.nii.gz',
        'sphere724.txt',
        path + '/peaks_SNR10.nii.gz',
        relative_peak_threshold=0.2,
        min_separation_angle=15.0,
        max_peak_number=3
    )


if __name__ == '__main__':
    main()
