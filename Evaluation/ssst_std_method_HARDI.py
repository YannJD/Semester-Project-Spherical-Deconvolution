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


def main():
    data, affine = load_nifti('Hardi_dataset/hardi-scheme_SNR-30.nii.gz')
    bvals, bvecs = read_bvals_bvecs('Hardi_dataset/hardi-scheme.bval', 'Hardi_dataset/hardi-scheme.bvec')
    gtab = gradient_table(bvals, bvecs)

    bvals = gtab.bvals
    bvecs = gtab.bvecs

    sel_b = np.logical_or(bvals == 0, bvals == 3000)
    data = data[..., sel_b]

    gtab = gradient_table(bvals[sel_b], bvecs[sel_b])

    l_max = 8

    if pathlib.Path('mask.nii').exists():
        mask, affine = load_nifti('mask.nii')
        mask = mask.astype(bool)
        data[~mask] = 0
        data = mppca(data, mask=mask, patch_radius=2)

    # response_fun = get_ss_calibration_response(data, gtab, l_max)
    response_fun, ratio = auto_response_ssst(gtab, data, roi_center=[44, 24, 25], roi_radii=3, fa_thr=0.7)

    sphere = get_sphere('symmetric724')

    csd_model = ConstrainedSphericalDeconvModel(gtab, response_fun, sh_order=l_max)
    csd_fit = csd_model.fit(data)
    csd_odf = csd_fit.odf(sphere)

    name = 'std_ssst_HARDI'

    save_nifti(name + '_odfs.nii.gz', csd_odf, affine)

    save_to_mrtrix_format(shm.sf_to_sh(csd_odf, sphere, l_max), l_max, sphere, 1, name)


if __name__ == '__main__':
    main()
