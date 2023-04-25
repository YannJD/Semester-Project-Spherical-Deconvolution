import pathlib

import numpy as np
import dipy.direction.peaks as dp
import nibabel as nib
from dipy.core.gradients import unique_bvals_tolerance, gradient_table
from dipy.data import get_sphere
from dipy.denoise.localpca import mppca
from dipy.io import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.reconst import shm
from dipy.reconst.mcsd import MultiShellDeconvModel, multi_shell_fiber_response, response_from_mask_msmt, \
    mask_for_response_msmt
from dipy.segment.mask import median_otsu
from dipy.segment.tissue import TissueClassifierHMRF

from mrtrix_functions import save_to_mrtrix_format


def run():
    data, affine = load_nifti('Hardi_dataset/testing-data_DWIS_hardi-scheme_SNR-20.nii.gz')
    bvals, bvecs = read_bvals_bvecs('Hardi_dataset/hardi-scheme.bval', 'Hardi_dataset/hardi-scheme.bvec')
    gtab = gradient_table(bvals, bvecs)

    bvals = gtab.bvals
    bvecs = gtab.bvecs

    sel_b = np.logical_or(bvals == 0, bvals == 3000)
    data = data[..., sel_b]

    gtab = gradient_table(bvals[sel_b], bvecs[sel_b])

    b0_mask, mask = median_otsu(data, median_radius=2, numpass=1, vol_idx=[0, 1])
    if pathlib.Path('mask.nii').exists():
        mask, affine = load_nifti('mask.nii')
        mask = mask.astype(bool)

    l_max = 8
    sphere = get_sphere('symmetric724')

    ubvals = unique_bvals_tolerance(gtab.bvals)

    denoised_arr = mppca(data, mask=mask, patch_radius=2)

    qball_model = shm.QballModel(gtab, l_max)

    peaks = dp.peaks_from_model(model=qball_model, data=denoised_arr,
                                relative_peak_threshold=.5,
                                min_separation_angle=25,
                                sphere=sphere, mask=mask)
    ap = shm.anisotropic_power(peaks.shm_coeff)

    nclass = 3
    beta = 0.1
    hmrf = TissueClassifierHMRF()
    initial_segmentation, final_segmentation, PVE = hmrf.classify(ap, nclass, beta)
    csf = np.where(final_segmentation == 1, 1, 0)
    gm = np.where(final_segmentation == 2, 1, 0)
    wm = np.where(final_segmentation == 3, 1, 0)

    mask_wm, mask_gm, mask_csf = mask_for_response_msmt(gtab, data, roi_radii=10,
                                                        wm_fa_thr=0.7,
                                                        gm_fa_thr=0.3,
                                                        csf_fa_thr=0.15,
                                                        gm_md_thr=0.001,
                                                        csf_md_thr=0.0032)

    save_nifti('standard_mask.nii.gz', mask_wm.astype(int), affine)

    mask_wm *= wm
    mask_gm *= gm
    mask_csf *= csf

    response_wm, response_gm, response_csf = response_from_mask_msmt(gtab, data,
                                                                     mask_wm,
                                                                     mask_gm,
                                                                     mask_csf)

    response_mcsd = multi_shell_fiber_response(sh_order=l_max,
                                               bvals=ubvals,
                                               wm_rf=response_wm,
                                               gm_rf=response_gm,
                                               csf_rf=response_csf)

    mcsd_model = MultiShellDeconvModel(gtab, response_mcsd)
    mcsd_fit = mcsd_model.fit(denoised_arr)
    mcsd_pred = mcsd_fit.predict()
    mcsd_odf = mcsd_fit.odf(sphere)

    save_nifti('standard_odf.nii.gz', mcsd_odf, affine)

    save_to_mrtrix_format(mcsd_fit.all_shm_coeff, l_max, sphere, 3, 'std_phantom')


if __name__ == '__main__':
    run()
