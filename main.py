import numpy as np
import dipy.reconst.shm as shm
import dipy.direction.peaks as dp
import matplotlib.pyplot as plt

from dipy.denoise.localpca import mppca
from dipy.core.gradients import gradient_table, unique_bvals_tolerance
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.segment.mask import median_otsu
from dipy.reconst.mcsd import (auto_response_msmt,
                               mask_for_response_msmt,
                               response_from_mask_msmt)
from dipy.segment.tissue import TissueClassifierHMRF
from dipy.reconst.mcsd import MultiShellDeconvModel, multi_shell_fiber_response
from dipy.viz import window, actor

from dipy.data import get_sphere, get_fnames


def main():
    sphere = get_sphere('symmetric724')
    data, gtab = load_data()
    response_fun = get_ms_response(data, gtab, sphere)
    # shm.spherical_harmonics()


def load_data():
    fraw, fbval, fbvec, t1_fname = get_fnames('cfin_multib')

    data, affine = load_nifti(fraw)
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)

    bvals = gtab.bvals
    bvecs = gtab.bvecs

    sel_b = np.logical_or(np.logical_or(bvals == 0, bvals == 1000), bvals == 2000)
    data = data[..., sel_b]

    gtab = gradient_table(bvals[sel_b], bvecs[sel_b])

    return data, gtab


def get_ms_response(data, gtab, sphere):
    b0_mask, mask = median_otsu(data, median_radius=2, numpass=1, vol_idx=[0, 1])

    denoised_arr = mppca(data, mask=mask, patch_radius=2)

    qball_model = shm.QballModel(gtab, 8)

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

    mask_wm *= wm
    mask_gm *= gm
    mask_csf *= csf

    nvoxels_wm = np.sum(mask_wm)
    nvoxels_gm = np.sum(mask_gm)
    nvoxels_csf = np.sum(mask_csf)

    response_wm, response_gm, response_csf = response_from_mask_msmt(gtab, data,
                                                                     mask_wm,
                                                                     mask_gm,
                                                                     mask_csf)

    ubvals = unique_bvals_tolerance(gtab.bvals)

    response_mcsd = multi_shell_fiber_response(sh_order=8,
                                               bvals=ubvals,
                                               wm_rf=response_wm,
                                               gm_rf=response_gm,
                                               csf_rf=response_csf)

    return response_mcsd


if __name__ == '__main__':
    main()
