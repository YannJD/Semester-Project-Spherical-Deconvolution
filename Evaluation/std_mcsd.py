import argparse
import os
import pathlib

import numpy as np
import dipy.direction.peaks as dp
import nibabel as nib
from dipy.core.gradients import unique_bvals_tolerance, gradient_table
from dipy.data import get_sphere
from dipy.denoise.localpca import mppca
from dipy.io import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.reconst import shm, dti
from dipy.reconst.dti import fractional_anisotropy
from dipy.reconst.mcsd import MultiShellDeconvModel, multi_shell_fiber_response, response_from_mask_msmt, \
    mask_for_response_msmt
from dipy.segment.mask import median_otsu
from dipy.segment.tissue import TissueClassifierHMRF

import dipy_functions
from Evaluation import dipy_peak_extraction
from mrtrix_functions import save_to_mrtrix_format


def main(fpath, bvals_path, bvecs_path, mask_path, l_max, save_path):
    data, affine = load_nifti(fpath)
    bvals, bvecs = read_bvals_bvecs(bvals_path, bvecs_path)
    gtab = gradient_table(bvals, bvecs)

    # b0_mask, mask = median_otsu(data, median_radius=2, numpass=1, vol_idx=[0, 1])
    mask, affine = load_nifti(mask_path)
    mask = mask.astype(bool)

    sphere = get_sphere('symmetric724')

    ubvals = unique_bvals_tolerance(gtab.bvals)

    if not(pathlib.Path(save_path).exists()):
        os.mkdir(save_path)
    response_path = save_path + '/response.npy'

    denoised_arr = mppca(data, mask=mask, patch_radius=2)

    # If it is already computed, load the existing response function
    if pathlib.Path(response_path).exists():
        with open(response_path, 'rb') as f:
            [response_wm, response_gm, response_csf] = np.load(f)
            response_mcsd = multi_shell_fiber_response(sh_order=l_max,
                                                       bvals=ubvals,
                                                       wm_rf=response_wm,
                                                       gm_rf=response_gm,
                                                       csf_rf=response_csf)

    else:
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

        tenmodel = dti.TensorModel(gtab)
        tenfit = tenmodel.fit(data, mask=mask)

        FA = np.max(fractional_anisotropy(tenfit.evals))

        mask_wm, mask_gm, mask_csf = mask_for_response_msmt(gtab, data, roi_radii=10,
                                                            wm_fa_thr=0.7 * FA,
                                                            gm_fa_thr=0.3,
                                                            csf_fa_thr=0.15,
                                                            gm_md_thr=0.001,
                                                            csf_md_thr=0.0032)

        # save_nifti('standard_mask.nii.gz', mask_wm.astype(int), affine)

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

    # Save response function for future uses
    # with open(response_path, 'wb') as f:
        # np.save(f, [response_wm, response_gm, response_csf])

    mcsd_model = MultiShellDeconvModel(gtab, response_mcsd, sh_order=l_max)
    mcsd_fit = mcsd_model.fit(denoised_arr)
    mcsd_pred = mcsd_fit.predict()
    mcsd_odf = mcsd_fit.odf(sphere)

    save_nifti(save_path + '/odfs.nii.gz', mcsd_odf, affine)

    """
    dipy_peak_extraction.peak_extraction(
        save_path + '/odfs.nii.gz',
        'sphere724.txt',
        save_path + '/peaks.nii.gz',
        relative_peak_threshold=0.2,
        min_separation_angle=15.0,
        max_peak_number=3
    )
    """

    # save_to_mrtrix_format(mcsd_fit.all_shm_coeff, l_max, sphere, 3, name)


if __name__ == '__main__':
    import sys

    args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str, required=True)
    parser.add_argument('--bvals', '-va', type=str, required=True)
    parser.add_argument('--bvecs', '-ve', type=str, required=True)
    parser.add_argument('--mask', '-m', type=str, required=True)
    parser.add_argument('--order', '-o', type=int, default=8)
    parser.add_argument('--save_to', '-s', type=str, required=True)
    args = parser.parse_args(args)

    main(args.path, args.bvals, args.bvecs, args.mask, args.order, args.save_to)
