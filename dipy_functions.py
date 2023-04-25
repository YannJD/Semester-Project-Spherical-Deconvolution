import pathlib

import dipy.direction.peaks as dp
import nibabel as nib
import numpy as np
import torch
from dipy.core.geometry import cart2sphere
from dipy.core.gradients import unique_bvals_tolerance
from dipy.data import default_sphere, small_sphere
from dipy.denoise.localpca import mppca
from dipy.reconst import shm, dti
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, recursive_response
from dipy.reconst.dti import fractional_anisotropy
from dipy.reconst.mcsd import multi_tissue_basis, _basic_delta, _inflate_response, multi_shell_fiber_response, \
    mask_for_response_msmt, response_from_mask_msmt
from dipy.segment.tissue import TissueClassifierHMRF

from mrtrix_functions import nb_coeff, save_to_mrtrix_format


def compute_odf_functions(evaluate_odf_sh, data, mask, device, l_max, sphere, iso, data_name):
    """
    Compute the fODF for all voxels.

    :param evaluate_odf_sh: evaluates the fODF SH for all voxels, function
    :param data: the dMRI volumes, 4D numpy array
    :param mask: 3D numpy array
    :param device: str
    :param l_max: int
    :param sphere: the sphere on which to evaluate the fODFs
    :param iso: the number of tissue types minus one, int
    :param data_name: the saving file names prefix, str
    :returns: the fODFs and the fODF SH coefficients
    """

    odf_sh = compute_odfs_sh(evaluate_odf_sh, data, mask, device, l_max, iso)
    odfs = shm.sh_to_sf(odf_sh[..., iso:], sphere, l_max)

    save_to_mrtrix_format(odf_sh.numpy(), l_max, sphere, iso + 1, data_name)

    odf_img = nib.Nifti1Image(odfs, None)
    nib.save(odf_img, data_name + "_odfs.nii.gz")

    return odfs, odf_sh


def compute_odfs_sh(evaluate_odf_sh, data, mask, device, l_max, iso):
    """
    Computes the SH coefficients of all voxel's fODF.

    :param evaluate_odf_sh: the function used to evaluate the SH coefficients
    :param data: 4D numpy array
    :param mask: 3D numpy array
    :param device: str
    :param l_max: int
    :param iso: the number of tissues minus one
    :returns: the odfs SH coefficients
    """

    size = nb_coeff(l_max) if iso == 0 else nb_coeff(l_max) + 2
    odfs_sh = torch.empty(data.shape[:3] + (size,), dtype=torch.float64)
    odfs_sh = odfs_sh.to(device)

    # Compute SH for each voxel
    for ijk in np.ndindex(data.shape[:3]):  # TODO : speedup loop
        signal = data[ijk[0], ijk[1], ijk[2], :]
        signal = signal[np.newaxis, ...]
        # Evaluate only if WM
        if mask[ijk[0], ijk[1], ijk[2]]:
            odfs_sh[ijk[0], ijk[1], ijk[2]] = evaluate_odf_sh(signal, device)

    return torch.Tensor.cpu(odfs_sh)


def compute_reg_matrix(reg_sphere=default_sphere, iso=2, sh_order=8):
    """
    Computes the regularization matrix (matrix B)

    :param reg_sphere: HemiSphere
    :param iso: the number of tissue types minus one, int
    :param sh_order: int
    :returns: the regularization matrix
    """

    if iso == 0:
        m, n = shm.sph_harm_ind_list(sh_order)
        sphere = reg_sphere or small_sphere
        r, theta, phi = cart2sphere(sphere.x, sphere.y, sphere.z)
        return shm.real_sh_descoteaux_from_index(m, n, theta[:, None], phi[:, None])

    r, theta, phi = cart2sphere(*reg_sphere.vertices.T)
    odf_reg, _, _ = shm.real_sh_descoteaux(sh_order, theta, phi)
    reg = np.zeros([i + iso for i in odf_reg.shape])
    reg[:iso, :iso] = np.eye(iso)
    reg[iso:, iso:] = odf_reg
    return reg


def compute_kernel(gtab, l_max, response_fun, single_shell):
    """
    Computes the response functions dictionary (kernel).

    :param gtab: GradientTable
    :param l_max: max SH order, int
    :param response_fun: the response function
    :param single_shell: whether to estimate for single or multiple tissues
    :return: the response function dictionary
    """

    if single_shell:
        csd_model = ConstrainedSphericalDeconvModel(gtab, response_fun)
        return csd_model._X

    msmt_Y, m, n = multi_tissue_basis(gtab, l_max, 2)
    delta = _basic_delta(response_fun.iso, response_fun.m, response_fun.n, 0., 0.)
    msmt_H = _inflate_response(response_fun, gtab, n, delta)

    return msmt_Y * msmt_H


def get_ss_calibration_response(data, gtab, l_max):
    """
    Computes the response function from data calibration.

    :param data: 4D numpy array
    :param gtab: GradientTable
    :param l_max: max SH order, int
    :return: the response function
    """

    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask=data[..., 0] > 200)

    FA = fractional_anisotropy(tenfit.evals)
    MD = dti.mean_diffusivity(tenfit.evals)
    wm_mask = (np.logical_or(FA >= 0.4, (np.logical_and(FA >= 0.15, MD >= 0.0011))))

    return recursive_response(gtab, data, mask=wm_mask, sh_order=l_max,
                              peak_thr=0.01, init_fa=0.08,
                              init_trace=0.0021, iter=8, convergence=0.001,
                              parallel=False, num_processes=1)


def get_ms_response(data, mask, gtab, sphere, l_max, data_name):
    """
    Computes the multi-shell multi-tissue response function.

    :param data: 4D numpy array
    :param mask: 3D numpy array
    :param gtab: GradientTable
    :param sphere: a sphere
    :param l_max: the max SH order, int
    :param data_name: the saving file name prefix, str
    :return: the multi-shell multi-tissue response function
    """

    ubvals = unique_bvals_tolerance(gtab.bvals)
    response_path = data_name + '_response.npy'

    # If it is already computed, load the existing response function
    if pathlib.Path(response_path).exists():
        with open(response_path, 'rb') as f:
            [response_wm, response_gm, response_csf] = np.load(f)
            response_mcsd = multi_shell_fiber_response(sh_order=l_max,
                                                       bvals=ubvals,
                                                       wm_rf=response_wm,
                                                       gm_rf=response_gm,
                                                       csf_rf=response_csf)

            return response_mcsd

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

    mask_wm *= wm
    mask_gm *= gm
    mask_csf *= csf

    response_wm, response_gm, response_csf = response_from_mask_msmt(gtab, data,
                                                                     mask_wm,
                                                                     mask_gm,
                                                                     mask_csf)

    # Save response function for future uses
    with open(response_path, 'wb') as f:
        np.save(f, [response_wm, response_gm, response_csf])

    response_mcsd = multi_shell_fiber_response(sh_order=l_max,
                                               bvals=ubvals,
                                               wm_rf=response_wm,
                                               gm_rf=response_gm,
                                               csf_rf=response_csf)

    return response_mcsd
