import argparse
import pathlib
import warnings

import dipy.reconst.shm as shm
import dipy.direction.peaks as dp
import numpy as np
import torch.optim
from dipy.core.geometry import cart2sphere

from dipy.denoise.localpca import mppca
from dipy.core.gradients import gradient_table, unique_bvals_tolerance
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.reconst import dti
from dipy.reconst.csdeconv import recursive_response, estimate_response, AxSymShResponse, \
    ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.reconst.dti import fractional_anisotropy
from dipy.segment.mask import median_otsu
from dipy.reconst.mcsd import (auto_response_msmt,
                               mask_for_response_msmt,
                               response_from_mask_msmt, multi_tissue_basis, _inflate_response, _basic_delta)
from dipy.segment.tissue import TissueClassifierHMRF
from dipy.reconst.mcsd import MultiShellDeconvModel, multi_shell_fiber_response
from dipy.viz import window, actor

from dipy.data import get_sphere, get_fnames, small_sphere

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sl_nn import *
from dipy.core import geometry as geo
from dipy.data import default_sphere
import nibabel as nib

import sl_nn


def main():
    single_fiber = True  # TODO: use argParser
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    #data, gtab = load_phantom_data()  # TODO: use real data
    data, gtab = load_data()
    #b0_mask, mask = median_otsu(data, median_radius=2, numpass=1, vol_idx=[0, 1])
    b0_mask, mask = median_otsu(data, median_radius=4, numpass=4, vol_idx=[0, 1])
    #if pathlib.Path('Evaluation/mask.nii').exists():
        #mask, affine = load_nifti('Evaluation/mask.nii')
        #mask = mask.astype(bool)
    l_max = 8

    saved_weights = 'model_weights.pth'
    b0_nb = 1  # TODO: add cleaner way
    input_size = data.shape[3] - b0_nb if single_fiber else data.shape[3]
    output_size = int((l_max + 1) * (l_max + 2) / 2) if single_fiber else int((l_max + 1) * (l_max + 2) / 2) + 2
    nn_arch = [input_size, 256, 128, output_size]

    sphere = get_sphere('symmetric724')

    if single_fiber:
        response_fun = get_ss_response(data, gtab)
        B = compute_reg_matrix(iso=0)
        data = data[..., b0_nb:]
    else:
        response_fun = get_ms_response(data, mask, gtab, sphere, l_max)
        B = compute_reg_matrix()

    kernel = compute_kernel(gtab, l_max, response_fun, single_fiber)
    B_t = B.transpose()
    M = np.linalg.inv(B_t @ B) @ B_t

    if not pathlib.Path(saved_weights).exists():
        train_network(
            data,
            mask,
            nn_arch,
            kernel,
            B,
            M,
            device,
            saved_weights
        )

    iso = 0 if single_fiber else 2
    odf, odf_sh = compute_odf_function(nn_arch, kernel, B, M, saved_weights, data.astype(np.float32), mask, device, l_max, sphere, iso)
    # print("Number of negative values :", len(odf[odf < 0]))
    # compare_MSE(response_fun, odf_sh.numpy(), data, gtab, l_max, single_fiber)
    # plot_wm_odfs(odf[:, :, 22:23, :], sphere)


def compare_MSE(response, nn_f, signal, gtab, l_max, single_fiber):
    classic_f, affine = load_nifti('CSD_example/mrtrix_sh.nii.gz')
    kernel = compute_kernel(gtab, l_max, response, single_fiber).T
    pred = classic_f @ kernel
    loss_fun = nn.MSELoss()

    signal = torch.tensor(signal, dtype=torch.float32)
    pred = torch.tensor(pred, dtype=torch.float32)
    print("MSE classic:", loss_fun(pred, signal))

    nn_pred = nn_f @ kernel
    nn_pred = torch.tensor(nn_pred, dtype=torch.float32)
    print("MSE neural network:", loss_fun(nn_pred, signal))


def convert_to_mrtrix(order):
    """
    Returns the linear matrix used to convert coefficients into the mrtrix
    convention for spherical harmonics.

    Parameters
    ----------
    order : int

    Returns
    -------
    conversion_matrix : array-like, shape (dim_sh, dim_sh)
    """
    dim_sh = int((order + 1) * (order + 2) / 2)
    conversion_matrix = np.zeros((dim_sh, dim_sh))
    for j in range(dim_sh):
        # l = sh_degree(j)
        m = sh_order(j)
        if m == 0:
            conversion_matrix[j, j] = 1
        else:
            conversion_matrix[j, j - 2 * m] = np.sqrt(2)
    return conversion_matrix


def sh_order(j):
    t = int(round((np.sqrt(8 * j + 9) - 3) / 2))
    return int(shm.order_from_ncoef(int((t + 1) * (t + 2) / 2)))


def compute_odf_function(nn_arch, H, B, M, saved_weights, data, mask, device, l_max, sphere, iso):
    nn_model = sl_nn.sl_nn(nn_arch, H, B, M)
    print("Number of parameters :", sum([np.prod(p.size()) for p in nn_model.parameters()]))
    nn_model.load_state_dict(torch.load(saved_weights))

    odf_sh = compute_odf_sh(nn_model, data, mask, device, l_max, iso)
    wm_sh = odf_sh[:, :, :, iso:]  # TODO: single shell
    mcsd_odf = shm.sh_to_sf(wm_sh, sphere, l_max)
    # csf_odf = shm.sh_to_sf(odf_sh[:, :, :, 0:1], sphere, 0)
    # gm_odf = shm.sh_to_sf(odf_sh[:, :, :, 1:2], sphere, 1)

    save_to_mrtrix_format(odf_sh.numpy(), wm_sh.numpy(), l_max, sphere, iso + 1)  # TODO: single shell
    odf_img = nib.Nifti1Image(mcsd_odf, None)
    nib.save(odf_img, "odfs.nii.gz")

    return mcsd_odf, odf_sh


def save_to_mrtrix_format(odf_sh, wm_sh, l_max, sphere, tissue_classes):
    sh_const = .5 / np.sqrt(np.pi)
    vf = odf_sh[..., :tissue_classes] / sh_const
    wm_vf = vf[..., -1]
    conversion_matrix = convert_to_mrtrix(l_max)
    mrtrix_sh = np.dot(wm_sh, conversion_matrix.T) * wm_vf[..., np.newaxis]
    sh_img = nib.Nifti1Image(mrtrix_sh, None)
    nib.save(sh_img, "mrtrix_sh.nii.gz")
    new_odf = shm.sh_to_sf(mrtrix_sh, sphere, l_max)
    fods_img = nib.Nifti1Image(new_odf, None)
    nib.save(fods_img, "mrtrix_odfs.nii.gz")


def train_network(data, mask, nn_arch, kernel, B, M, device, saved_weights):
    masked_data = data[mask]
    signal = torch.tensor(masked_data.astype(np.float32), dtype=torch.float32)
    train_data = TensorDataset(signal, signal)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

    nn_model = sl_nn.sl_nn(nn_arch, kernel, B, M)
    nn_model.to(device)

    # loss_fun = sl_nn.ConstrainedMSE(kernel, B, M, device)

    reg_factor = 1
    loss_fun = sl_nn.RegularizedMSE(kernel, B, reg_factor, device)

    # loss_fun = sl_nn.CustomMSE(kernel, device)

    optimizer = torch.optim.RMSprop(nn_model.parameters(), lr=0.001)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=1e-10)

    train_model(
        nn_model,
        device,
        train_loader,
        loss_fun,
        optimizer,
        lr_sched,
        epochs=200,
        load_best_model=True,
        return_loss_time=False
    )

    torch.save(nn_model.state_dict(), saved_weights)


def plot_wm_odfs(mcsd_odf, sphere):
    print("ODF")

    fodf_spheres = actor.odf_slicer(mcsd_odf, sphere=sphere, scale=1,
                                    norm=False, colormap='plasma')

    interactive = False
    scene = window.Scene()
    scene.add(fodf_spheres)
    scene.reset_camera_tight()

    print('Saving illustration as msdodf.png')
    window.record(scene, out_path='msdodf.png', size=(1920, 1920), magnification=2)

    if interactive:
        window.show(scene)


def compute_reg_matrix(reg_sphere=default_sphere, iso=2, sh_order=8):
    if iso == 0:
        m, n = shm.sph_harm_ind_list(sh_order)
        sphere = reg_sphere or small_sphere
        r, theta, phi = cart2sphere(sphere.x, sphere.y, sphere.z)
        return shm.real_sh_descoteaux_from_index(m, n, theta[:, None], phi[:, None])

    r, theta, phi = geo.cart2sphere(*reg_sphere.vertices.T)
    odf_reg, _, _ = shm.real_sh_descoteaux(sh_order, theta, phi)
    reg = np.zeros([i + iso for i in odf_reg.shape])
    reg[:iso, :iso] = np.eye(iso)
    reg[iso:, iso:] = odf_reg
    return reg


def compute_odf_sh(nn_model, data, mask, device, l_max, iso):
    size = int((l_max + 1) * (l_max + 2) / 2) if iso == 0 else int((l_max + 1) * (l_max + 2) / 2) + 2
    f_sh = torch.empty(data.shape[:3] + (size,), dtype=torch.float64)
    f_sh = f_sh.to(device)
    for ijk in np.ndindex(data.shape[:3]):
        signal = data[ijk[0], ijk[1], ijk[2], :]
        signal = signal[np.newaxis, ...]
        if mask[ijk[0], ijk[1], ijk[2]]:
            f_sh[ijk[0], ijk[1], ijk[2]] = nn_model.evaluate_odf_sh(signal, device)

    return torch.Tensor.cpu(f_sh)


def compute_kernel(gtab, l_max, response_fun, single_shell):
    if single_shell:
        csd_model = ConstrainedSphericalDeconvModel(gtab, response_fun)
        kernel = csd_model._X
        return kernel

    msmt_Y, m, n = multi_tissue_basis(gtab, l_max, 2)
    delta = _basic_delta(response_fun.iso, response_fun.m, response_fun.n, 0., 0.)
    msmt_H = _inflate_response(response_fun, gtab, n, delta)

    return msmt_Y * msmt_H


def load_data():
    # fraw, fbval, fbvec, t1_fname = get_fnames('cfin_multib')
    fraw, fbval, fbvec = get_fnames('stanford_hardi')
    # fraw, fbval, fbvec = get_fnames('sherbrooke_3shell')

    data, affine = load_nifti(fraw)
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)

    bvals = gtab.bvals
    bvecs = gtab.bvecs

    sel_b = np.logical_or(np.logical_or(bvals == 0, bvals == 1000), bvals == 2000)
    # sel_b = np.logical_or(bvals == 0, bvals == 2000)
    data = data[..., sel_b]

    gtab = gradient_table(bvals[sel_b], bvecs[sel_b])

    return data, gtab


def load_phantom_data():
    data, affine = load_nifti('Evaluation/Hardi_dataset/testing-data_DWIS_hardi-scheme_SNR-30.nii.gz')
    bvals, bvecs = read_bvals_bvecs('Evaluation/Hardi_dataset/hardi-scheme.bval', 'Evaluation/Hardi_dataset/hardi'
                                                                                  '-scheme.bvec')
    gtab = gradient_table(bvals, bvecs)

    bvals = gtab.bvals
    bvecs = gtab.bvecs

    sel_b = np.logical_or(bvals == 0, bvals == 3000)
    data = data[..., sel_b]

    gtab = gradient_table(bvals[sel_b], bvecs[sel_b])

    return data, gtab


def get_ss_response(data, gtab):
    response, ratio = auto_response_ssst(gtab, data, roi_center=[44, 24, 25], roi_radii=3, fa_thr=0.7)
    # response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
    return response

    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask=data[..., 0] > 200)

    FA = fractional_anisotropy(tenfit.evals)
    MD = dti.mean_diffusivity(tenfit.evals)
    wm_mask = (np.logical_or(FA >= 0.4, (np.logical_and(FA >= 0.15, MD >= 0.0011))))

    return recursive_response(gtab, data, mask=wm_mask, sh_order=8,
                              peak_thr=0.01, init_fa=0.08,
                              init_trace=0.0021, iter=8, convergence=0.001,
                              parallel=False, num_processes=1)


def get_ms_response(data, mask, gtab, sphere, l_max):
    ubvals = unique_bvals_tolerance(gtab.bvals)

    if pathlib.Path('response_sh.npy').exists():
        with open('response_sh.npy', 'rb') as f:
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

    nvoxels_wm = np.sum(mask_wm)
    nvoxels_gm = np.sum(mask_gm)
    nvoxels_csf = np.sum(mask_csf)

    response_wm, response_gm, response_csf = response_from_mask_msmt(gtab, data,
                                                                     mask_wm,
                                                                     mask_gm,
                                                                     mask_csf)

    with open('response_sh.npy', 'wb') as f:
        np.save(f, [response_wm, response_gm, response_csf])

    response_mcsd = multi_shell_fiber_response(sh_order=l_max,
                                               bvals=ubvals,
                                               wm_rf=response_wm,
                                               gm_rf=response_gm,
                                               csf_rf=response_csf)

    return response_mcsd


if __name__ == '__main__':
    main()
