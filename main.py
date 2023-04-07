import os
import pathlib

import dipy.reconst.shm as shm
import dipy.direction.peaks as dp
import torch.optim

from dipy.denoise.localpca import mppca
from dipy.core.gradients import gradient_table, unique_bvals_tolerance
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.segment.mask import median_otsu
from dipy.reconst.mcsd import (auto_response_msmt,
                               mask_for_response_msmt,
                               response_from_mask_msmt, multi_tissue_basis, _inflate_response, _basic_delta)
from dipy.segment.tissue import TissueClassifierHMRF
from dipy.reconst.mcsd import MultiShellDeconvModel, multi_shell_fiber_response
from dipy.viz import window, actor

from dipy.data import get_sphere, get_fnames

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sl_nn import *
from dipy.core import geometry as geo
from dipy.data import default_sphere
import nibabel as nib

import sl_nn


# TODO: train with phantom (SNR = 30), compare peaks


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    data, gtab = load_data()
    b0_mask, mask = median_otsu(data, median_radius=2, numpass=1, vol_idx=[0, 1])
    l_max = 8

    saved_weights = 'model_weights.pth'
    #nn_arch = [data.shape[3], 256, 128, int((l_max + 1) * (l_max + 2) / 2) + 2]
    nn_arch = [data.shape[3], 256, 128, int((l_max + 1) * (l_max + 2) / 2) + 2]

    sphere = get_sphere('symmetric724')

    if not pathlib.Path(saved_weights).exists():
        response_fun = get_ms_response(data, mask, gtab, sphere)

        train_network(
            data,
            mask,
            nn_arch,
            device,
            gtab,
            l_max,
            response_fun,
            saved_weights
        )

    mcsd_odf = compute_odf_function(nn_arch, saved_weights, data, mask, device, l_max, sphere)
    plot_wm_odfs(mcsd_odf[:, :, 10:11, :], sphere)

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
    """
    for j in range(dim_sh):
        #l = sh_degree(j)
        m = sh_order(j)
        if m == 0:
            conversion_matrix[j, j] = 1
        else:
            conversion_matrix[j, j - 2*m] = np.sqrt(2)"""
    return conversion_matrix


def compute_odf_function(nn_arch, saved_weights, data, mask, device, l_max, sphere):
    nn_model = sl_nn.sl_nn(nn_arch)
    print("Number of parameters : ", sum([np.prod(p.size()) for p in nn_model.parameters()]))
    nn_model.load_state_dict(torch.load(saved_weights))

    odf_sh = compute_odf_sh(nn_model, data, mask, device, l_max)
    mcsd_odf = shm.sh_to_sf(odf_sh[:, :, :, 2:odf_sh.shape[-1]], sphere, l_max)
    #csf_odf = shm.sh_to_sf(odf_sh[:, :, :, 0:1], sphere, 0)
    #gm_odf = shm.sh_to_sf(odf_sh[:, :, :, 1:2], sphere, 1)

    """
    conversion_matrix = convert_to_mrtrix(l_max)
    fods_img = nib.Nifti1Image(np.dot(mcsd_odf, conversion_matrix.T)
                               * wm_vf[..., np.newaxis], None)
    nib.save(fods_img, "fods.nii.gz")"""

    return mcsd_odf


def train_network(data, mask, nn_arch, device, gtab, l_max, response_fun, saved_weights):
    masked_data = data[mask]
    # data_2d = data_to_data_2d(masked_data)
    data_2d = torch.tensor(masked_data, dtype=torch.float32)
    train_data = TensorDataset(data_2d, data_2d)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    nn_model = sl_nn.sl_nn(nn_arch)
    nn_model.to(device)

    kernel = compute_kernel(gtab, l_max, response_fun)
    B = compute_reg_matrix()
    B_t = B.transpose()
    M = np.linalg.inv(B_t @ B) @ B_t
    #loss_fun = sl_nn.ConstrainedMSE(kernel, B, M, device)

    reg_factor = 1
    loss_fun = sl_nn.RegularizedMSE(kernel, B, reg_factor, device)

    optimizer = torch.optim.RMSprop(nn_model.parameters(), lr=0.001)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=1e-10)

    train_model(
        nn_model,
        device,
        train_loader,
        loss_fun,
        optimizer,
        lr_sched,
        epochs=50,
        load_best_model=True,
        return_loss_time=False
    )

    torch.save(nn_model.state_dict(), saved_weights)


def plot_wm_odfs(mcsd_odf, sphere):
    print("ODF")
    print(mcsd_odf.shape)
    print(mcsd_odf[50, 50, 0])

    fodf_spheres = actor.odf_slicer(mcsd_odf, sphere=sphere, scale=1.2,
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
    r, theta, phi = geo.cart2sphere(*reg_sphere.vertices.T)
    odf_reg, _, _ = shm.real_sh_descoteaux(sh_order, theta, phi)
    reg = np.zeros([i + iso for i in odf_reg.shape])
    reg[:iso, :iso] = np.eye(iso)
    reg[iso:, iso:] = odf_reg
    return reg


def compute_odf_sh(nn_model, data, mask, device, l_max):
    f_sh = np.ndarray(data.shape[:3] + (int((l_max + 1) * (l_max + 2) / 2) + 2,), dtype=np.float64)
    for ijk in np.ndindex(data.shape[:3]):
        signal = data[ijk[0], ijk[1], ijk[2], :]
        signal = signal[np.newaxis, ...]
        if mask[ijk[0], ijk[1], ijk[2]]:
            f_sh[ijk[0], ijk[1], ijk[2]] = torch.Tensor.cpu(nn_model.evaluate_odf_sh(signal, device))

    return f_sh


def compute_kernel(gtab, l_max, response_fun):
    msmt_Y, m, n = multi_tissue_basis(gtab, l_max, 2)
    delta = _basic_delta(response_fun.iso, response_fun.m, response_fun.n, 0., 0.)
    msmt_H = _inflate_response(response_fun, gtab, n, delta)

    return msmt_Y * msmt_H


def data_to_data_2d(data: np.ndarray):
    data_2d = data.reshape((data.shape[0] * data.shape[1] * data.shape[2], data.shape[3]))
    return data_2d


def data_2d_to_data(data_2d: np.ndarray):
    data = data_2d.reshape((96, 96, 19, data_2d.shape[1]))
    return data


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


def get_ms_response(data, mask, gtab, sphere):
    wm_sh = 'wm_sh.csv'
    gm_sh = 'gm_sh.csv'
    csf_sh = 'csf_sh.csv'

    ubvals = unique_bvals_tolerance(gtab.bvals)

    if pathlib.Path(wm_sh).exists() and pathlib.Path(gm_sh).exists() and pathlib.Path(csf_sh).exists():
        response_wm = np.loadtxt(wm_sh, delimiter=',')
        response_gm = np.loadtxt(gm_sh, delimiter=',')
        response_csf = np.loadtxt(csf_sh, delimiter=',')
        response_mcsd = multi_shell_fiber_response(sh_order=8,
                                                   bvals=ubvals,
                                                   wm_rf=response_wm,
                                                   gm_rf=response_gm,
                                                   csf_rf=response_csf)

        return response_mcsd

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

    np.savetxt(wm_sh, response_wm, delimiter=',')
    np.savetxt(gm_sh, response_gm, delimiter=',')
    np.savetxt(csf_sh, response_csf, delimiter=',')

    response_mcsd = multi_shell_fiber_response(sh_order=8,
                                               bvals=ubvals,
                                               wm_rf=response_wm,
                                               gm_rf=response_gm,
                                               csf_rf=response_csf)

    return response_mcsd


if __name__ == '__main__':
    main()
