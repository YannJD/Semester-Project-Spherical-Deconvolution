import pathlib

import numpy as np
import dipy.reconst.shm as shm
import dipy.direction.peaks as dp
import matplotlib.pyplot as plt
import torch

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
import torch.nn as nn
from sl_nn import *

import sl_nn


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    data, gtab = load_data()
    l_max = 8

    saved_weights = 'model_weights.pth'
    nn_arch = [data.shape[3], 256, 128, int((l_max + 1)*(l_max + 2)/2) + 2]

    if pathlib.Path(saved_weights).exists():
        nn_model = sl_nn.sl_nn(nn_arch)
        nn_model.load_state_dict(torch.load(saved_weights))
        f_sh = compute_fod_sh(nn_model, data, device)
        return

    sphere = get_sphere('symmetric724')

    data_2d = data_to_data_2d(data)
    response_fun = get_ms_response(data, gtab, sphere)
    kernel = torch.tensor(compute_kernel(gtab, l_max, response_fun), dtype=torch.float32)

    data_2d = torch.tensor(data_2d, dtype=torch.float32)
    train_data = TensorDataset(data_2d, data_2d)
    train_loader = DataLoader(train_data, batch_size=1024, shuffle=True)

    nn_model = sl_nn.sl_nn(nn_arch)
    nn_model.to(device)

    loss_fun = nn.MSELoss()
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=1e-6)

    train_model(
        nn_model,
        kernel,
        device,
        train_loader,
        loss_fun,
        optimizer,
        lr_sched,
        epochs=10,
        load_best_model=True,
        return_loss_time=False
    )

    torch.save(nn_model.state_dict(), saved_weights)


def compute_fod_sh(nn_model, data, device):
    f_sh = np.ndarray(data.shape[:3], dtype=object)
    for ijk in np.ndindex(data.shape[:3]):
        signal = data[ijk[0], ijk[1], ijk[2], :]
        signal = signal[np.newaxis, ...]
        f_sh[ijk[0], ijk[1], ijk[2]] = nn_model.evaluate_fod_sh(signal, device)
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
