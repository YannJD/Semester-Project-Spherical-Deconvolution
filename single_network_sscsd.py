import argparse
import pathlib

import numpy as np
import torch.optim
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere, get_fnames
from dipy.denoise.localpca import mppca
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.reconst import shm
from dipy.reconst.csdeconv import auto_response_ssst
from dipy.segment.mask import median_otsu
from dipy.viz import window, actor
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import mrtrix_functions
import multi_network_sscsd
from dipy_functions import get_ss_calibration_response, compute_reg_matrix, get_ms_response, compute_kernel, \
    compute_odf_functions
from mrtrix_functions import nb_coeff
from sscsd import *
from Evaluation.dipy_peak_extraction import peak_extraction


def main(fname, bvals, bvecs, mask_path, l_max, single_fiber, save_to):
    sphere = get_sphere('symmetric724')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load dMRI volumes with all directions and b-values
    data, gtab = load_data()
    mask_path = None
    # data, gtab = load_phantom_data()
    # data, gtab = multi_network_sscsd.load_from_path(fname, bvals, bvecs)

    # If a mask is specified, use it instead of recomputing another one
    if mask_path is not None and pathlib.Path(mask_path).exists():
        mask, affine = load_nifti(mask_path)
        mask = mask.astype(bool)
    else:
        # mask = np.ones(data[..., 0].shape).astype(bool)
        # b0_mask, mask = median_otsu(data, median_radius=2, numpass=1, vol_idx=[0, 1])
        b0_mask, mask = median_otsu(data, median_radius=4, numpass=4, vol_idx=[0, 1])  # TODO: choose parameters

    nb_b0 = np.sum(gtab.bvals == 0)

    # Create architecture of the MLP
    input_size = data.shape[3] - nb_b0 if single_fiber else data.shape[3]
    output_size = nb_coeff(l_max) if single_fiber else nb_coeff(l_max) + 2
    nn_arch = [input_size, 300, 300, 300, 400, 500, 600, output_size]
    # nn_arch = [input_size, 300, 300, 300, 400, 500, 600, 700, 800, 900, 1000, output_size]

    b0_mean = np.mean(data[..., :nb_b0], 3)[mask, np.newaxis]

    # Compute the response function and the regularization matrix
    if single_fiber:
        data = mppca(data, mask=mask, patch_radius=2)
        response_fun = get_ss_calibration_response(data, gtab, l_max)
        # response_fun, ratio = auto_response_ssst(gtab, data, roi_center=[44, 24, 25], roi_radii=3, fa_thr=0.7)
        B = compute_reg_matrix(iso=0, sh_order=l_max)
        data = data[..., nb_b0:]
    else:
        denoised_data = mppca(data, mask=mask, patch_radius=2)
        response_fun = get_ms_response(data, denoised_data, mask, gtab, sphere, l_max, save_to)
        data = denoised_data
        B = compute_reg_matrix(sh_order=l_max)

    # Compute the response functions dictionary (kernel)
    kernel = compute_kernel(gtab, l_max, response_fun, single_fiber)
    B_t = B.transpose()
    M = np.linalg.inv(B_t @ B) @ B_t

    # If there are already MLP weights use them instead of retraining
    saved_weights = save_to + '/weights.pth'
    if not pathlib.Path(saved_weights).exists():
        masked_data = data[mask]
        train_network(
            masked_data,
            nn_arch,
            kernel,
            B,
            M,
            b0_mean,
            device,
            saved_weights
        )

    # 0 for single shell single tissue and 2 for WM/GM/CSF
    iso = 0 if single_fiber else 2

    # Load the MLP weights
    nn_model = SSCSD(nn_arch, kernel, B, M)
    print("Number of parameters :", sum([np.prod(p.size()) for p in nn_model.parameters()]))
    nn_model.load_state_dict(torch.load(saved_weights))

    # fODF prediction

    odf, odf_sh = compute_odf_functions(nn_model.evaluate_odf_sh,
                                        data.astype(np.float32),
                                        mask,
                                        device,
                                        l_max,
                                        sphere,
                                        iso,
                                        save_to)

    mrtrix_functions.save_to_mrtrix_format(odf_sh[..., iso:], l_max, sphere, None, save_to)

    """
    peak_extraction(
        save_to + '/odfs.nii.gz',
        'Evaluation/sphere724.txt',
        save_to + '/peaks.nii.gz',
        relative_peak_threshold=0.2,
        min_separation_angle=15.0,
        max_peak_number=3
    )"""

    plot_wm_odfs(odf, sphere)


def train_network(data, nn_arch, kernel, B, M, b0_mean, device, saved_weights):
    """
    Trains the network with the given data and network architecture.
    Saves the network weights at saved_weights when it is trained.

    :param data: 2D numpy array (voxel, dir/b-val)
    :param nn_arch: array
    :param kernel: 2D numpy array
    :param B: 2D numpy array
    :param M: 2D numpy array
    :param device: str
    :param saved_weights: str
    """

    # Create training data set. Min-max rescaling.
    # norm_data, data_min, data_max = minMaxNormalization(data)
    norm_data = b0_normalization(data, b0_mean)
    norm_signal = torch.tensor(norm_data.astype(np.float32), dtype=torch.float32)
    signal = torch.tensor(data.astype(np.float32), dtype=torch.float32)
    train_data = TensorDataset(signal, signal)
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

    nn_model = SSCSD(nn_arch, kernel, B, M)
    nn_model.to(device)

    reg_factor = 0.19
    loss_fun = RegularizedLoss(nn.MSELoss(), kernel, B, reg_factor, device)  # TODO: try other losses

    optimizer = torch.optim.RMSprop(nn_model.parameters(), lr=2e-4)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.05, patience=2, min_lr=1e-16)

    train_model(
        nn_model,
        device,
        train_loader,
        loss_fun,
        optimizer,
        lr_sched,
        epochs=30,
        load_best_model=True,
        return_loss_time=False
    )

    torch.save(nn_model.state_dict(), saved_weights)


def b0_normalization(data, b0_mean):
    return data / b0_mean


def parse_args():
    """
    Parses the command line arguments given to the program. Returns the saving file name prefix, the path of the mask
    (None if not given), the maximum order and whether to estimate single or multiple fiber.

    :returns: the file name, the mask path, the order and whether to estimate single or multiple fiber
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, required=True)
    parser.add_argument('--mask', '-m', type=str)
    parser.add_argument('--order', '-o', type=int, default=8)
    parser.add_argument('--ssst', '-s', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    fname = args.name
    mask_path = args.mask
    order = args.order
    single_fiber = args.ssst

    return fname, mask_path, order, single_fiber


def load_data():  # TODO : make different load methods
    # fraw, fbval, fbvec, t1_fname = get_fnames('cfin_multib')
    fraw, fbval, fbvec = get_fnames('stanford_hardi')
    # fraw, fbval, fbvec = get_fnames('sherbrooke_3shell')

    data, affine = load_nifti(fraw)
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)

    bvals = gtab.bvals
    bvecs = gtab.bvecs

    # sel_b = np.logical_or(np.logical_or(bvals == 0, bvals == 1000), bvals == 2000)
    sel_b = np.logical_or(bvals == 0, bvals == 2000)
    data = data[..., sel_b]

    gtab = gradient_table(bvals[sel_b], bvecs[sel_b])

    return data, gtab


def load_phantom_data():
    data, affine = load_nifti('Evaluation/Hardi_dataset/testing-data_DWIS_hardi-scheme_SNR-10.nii.gz')
    bvals, bvecs = read_bvals_bvecs('Evaluation/Hardi_dataset/hardi-scheme.bval',
                                    'Evaluation/Hardi_dataset/hardi-scheme.bvec')
    gtab = gradient_table(bvals, bvecs)

    bvals = gtab.bvals
    bvecs = gtab.bvecs

    sel_b = np.logical_or(bvals == 0, bvals == 3000)
    data = data[..., sel_b]

    gtab = gradient_table(bvals[sel_b], bvecs[sel_b])

    return data, gtab


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


def plot_wm_odfs(mcsd_odf, sphere):
    print("ODF")

    fodf_spheres = actor.odf_slicer(mcsd_odf, sphere=sphere, scale=1,
                                    norm=False, colormap='plasma')

    interactive = False
    scene = window.Scene()
    scene.add(fodf_spheres)
    scene.reset_camera_tight()

    print('Saving illustration as msdodf.png')
    window.record(scene, out_path='sscsd_output/msdodf.png', size=(1920, 1920), magnification=2)

    if interactive:
        window.show(scene)


if __name__ == '__main__':
    import sys

    args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, required=True)
    parser.add_argument('--bvals', '-va', type=str, required=True)
    parser.add_argument('--bvecs', '-ve', type=str, required=True)
    parser.add_argument('--mask', '-m', type=str)
    parser.add_argument('--order', '-o', type=int, default=8)
    parser.add_argument('--ssst', '-s', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--save_to', '-to', type=str, required=True)
    args = parser.parse_args(args)

    fname = args.name
    bvals = args.bvals
    bvecs = args.bvecs
    mask_path = args.mask
    order = args.order
    single_fiber = args.ssst
    save_to = args.save_to

    main(fname, bvals, bvecs, mask_path, order, single_fiber, save_to)
