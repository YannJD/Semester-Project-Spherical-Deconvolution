import argparse
import pathlib

import numpy as np
import torch.optim
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere, get_fnames
from dipy.denoise.localpca import mppca
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.reconst.csdeconv import auto_response_ssst
from dipy.segment.mask import median_otsu
from dipy.viz import window, actor
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import single_network_sscsd
from dipy_functions import get_ss_calibration_response, compute_reg_matrix, get_ms_response, compute_kernel, \
    compute_odf_functions
from mrtrix_functions import nb_coeff
from sscsd import *


def main():
    # Parse command line arguments passed
    data_name, mask_path, l_max, single_fiber = single_network_sscsd.parse_args()
    data_name = 'sscsd_output/' + data_name

    sphere = get_sphere('symmetric724')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load dMRI volumes with all directions and b-values
    # data, gtab = load_data()
    data, gtab = single_network_sscsd.load_phantom_data()

    # If a mask is specified, use it instead of recomputing another one
    if mask_path is not None and pathlib.Path(mask_path).exists():
        mask, affine = load_nifti(mask_path)
        mask = mask.astype(bool)
    else:
        b0_mask, mask = median_otsu(data, median_radius=2, numpass=1, vol_idx=[0, 1])
        # b0_mask, mask = median_otsu(data, median_radius=4, numpass=4, vol_idx=[0, 1])  # TODO: choose parameters

    nb_b0 = np.sum(gtab.bvals == 0)

    # Create architecture of the MLP
    input_size = data.shape[3] - nb_b0 if single_fiber else data.shape[3]
    arch = network_architecture(input_size, l_max, single_fiber)

    # Compute the response function and the regularization matrix
    if single_fiber:
        data = mppca(data, mask=mask, patch_radius=2)
        response_fun = get_ss_calibration_response(data, gtab, l_max)
        # response_fun, ratio = auto_response_ssst(gtab, data, roi_center=[44, 24, 25], roi_radii=3, fa_thr=0.7)
        B = compute_reg_matrix(iso=0, sh_order=l_max)
        data = data[..., nb_b0:]
    else:
        denoised_data = mppca(data, mask=mask, patch_radius=2)
        response_fun = get_ms_response(data, denoised_data, mask, gtab, sphere, l_max, data_name)
        data = denoised_data
        B = compute_reg_matrix(sh_order=l_max)

    # Compute the response functions dictionary (kernel)
    kernel = compute_kernel(gtab, l_max, response_fun, single_fiber)
    B_t = B.transpose()
    M = np.linalg.inv(B_t @ B) @ B_t

    # If there are already MLP weights use them instead of retraining
    saved_weights = data_name + '_weights.pth'
    if not pathlib.Path(saved_weights).exists():
        masked_data = data[mask]
        train_network(
            masked_data,
            arch,
            kernel,
            B,
            device,
            saved_weights
        )

    # 0 for single shell single tissue and 2 for WM/GM/CSF
    iso = 0 if single_fiber else 2

    # Load the MLP weights
    nn_model = MultiNetworkSSCSD(arch, device)
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
                                        data_name)

    single_network_sscsd.plot_wm_odfs(odf, sphere)


def train_network(data, nn_arch, kernel, B, device, saved_weights):
    """
    Trains the network with the given data and network architecture.
    Saves the network weights at saved_weights when it is trained.

    :param data: 2D numpy array (voxel, dir/b-val)
    :param nn_arch: array
    :param kernel: 2D numpy array
    :param B: 2D numpy array
    :param l_max: max sh order
    :param device: str
    :param saved_weights: str
    """

    # Create training data set. Min-max rescaling.
    norm_data, data_min, data_max = minMaxNormalization(data)
    norm_signal = torch.tensor(norm_data.astype(np.float32), dtype=torch.float32)
    signal = torch.tensor(data.astype(np.float32), dtype=torch.float32)
    train_data = TensorDataset(signal, signal)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    nn_model = MultiNetworkSSCSD(nn_arch, device)
    nn_model.to(device)

    reg_factor = 2e-1
    loss_fun = RegularizedLoss(nn.L1Loss(), kernel, B, reg_factor, device)  # TODO: try other losses

    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=2e-4)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.05, patience=2, min_lr=1e-8)

    train_model(
        nn_model,
        device,
        train_loader,
        loss_fun,
        optimizer,
        lr_sched,
        epochs=20,
        load_best_model=True,
        return_loss_time=False
    )

    torch.save(nn_model.state_dict(), saved_weights)


def network_architecture(input_size, lmax, is_ssst):
    """
    Create a multi network SSCSD architecture.

    :param input_size: the data size
    :param lmax: the max spherical harmonic order
    :param is_ssst: if we are doing a ssst estimation
    :return: a dictionary of all networks architectures associated with the sh order
    """

    # We use the same basis for all networks, only the output size changes
    standard_arch = [input_size, 300, 300, 300, 400, 500, 600, 700, 800, 900, 1000]
    # standard_arch = [input_size, 300, 300, 300, 400, 500, 1000, 1100, 1200, 1300, 1400]
    arch = {}
    for l in range(0, lmax + 1, 2):
        output_size = 2 * l + 1
        # If we are doing multi-shell, multi-tissue estimation, we estimate 2 more sh coefficients
        if (not is_ssst) and l == 0:
            output_size += 2
        standard_arch_extended = standard_arch.copy()
        standard_arch_extended.append(output_size)
        arch[l] = standard_arch_extended
    return arch


if __name__ == '__main__':
    main()
