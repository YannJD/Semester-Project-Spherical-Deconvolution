import nibabel as nib
import numpy as np
from dipy.reconst import shm


def convert_to_mrtrix(order):
    """
    Returns the linear matrix used to convert coefficients into the mrtrix
    convention for spherical harmonics.

    :param order: int

    :returns: conversion_matrix: array-like, shape (dim_sh, dim_sh)
    """

    dim_sh = nb_coeff(order)
    conversion_matrix = np.zeros((dim_sh, dim_sh))
    for j in range(dim_sh):
        # l = sh_degree(j)
        m = sh_order(j)
        if m == 0:
            conversion_matrix[j, j] = 1
        else:
            conversion_matrix[j, j - 2 * m] = np.sqrt(2)
    return conversion_matrix


def save_to_mrtrix_format(odf_sh, l_max, sphere, tissue_classes, save_path):
    """
    Saves the spherical harmonic coefficients and the fODF evaluated on the sphere in the mrtrix format at location
    and name data_name.

    :param odf_sh: all spherical harmonics coefficients, 4D numpy array
    :param l_max: max order, int
    :param sphere: the sphere on which to evaluate the fODF
    :param tissue_classes: the number of tissues to evaluate (WM/GM/CSF for example), int
    :param save_path: the path where to save the files, str
    """

    sh_const = .5 / np.sqrt(np.pi)
    vf = odf_sh[..., :tissue_classes] / sh_const
    wm_vf = vf[..., -1]
    conversion_matrix = convert_to_mrtrix(l_max)

    mrtrix_sh = np.dot(odf_sh[..., tissue_classes - 1:], conversion_matrix.T) * wm_vf[..., np.newaxis]
    sh_img = nib.Nifti1Image(mrtrix_sh, None)
    nib.save(sh_img, save_path + "/mrtrix_sh.nii.gz")

    new_odf = shm.sh_to_sf(mrtrix_sh, sphere, l_max)
    fods_img = nib.Nifti1Image(new_odf, None)
    nib.save(fods_img, save_path + "/mrtrix_odfs.nii.gz")


def sh_order(j):
    """
    Return the spherical harmonic order from index.

    :param j: int
    :return: the order, int
    """

    t = int(round((np.sqrt(8 * j + 9) - 3) / 2))
    return int(shm.order_from_ncoef(nb_coeff(t)))


def nb_coeff(order):
    """
    Returns the number of spherical harmonics coefficients for a given max order.

    :param order: int
    :return: the number of coefficients, int
    """

    return int((order + 1) * (order + 2) / 2)
