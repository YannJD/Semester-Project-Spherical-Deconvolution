import nibabel as nib
import numpy as np
from dipy.reconst import shm
from dipy.reconst.rumba import RumbaFit


def convert_to_mrtrix(order):
    """
    Returns the linear matrix used to convert coefficients into the mrtrix
    convention for spherical harmonics.

    :param order: int

    :returns: conversion_matrix: array-like, shape (dim_sh, dim_sh)
    """

    dim_sh = dimension(order)
    conversion_matrix = np.zeros((dim_sh, dim_sh))
    for j in range(dim_sh):
        l = sh_degree(j)
        m = sh_order(j)
        if m == 0:
            conversion_matrix[j, j] = 1
        else:
            conversion_matrix[j, j - 2 * m] = np.sqrt(2)
    return conversion_matrix


def save_to_mrtrix_format(odf_sh, l_max, sphere, affine, save_path):
    """
    Saves the spherical harmonic coefficients and the fODF evaluated on the sphere in the mrtrix format at location
    and name data_name.

    :param odf_sh: all spherical harmonics coefficients, 4D numpy array
    :param l_max: max order, int
    :param sphere: the sphere on which to evaluate the fODF
    :param save_path: the path where to save the files, str
    """

    conversion_matrix = convert_to_mrtrix(l_max)

    fods = shm.sh_to_sf(odf_sh, sphere, l_max)
    wm_vf = np.sum(fods, axis=3)
    mrtrix_sh = np.dot(odf_sh, conversion_matrix.T) * wm_vf[..., np.newaxis]
    sh_img = nib.Nifti1Image(mrtrix_sh, affine)

    nib.save(sh_img, save_path + "/mrtrix_sh.nii.gz")


def sh_order(j):
    """
    Returns the order, ``m``, of the spherical harmonic associated to index
    ``j``.

    Parameters
    ----------
    j : int
        The flattened index of the spherical harmonic.

    Returns
    -------
    m : int
        The associated order.
    """
    l = sh_degree(j)
    return j + l + 1 - dimension(l)


def sh_degree(j):
    """
    Returns the degree, ``l``, of the spherical harmonic associated to index
    ``j``.

    Parameters
    ----------
    j : int
        The flattened index of the spherical harmonic.

    Returns
    -------
    l : int
        The associated even degree.
    """
    l = 0
    while dimension(l) - 1 < j:
        l += 2
    return l


def dimension(order):
    r"""
    Returns the dimension, :math:`R`, of the real, antipodally symmetric
    spherical harmonics basis for a given truncation order.

    Parameters
    ----------
    order : int
        The trunction order.

    Returns
    -------
    R : int
        The dimension of the truncated spherical harmonics basis.
    """
    return ((order + 1) * (order + 2)) // 2


def nb_coeff(order):
    """
    Returns the number of spherical harmonics coefficients for a given max order.

    :param order: int
    :return: the number of coefficients, int
    """

    return int((order + 1) * (order + 2) / 2)
