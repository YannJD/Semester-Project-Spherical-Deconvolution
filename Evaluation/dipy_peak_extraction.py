#! /usr/bin/env python

import nibabel as nib
import numpy as np
import argparse

from dipy.core.ndindex import ndindex
#from dipy.reconst.odf import peak_directions
from dipy.core.sphere import Sphere
from dipy.direction.closest_peak_direction_getter import peak_directions


def peak_extraction(odfs_file, sphere_vertices_file, out_file, relative_peak_threshold=.5,
                    peak_normalize=1, min_separation_angle=45, max_peak_number=5):

    in_nifti = nib.load(odfs_file)
    refaff = in_nifti.affine
    odfs = in_nifti.get_fdata()

    vertices = np.loadtxt(sphere_vertices_file)
    sphere = Sphere(xyz=vertices)

    num_peak_coeffs = max_peak_number * 3
    peaks = np.zeros(odfs.shape[:-1] + (num_peak_coeffs,))

    for index in ndindex(odfs.shape[:-1]):
        vox_peaks, values, _ = peak_directions(odfs[index], sphere,
                                               float(relative_peak_threshold),
                                               float(min_separation_angle))

        if peak_normalize == 1 and values[0] != 0:
            values /= values[0]
            vox_peaks = vox_peaks * values[:, None]

        vox_peaks = vox_peaks.ravel()
        m = vox_peaks.shape[0]
        if m > num_peak_coeffs:
            m = num_peak_coeffs
        peaks[index][:m] = vox_peaks[:m]

    peaks = np.pad(peaks, [(0, 0), (0, 0), (0, 0), (0, 15 - peaks.shape[3])], mode='constant')
    peaks_img = nib.Nifti1Image(peaks.astype(np.float32), refaff)
    nib.save(peaks_img, out_file)


def buildArgsParser():
    description = 'Extract Peak Directions from Spherical function.'

    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument(action='store', dest='spherical_functions_file',
                   help='Input nifti file representing the orientation '
                        'distribution function.')
    p.add_argument(action='store', dest='sphere_vertices_file',
                   help="""Sphere vertices in a text file (Nx3)
    x1 x2 x3
     ...
    xN yN zN""")
    p.add_argument(action='store', dest='out_file',
                   help='Output nifti file with the peak directions.')
    p.add_argument('-t', '--peak_threshold', action='store',
                   dest='peak_thresh', metavar='float', required=False,
                   default=0.5, help='Relative peak threshold (default 0.5)')
    p.add_argument('-n', '--peak_normalize', action='store', dest='peak_norm',
                   metavar='int', required=False, default=1,
                   help='Normalize peaks according to spherical function '
                        'value (default 1)')
    p.add_argument('-a', '--angle', action='store', dest='angle',
                   metavar='float', required=False, default=45.0,
                   help='Minimum separation angle (default 45 degrees)')
    p.add_argument('-m', '--max_peak_number', action='store',
                   dest='max_peak_num', metavar='int', required=False,
                   default=5,
                   help='Maximum number of peaks found (default 5 peaks)')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    spherical_functions_file = args.spherical_functions_file
    sphere_vertices_file = args.sphere_vertices_file
    out_file = args.out_file

    peak_thresh = args.peak_thresh
    peak_norm = args.peak_norm
    max_peak_num = args.max_peak_num
    angle = args.angle

    peak_extraction(spherical_functions_file, sphere_vertices_file, out_file,
                    relative_peak_threshold=peak_thresh,
                    peak_normalize=int(peak_norm), min_separation_angle=angle,
                    max_peak_number=int(max_peak_num))


if __name__ == "__main__":
    main()