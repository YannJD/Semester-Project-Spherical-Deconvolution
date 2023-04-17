#
# This script computes standard scores about local reconstruction quality (detail on http://hardi.epfl.ch/static/events/2013_ISBI/).
# USAGE:
#
# Put all your reconstructions (in nii.gz format as specified on the website) in a folder "reconstructions".
# The script opens each reconstruction file and stores the corresponding scores in a different row of an output spreadsheet file (local_masures.xls)
#
# The scripts expects two files to be in the current folder:
#   - ground-truth-peaks.nii.gz  -->  contains the ground truth peaks provided by the organizers
#   - mask.nii.gz  -->  a binary mask to restrict the computation of the metrics in specific regions of the dataset (e.g. white matter voxels)
#
# For more informations or concerns about this script, please send an email to the organizers.
#
import nibabel as nib
import numpy as np
from numpy.linalg import norm
from scipy import stats
from math import *
import sys, glob, xlwt, os.path, re

### load ground-truth directions
################################
print("-> opening ground-truth...", end=' ')

niiGT = nib.load('ground-truth-peaks.nii.gz')
niiGT_hdr = niiGT.header
niiGT_img = niiGT.get_fdata()

niiGT_dim = niiGT_hdr.get_data_shape()

nx = niiGT_dim[0]
ny = niiGT_dim[1]
nz = niiGT_dim[2]

print("[OK]\n")

### load mask of interest
#########################
print("-> opening binary mask of interest...", end=' ')

niiMASK = nib.load('mask.nii')
niiMASK_hdr = niiMASK.header
niiMASK_img = niiMASK.get_fdata()

niiMASK_dim = niiMASK_hdr.get_data_shape()

niiMASK_idx = niiMASK_img == 1

print("[OK]\n")

### load reconstructions
########################
SUBMISSIONs = glob.glob("reconstructions/*.nii.gz")

# prepare EXCEL output
XLS = xlwt.Workbook()
XLS_sheet = XLS.add_sheet("Local measures")

XLS_sheet.write(0, 0, "filename")
XLS_sheet.write(0, 1, "Pd, mean")
XLS_sheet.write(0, 2, "Pd, std")
XLS_sheet.write(0, 3, "Pd, min")
XLS_sheet.write(0, 4, "Pd, 25 perc")
XLS_sheet.write(0, 5, "Pd, 50 perc")
XLS_sheet.write(0, 6, "Pd, 75 perc")
XLS_sheet.write(0, 7, "Pd, max")
XLS_sheet.write(0, 8, "n-, mean")
XLS_sheet.write(0, 9, "n-, std")
XLS_sheet.write(0, 10, "n-, min")
XLS_sheet.write(0, 11, "n-, 25 perc")
XLS_sheet.write(0, 12, "n-, 50 perc")
XLS_sheet.write(0, 13, "n-, 75 perc")
XLS_sheet.write(0, 14, "n-, max")
XLS_sheet.write(0, 15, "n+, mean")
XLS_sheet.write(0, 16, "n+, std")
XLS_sheet.write(0, 17, "n+, min")
XLS_sheet.write(0, 18, "n+, 25 perc")
XLS_sheet.write(0, 19, "n+, 50 perc")
XLS_sheet.write(0, 20, "n+, 75 perc")
XLS_sheet.write(0, 21, "n+, max")
XLS_sheet.write(0, 22, "AE, mean")
XLS_sheet.write(0, 23, "AE, std")
XLS_sheet.write(0, 24, "AE, min")
XLS_sheet.write(0, 25, "AE, 25 perc")
XLS_sheet.write(0, 26, "AE, 50 perc")
XLS_sheet.write(0, 27, "AE, 75 perc")
XLS_sheet.write(0, 28, "AE, max")

XLS_row = 1
for filename in SUBMISSIONs:
    print("-> %s" % os.path.basename(filename))

    print("\t* opening data...", end=' ')

    niiRECON = nib.load(filename)

    niiRECON_hdr = niiRECON.header
    niiRECON_img = niiRECON.get_fdata()

    niiRECON_dim = niiRECON_hdr.get_data_shape()

    print("[OK]")

    ### check consistency
    print("\t* checking consistency...", end=' ')

    if len(niiRECON_dim) != len(niiGT_dim):
        raise Exception("The shape of GROUND-TRUTH and RECONSTRUCTION do not match")
    if niiRECON_dim != niiGT_dim:
        raise Exception("'dim' of GROUND-TRUTH and RECONSTRUCTION do not match")

    print("[OK]")

    ### compute local metrics
    print("\t* computing local metrics...", end=' ')
    sys.stdout.flush()

    Pd = np.zeros(niiGT_dim[0:3])
    nP = np.zeros(niiGT_dim[0:3])
    nM = np.zeros(niiGT_dim[0:3])
    AE = np.zeros(niiGT_dim[0:3])

    for z in range(0, nz):
        for y in range(0, ny):
            for x in range(0, nx):
                if niiMASK_img[x, y, z] == 0:
                    continue

                # NUMBER OF FIBER POPULATIONS
                #############################

                DIR_true = np.zeros((3, 5))
                DIR_est = np.zeros((3, 5))

                # compute M_true, DIR_true, M_est, DIR_est
                M_true = 0
                for d in range(5):
                    dir = niiGT_img[x, y, z, range(d * 3, d * 3 + 3)]
                    f = norm(dir)
                    if f > 0:
                        DIR_true[:, M_true] = dir / f
                        M_true += 1
                if M_true == 0:
                    niiMASK_img[x, y, z] = 0  # do not consider this voxel in the final score
                    continue  # no fiber compartments found in the voxel

                M_est = 0
                for d in range(5):
                    dir = niiRECON_img[x, y, z, range(d * 3, d * 3 + 3)]
                    f = norm(dir)
                    if f > 0:
                        DIR_est[:, M_est] = dir / f
                        M_est += 1

                # compute Pd, nM and nP
                M_diff = M_true - M_est
                Pd[x, y, z] = 100 * abs(M_diff) / M_true
                if M_diff > 0:
                    nM[x, y, z] = M_diff
                else:
                    nP[x, y, z] = -M_diff

                # ANGULAR ACCURACY
                ##################

                # precompute matrix with angular errors among all estimated and true fibers
                A = np.zeros((M_true, M_est))
                for i in range(0, M_true):
                    for j in range(0, M_est):
                        err = acos(
                            min(1.0, abs(np.dot(DIR_true[:, i], DIR_est[:, j]))))  # crop to 1 for internal precision
                        A[i, j] = min(err, pi - err) / pi * 180

                # compute the "base" error
                M = min(M_true, M_est)
                err = np.zeros(M)
                notUsed_true = np.array(range(0, M_true))
                notUsed_est = np.array(range(0, M_est))
                AA = np.copy(A)
                for i in range(0, M):
                    err[i] = np.min(AA)
                    r, c = np.nonzero(AA == err[i])
                    AA[r[0], :] = float('Inf')
                    AA[:, c[0]] = float('Inf')
                    notUsed_true = notUsed_true[notUsed_true != r[0]]
                    notUsed_est = notUsed_est[notUsed_est != c[0]]

                # account for OVER-ESTIMATES
                if M_true < M_est:
                    if M_true > 0:
                        for i in notUsed_est:
                            err = np.append(err, min(A[:, i]))
                    else:
                        err = np.append(err, 45)
                # account for UNDER-ESTIMATES
                elif M_true > M_est:
                    if M_est > 0:
                        for i in notUsed_true:
                            err = np.append(err, min(A[i, :]))
                    else:
                        err = np.append(err, 45)

                AE[x, y, z] = np.mean(err)

    print("[OK]")

    # write to EXCEL file
    XLS_sheet.write(XLS_row, 0, filename)

    values = Pd[niiMASK_idx]
    XLS_sheet.write(XLS_row, 1, np.mean(values))
    XLS_sheet.write(XLS_row, 2, np.std(values))
    XLS_sheet.write(XLS_row, 3, np.min(values))
    XLS_sheet.write(XLS_row, 4, stats.scoreatpercentile(values, 25))
    XLS_sheet.write(XLS_row, 5, np.median(values))
    XLS_sheet.write(XLS_row, 6, stats.scoreatpercentile(values, 75))
    XLS_sheet.write(XLS_row, 7, np.max(values))

    values = nM[niiMASK_idx]
    XLS_sheet.write(XLS_row, 8, np.mean(values))
    XLS_sheet.write(XLS_row, 9, np.std(values))
    XLS_sheet.write(XLS_row, 10, np.min(values))
    XLS_sheet.write(XLS_row, 11, stats.scoreatpercentile(values, 25))
    XLS_sheet.write(XLS_row, 12, np.median(values))
    XLS_sheet.write(XLS_row, 13, stats.scoreatpercentile(values, 75))
    XLS_sheet.write(XLS_row, 14, np.max(values))

    values = nP[niiMASK_idx]
    XLS_sheet.write(XLS_row, 15, np.mean(values))
    XLS_sheet.write(XLS_row, 16, np.std(values))
    XLS_sheet.write(XLS_row, 17, np.min(values))
    XLS_sheet.write(XLS_row, 18, stats.scoreatpercentile(values, 25))
    XLS_sheet.write(XLS_row, 19, np.median(values))
    XLS_sheet.write(XLS_row, 20, stats.scoreatpercentile(values, 75))
    XLS_sheet.write(XLS_row, 21, np.max(values))

    values = AE[niiMASK_idx]
    XLS_sheet.write(XLS_row, 22, np.mean(values))
    XLS_sheet.write(XLS_row, 23, np.std(values))
    XLS_sheet.write(XLS_row, 24, np.min(values))
    XLS_sheet.write(XLS_row, 25, stats.scoreatpercentile(values, 25))
    XLS_sheet.write(XLS_row, 26, np.median(values))
    XLS_sheet.write(XLS_row, 27, stats.scoreatpercentile(values, 75))
    XLS_sheet.write(XLS_row, 28, np.max(values))

    XLS_row += 1

XLS.save("local_masures.xls")
