import argparse
import os

import multi_network_sscsd
import std_mcsd
import dipy_peak_extraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str, required=True)
    parser.add_argument('--bvals', '-va', type=str, required=True)
    parser.add_argument('--bvecs', '-ve', type=str, required=True)
    parser.add_argument('--mask', '-m', type=str, required=True)
    parser.add_argument('--order', '-o', type=int, default=8)
    parser.add_argument('--save_to', '-s', type=str, required=True)
    parser.add_argument('--std_method', '-std', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    path = args.path
    bvals = args.bvals
    bvecs = args.bvecs
    mask_path = args.mask
    order = args.order
    save_to = args.save_to
    is_std_method = args.std_method

    snrs = [10, 20, 30]
    for snr in snrs:
        snr_path = path + '/SNR_' + str(snr)
        for i in range(10):
            data_path = snr_path + '/data_' + str(i + 1) + '.nii'
            if is_std_method:
                std_mcsd.main(data_path, bvals, bvecs, mask_path, order, save_to)
            else:
                multi_network_sscsd.main(data_path, bvals, bvecs, mask_path, order, False, save_to)
            dipy_peak_extraction.peak_extraction(
                save_to + '/odfs.nii.gz',
                'sphere724.txt',
                save_to + '/peaks_' + str(i + 1) + '_SNR_' + str(snr) + '.nii.gz',
                relative_peak_threshold=0.2,
                min_separation_angle=15.0,
                max_peak_number=3
            )
            os.remove(save_to + '/odfs.nii.gz')
        os.remove(save_to + '/weights.pth')


if __name__ == '__main__':
    main()
