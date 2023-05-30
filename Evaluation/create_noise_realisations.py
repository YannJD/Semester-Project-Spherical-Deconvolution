import argparse
import os

from dipy.io.image import load_nifti
from dipy.sims.phantom import add_noise
import nibabel as nib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str, required=True)
    parser.add_argument('--volumes', '-v', type=str, required=True)
    args = parser.parse_args()

    fpaths = args.path
    data_path = args.volumes

    data, affine = load_nifti(data_path)

    snrs = [10, 20, 30]
    for snr in snrs:
        os.mkdir(fpaths + '/SNR_' + str(snr))
        for i in range(10):
            noisy_data = add_noise(data, snr)

            data_img = nib.Nifti1Image(noisy_data, None)
            nib.save(data_img, fpaths + '/SNR_' + str(snr) + '/data_' + str(i + 1))


if __name__ == '__main__':
    main()
