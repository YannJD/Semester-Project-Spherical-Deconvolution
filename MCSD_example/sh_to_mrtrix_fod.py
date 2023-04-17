import numpy as np
from dipy.data import get_sphere
from dipy.reconst import shm
import nibabel as nib

from main import convert_to_mrtrix


def main():
    with open('odf_sh.npy', 'rb') as f:
        odf_sh = np.load(f, allow_pickle=True)
    sh_const = .5 / np.sqrt(np.pi)
    vf = odf_sh[..., :3] / sh_const
    wm_vf = vf[..., -1]
    conversion_matrix = convert_to_mrtrix(8)
    wm_sh = odf_sh[:, :, :, 2:]
    sphere = get_sphere('symmetric724')
    new_odf = shm.sh_to_sf(np.dot(wm_sh, conversion_matrix.T) * wm_vf[..., np.newaxis], sphere, 8)
    fods_img = nib.Nifti1Image(new_odf, None)
    nib.save(fods_img, "mrtrix_odfs.nii.gz")


if __name__ == '__main__':
    main()