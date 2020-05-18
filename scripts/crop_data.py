"""
Crop the tile scan into multiple patches. 
The size of each patch should be 450x450.
"""

from PIL import Image
from joblib import Parallel, delayed
import os
import numpy as np

def crop_patch(tile_scan_path, tile_scan_name, patch_path, PATCH_SIZE, STRIDE_SIZE, center=False):
    im = Image.open(os.path.join(tile_scan_path, tile_scan_name))
    name_prefix = tile_scan_name.split('.')[0]
    # patch_path = os.path.join(patch_path, name_prefix)
    if not os.path.exists(patch_path):
        try:
            os.makedirs(patch_path)
        except:
            pass
    width, height = im.size
    Hs = np.arange(0, height-PATCH_SIZE+1, STRIDE_SIZE)
    Ws = np.arange(0, width-PATCH_SIZE+1, STRIDE_SIZE)
    if center:
        # only crop center
        num_h = len(Hs)
        num_w = len(Ws)
        Hs = Hs[num_h//2-1: num_h//2+1]
        Ws = Ws[num_w//2-1: num_w//2+1]

    ww, hh = np.meshgrid(Ws, Hs)
    im = np.asarray(im)
    # print(im.shape)
    for x in range(len(Ws)):
        for y in range(len(Hs)):
            patch_im = im[Hs[y]:Hs[y]+PATCH_SIZE, Ws[x]:Ws[x]+PATCH_SIZE, :]
            patch_im = Image.fromarray(patch_im)
            patch_im.save(os.path.join(patch_path, '{:s}_x-{:d}_y-{:d}.tiff'.format(name_prefix, Ws[x], Hs[y])))



def main():
    # tile_scan_path = '../data/data_1015/tile_scan'
    # PATCH_SIZE = 450
    # STRIDE_SIZE = 450
    # patch_path = '../data/data_1015/patch_{:d}_{:d}'.format(PATCH_SIZE, STRIDE_SIZE)
    
    # tile_scan_names = os.listdir(tile_scan_path)
    # tile_scan_names = [ename for ename in tile_scan_names if ename[0] not in ['.', '_']]
    # tile_scan_names.sort()

    # Parallel(n_jobs=20)(delayed(crop_patch)(tile_scan_path, tile_scan_name, patch_path, PATCH_SIZE, STRIDE_SIZE) 
    #                     for tile_scan_name in tile_scan_names)

    tile_scan_path = '../data/10222019G wtih Suji/tile_scan/'
    PATCH_SIZE = 500
    STRIDE_SIZE = 500
    patch_path = '../data/10222019G wtih Suji/patch_{:d}_{:d}'.format(PATCH_SIZE, STRIDE_SIZE)
    
    exp_dirs = os.listdir(tile_scan_path)
    exp_dirs = [ename for ename in exp_dirs if ename[0] not in ['.', '_']]
    exp_dirs.sort()
    for exp_dir in exp_dirs:
        tmp_scan_path = os.path.join(tile_scan_path, exp_dir)
        tmp_patch_path = os.path.join(patch_path, exp_dir)

        tile_scan_names = os.listdir(tmp_scan_path)
        tile_scan_names = [ename for ename in tile_scan_names if ename[0] not in ['.', '_']]
        tile_scan_names.sort()

        Parallel(n_jobs=20)(delayed(crop_patch)(tmp_scan_path, tile_scan_name, tmp_patch_path, PATCH_SIZE, STRIDE_SIZE, False) 
                        for tile_scan_name in tile_scan_names)
   


if __name__ == '__main__':
    main()
