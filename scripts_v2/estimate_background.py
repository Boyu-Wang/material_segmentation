"""
Estimate background image by taking the median of original image:
    given a set of images, take the median of original raw images.

By: Boyu Wang (boywang@cs.stonybrook.edu)
Created Data: 10 May 2020
Last Modified Date: 17 May 2020
"""

import numpy as np
from PIL import Image
import cv2
import argparse
import os
import scipy
from scipy.spatial.distance import cdist
# from skimage.filters.rank import entropy
from scipy.stats import entropy
from skimage.morphology import disk
from skimage import io, color
from multiprocessing import Pool
from joblib import Parallel, delayed
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils
import robustfit

parser = argparse.ArgumentParser(description='flake segmentation')
parser.add_argument('--exp_sid', default=0, type=int, help='exp start id')
parser.add_argument('--exp_eid', default=1, type=int, help='exp end id')
parser.add_argument('--subexp_sid', default=0, type=int, help='subexp start id')
parser.add_argument('--subexp_eid', default=1, type=int, help='subexp end id')
parser.add_argument('--n_jobs', default=40, type=int, help='multiprocessing cores')


args = parser.parse_args()


def read_one_image(img_name):
    image = Image.open(img_name)
    im_rgb = np.array(image).astype('float')
    # im_gray = np.array(image.convert('L', (0.2989, 0.5870, 0.1140, 0))).astype('float')
    # imH, imW = im_gray.shape
    # # to have same result as matlab
    # im_hsv = color.rgb2hsv(im_rgb)
    # im_hsv[:,:,2] = im_hsv[:,:,2]/255.0
    # im_ghs = np.concatenate([np.expand_dims(im_gray,2), im_hsv[:,:,:2]], axis=2)

    return im_rgb


# estimate background images in one experiment
# the data folder should be organized like this:
# data
#   data_jan2019
#       Exp2
#           HighPressure
#               tile_x001_y001.tif        
#           LowPressure
#               tile_x001_y001.tif        
#           MediumPressure              
#               tile_x001_y001.tif        
def main():
    data_path = '../data/data_111x_individual/'
    result_path = '../results/data_111x_individual_result/background_image'

    exp_names = os.listdir(data_path)
    exp_names = [ename for ename in exp_names if ename[0]  not in ['.', '_']]
    exp_names.sort()

    for d in range(args.exp_sid, args.exp_eid):
        exp_name = exp_names[d]
        subexp_names = os.listdir(os.path.join(data_path, exp_name))
        subexp_names = [sname for sname in subexp_names if sname[0] not in ['.', '_']]
        subexp_names = [sname for sname in subexp_names if os.path.isdir(os.path.join(data_path, exp_name, sname))]
        subexp_names.sort()
        
        # process each subexp
        for s_d in range(args.subexp_sid, min(len(subexp_names), args.subexp_eid)):
            sname = subexp_names[s_d]
            print('processing images under this directory: %s'%(os.path.join(data_path, exp_name, sname)))
            img_names = os.listdir(os.path.join(data_path, exp_name, sname))
            img_names = [n_i for n_i in img_names if n_i[0] not in ['.', '_']]
            img_names.sort()
            try:    
                os.makedirs(os.path.join(result_path, exp_name, sname))
            except:
                pass
            all_rawrgb = Parallel(n_jobs=args.n_jobs)(delayed(read_one_image)(os.path.join(data_path, exp_name, sname, img_names[i])) for i in range(len(img_names)))
            median_rawrgb = np.median(np.stack(all_rawrgb, 0), 0)
            
            median_rawrgb = median_rawrgb.astype(np.uint8)

            cv2.imwrite( os.path.join(result_path, exp_name, sname, 'bg_median_raw.png'), np.flip(median_rawrgb.astype(np.uint8), 2))

if __name__ == '__main__':
    main()
    