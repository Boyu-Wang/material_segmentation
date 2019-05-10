import numpy as np
from PIL import Image
import cv2
import argparse
import os
import scipy
from skimage import io, color
import pickle

import utils
import robustfit


image = Image.open('../data/data_jan2019/YoungJaeShinSamples/5/tile_x004_y012.tif')

im_rgb = np.array(image).astype('float')
im_gray = np.array(image.convert('L', (0.2989, 0.5870, 0.1140, 0))).astype('float')
imH, imW = im_gray.shape
im_hsv = color.rgb2hsv(im_rgb)
im_hsv[:,:,2] = im_hsv[:,:,2]/255.0

res_map, image_labelmap, flake_centroids, flake_sizes, num_flakes = utils.perform_robustfit_multichannel(im_hsv, im_gray, 10, 0)