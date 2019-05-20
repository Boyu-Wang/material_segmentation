"""
Flake segmentation using robustfit. 
Given a image, the algorithm tries to fit a function as background. 
Everything doesn't fit the background function well are identified as outlier (flake/glue).

By: Boyu Wang (boywang@cs.stonybrook.edu)
Created Data: 21 Feb 2019
Last Modified Date: 8 Mar 2019
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

import utils

parser = argparse.ArgumentParser(description='flake segmentation')
parser.add_argument('--exp_sid', default=0, type=int, help='exp start id')
parser.add_argument('--exp_eid', default=1, type=int, help='exp end id')
parser.add_argument('--subexp_sid', default=0, type=int, help='subexp start id')
parser.add_argument('--subexp_eid', default=3, type=int, help='subexp end id')
parser.add_argument('--n_jobs', default=30, type=int, help='multiprocessing cores')

args = parser.parse_args()

hyperparams = { 'im_thre': 10, # given the residulal map, about this threshould is identified as foreground.
                'size_thre': 0, # after detect foreground regions, filter them based on its size. (0 means all of them are kept)
                }

# process one image
def process_one_image(img_name, bk_name, bk_flake_centroids, save_name, fig_save_name, im_i):
    if os.path.exists(save_name + '.p') and os.path.exists(fig_save_name + '.png'):
        return
    print('process %s' %(img_name))
    bk_image = Image.open(bk_name)
    bk_rgb = np.array(bk_image).astype('float')
    bk_gray = np.array(bk_image.convert('L', (0.2989, 0.5870, 0.1140, 0))).astype('float')
    # bk_hsv = np.array(bk_image.convert('HSV')).astype('float')
    # to have same result as matlab
    bk_hsv = color.rgb2hsv(bk_rgb)
    bk_hsv[:,:,2] = bk_hsv[:,:,2]/255.0

    image = Image.open(img_name)
    im_rgb = np.array(image).astype('float')
    im_gray = np.array(image.convert('L', (0.2989, 0.5870, 0.1140, 0))).astype('float')
    imH, imW = im_gray.shape
    # im_hsv = np.array(image.convert('HSV')).astype('float')
    # to have same result as matlab
    im_hsv = color.rgb2hsv(im_rgb)
    im_hsv[:,:,2] = im_hsv[:,:,2]/255.0

    res_map, image_labelmap, flake_centroids, flake_sizes, num_flakes = utils.perform_robustfit_multichannel(im_hsv, im_gray, hyperparams['im_thre'], hyperparams['size_thre'])
    # remove regions in bk
    dis = cdist(flake_centroids, bk_flake_centroids)
    dis_min = dis.min(1)
    # to_rmv = np.nonzero(dis_min < 5)[0]
    to_keep = np.nonzero(dis_min >= 5)[0]

    flake_sizes = flake_sizes[to_keep]
    flake_centroids = flake_centroids[to_keep]
    num_flakes = to_keep.shape[0]

    # process label map
    new_image_labels = np.zeros(image_labelmap.shape)
    cnt = 0
    for idx in to_keep:
        cnt += 1
        new_image_labels[image_labelmap==idx+1] = cnt
    image_labelmap = new_image_labels

    flakes = []

    kernel = np.ones((5,5),np.uint8)

    im_tosave = im_rgb.astype(np.uint8)

    # get features for each flake
    for i in range(num_flakes):
        f_mask_r, f_mask_c = np.nonzero(image_labelmap==i+1)
        f_mask_r_min = min(f_mask_r)
        f_mask_r_max = max(f_mask_r)
        f_mask_height = f_mask_r_max - f_mask_r_min
        f_mask_c_min = min(f_mask_c)
        f_mask_c_max = max(f_mask_c)
        f_mask_width = f_mask_c_max - f_mask_c_min
        flake_exact_bbox = [f_mask_r_min, f_mask_r_max+1, f_mask_c_min, f_mask_c_max+1]
        flake_large_bbox = [max(0, f_mask_r_min - int(0.1*f_mask_height)), min(imH, f_mask_r_max + int(0.1*f_mask_height)),
                                    max(0, f_mask_c_min - int(0.1*f_mask_width)), min(imW, f_mask_c_max + int(0.1*f_mask_width))]

        bwmap = (image_labelmap == i + 1).astype(np.uint8)
        _, flake_contours, _ = cv2.findContours(bwmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # print(im_tosave)
        # print(flake_contours[0])
        # print(flake_contours[0].shape)
        im_tosave = cv2.drawContours(im_tosave, flake_contours[0], -1, (255,0,0), 2)

        flake_contours = np.squeeze(flake_contours[0], 1)
        # row, column
        flake_contours = np.flip(flake_contours, 1)

        
        # compute convex hull of the contours
        flake_convexhull = cv2.convexHull(flake_contours)

        # shape fea
        flake_shape_len_area_ratio = flake_contours.shape[0] / (bwmap.sum() + 0.0)
        contours_center_dis = cdist(np.expand_dims(flake_centroids[i],0), flake_contours)
        flake_shape_contour_hist = np.histogram(contours_center_dis, bins=15)[0]
        flake_shape_contour_hist = flake_shape_contour_hist / flake_shape_contour_hist.sum()
        flake_shape_fracdim = utils.fractal_dimension(bwmap)
        
        inner_bwmap = cv2.erode(bwmap, kernel, iterations=1)

        # color fea
        flake_color_fea = [im_gray[bwmap>0].mean() - bk_gray[bwmap>0].mean(), 
                         im_hsv[bwmap>0, 2].mean() - bk_hsv[bwmap>0, 2].mean()] + \
                         [im_gray[bwmap>0].mean(), im_gray[bwmap>0].std()] + \
                         list(im_hsv[bwmap>0].mean(0)) + list(im_hsv[bwmap>0].std(0)) + \
                         list(im_rgb[bwmap>0].mean(0)) + list(im_rgb[bwmap>0].std(0))
        # flake_color_entropy = entropy(im_gray[bwmap>0].astype('uint8'), disk(5))
        flake_color_entropy = cv2.calcHist([im_gray[bwmap>0].astype('uint8')],[0],None,[256],[0,256])
        flake_color_entropy = entropy(flake_color_entropy, base=2)
        flake_inner_color_fea = [0] * 16
        flake_inner_color_entropy = 0
        if inner_bwmap.sum() > 0:
            flake_inner_color_fea = [im_gray[inner_bwmap>0].mean() - bk_gray[inner_bwmap>0].mean(), 
                         im_hsv[inner_bwmap>0, 2].mean() - bk_hsv[inner_bwmap>0, 2].mean()] + \
                         [im_gray[inner_bwmap>0].mean(), im_gray[inner_bwmap>0].std()] + \
                         list(im_hsv[inner_bwmap>0].mean(0)) + list(im_hsv[inner_bwmap>0].std(0)) + \
                         list(im_rgb[inner_bwmap>0].mean(0)) + list(im_rgb[inner_bwmap>0].std(0))

            # flake_inner_color_entropy = entropy(im_gray[inner_bwmap>0].astype('uint8'), disk(5))
            flake_inner_color_entropy = cv2.calcHist([im_gray[inner_bwmap>0].astype('uint8')],[0],None,[256],[0,256])
            flake_inner_color_entropy = entropy(flake_inner_color_entropy, base=2)

        flake_i = dict()
        flake_i['img_name'] = img_name
        flake_i['flake_id'] = i+1
        flake_i['flake_size'] = flake_sizes[i]
        flake_i['flake_exact_bbox'] = flake_exact_bbox
        flake_i['flake_large_bbox'] = flake_large_bbox
        flake_i['flake_contour_loc'] = flake_contours.astype('int16')
        flake_i['flake_convexcontour_loc'] = flake_convexhull.astype('int16')
        flake_i['flake_center'] = flake_centroids[i]
        # flake_i['flake_img'] = im_rgb[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3], :].astype(np.uint8)
        flake_i['flake_shape_fea'] = np.array([flake_shape_len_area_ratio] + list(flake_shape_contour_hist) + [flake_shape_fracdim])
        flake_i['flake_color_fea'] = np.array(flake_color_fea + [flake_color_entropy] + flake_inner_color_fea + [flake_inner_color_entropy])

        flakes.append(flake_i)


    # save mat and images
    to_save = dict()
    to_save['res_map'] = res_map
    to_save['image_labelmap'] = image_labelmap
    to_save['flakes'] = flakes

    pickle.dump(to_save, open(save_name + '.p', 'wb'))
    cv2.imwrite(fig_save_name + '.png', np.flip(im_tosave, 2))


# process images in one experiment in parallel
# the data folder should be organized like this:
# data
#   data_jan2019
#       Exp2
#           EXPT2_FieldRef_50X.tif
#           HighPressure
#               tile_x001_y001.tif        
#           LowPressure
#               tile_x001_y001.tif        
#           MediumPressure              
#               tile_x001_y001.tif        
def main():
    data_path = '../data/data_jan2019'
    result_path = '../results/data_jan2019_script/mat'
    result_fig_path = '../results/data_jan2019_script/fig'

    exp_names = os.listdir(data_path)
    exp_names = [ename for ename in exp_names if ename[0]  not in ['.', '_']]
    exp_names.sort()
    # print(exp_names)
    # exp_names = exp_names[args.exp_sid: args.exp_eid]

    for d in range(args.exp_sid, args.exp_eid):
        exp_name = exp_names[d]
        subexp_names = os.listdir(os.path.join(data_path, exp_name))
        subexp_names = [sname for sname in subexp_names if sname[0] not in ['.', '_']]
        # find bk image, and subexp folders
        bk_name = [sname for sname in subexp_names if 'FieldRef_50X.tif' in sname]
        if len(bk_name)>0 :
            bk_name = os.path.join(data_path, exp_name, bk_name[0])
        else:
            bk_name = os.path.join(data_path, exp_name, 'depth1time1N1shear0velocity5/tile_x001_y007.tif')
        subexp_names = [sname for sname in subexp_names if os.path.isdir(os.path.join(data_path, exp_name, sname))]
        subexp_names.sort()
        # print(subexp_names)

        # read bk image and identify regions in bk image
        bk_image = Image.open(bk_name)
        bk_gray = np.array(bk_image.convert('L', (0.2989, 0.5870, 0.1140, 0))).astype('float')
        _, _, bk_flake_centroids, _, _ = utils.perform_robustfit(bk_gray, 10, 0)
        # process each subexp
        for s_d in range(args.subexp_sid, min(len(subexp_names), args.subexp_eid)):
            sname = subexp_names[s_d]
            print('processing images under this directory: %s'%(os.path.join(data_path, exp_name, sname)))
            img_names = os.listdir(os.path.join(data_path, exp_name, sname))
            img_names = [n_i for n_i in img_names if n_i[0] not in ['.', '_']]
            img_names.sort()
            if not os.path.exists(os.path.join(result_path, exp_name, sname)):
                os.makedirs(os.path.join(result_path, exp_name, sname))
            if not os.path.exists(os.path.join(result_fig_path, exp_name, sname)):
                os.makedirs(os.path.join(result_fig_path, exp_name, sname))

            print('num of images: %d'%(len(img_names)))
            Parallel(n_jobs=args.n_jobs)(delayed(process_one_image)(os.path.join(data_path, exp_name, sname, img_names[i]),
                            bk_name, bk_flake_centroids, 
                            os.path.join(result_path, exp_name, sname, img_names[i][:-4]), 
                            os.path.join(result_fig_path, exp_name, sname, img_names[i][:-4]), i) for i in range(len(img_names)))
            
            # for i in range(len(img_names)):
            #     process_one_image(os.path.join(data_path, exp_name, sname, img_names[i]),
            #                 bk_name, bk_flake_centroids, 
            #                 os.path.join(result_path, exp_name, sname, img_names[i][:-4]), 
            #                 os.path.join(result_fig_path, exp_name, sname, img_names[i][:-4]), i)



if __name__ == '__main__':
    main()






