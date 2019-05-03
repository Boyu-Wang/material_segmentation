"""
Given a image and presegmented flake, draw a strike to label the junk area and good area

By: Boyu Wang (boywang@cs.stonybrook.edu)
Created Data: 25 Mar 2019
Last Modified Date: 28 Mar 2019
"""


import numpy as np
from PIL import Image
import cv2
import argparse
import os
# import scipy
from scipy.spatial.distance import cdist
# from skimage.filters.rank import entropy
from scipy.stats import entropy
from skimage.morphology import disk
# from multiprocessing import Pool
# from joblib import Parallel, delayed
import pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import itertools
import sklearn
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, AffinityPropagation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from mpl_toolkits.mplot3d import Axes3D
import gc
import copy
# from multiprocessing import Pool

parser = argparse.ArgumentParser(description='flake segmentation')
parser.add_argument('--exp_sid', default=5, type=int, help='exp start id')
parser.add_argument('--exp_eid', default=6, type=int, help='exp end id')
parser.add_argument('--subexp_sid', default=0, type=int, help='subexp start id')
parser.add_argument('--subexp_eid', default=1, type=int, help='subexp end id')
parser.add_argument('--img_sid', default=0, type=int)
parser.add_argument('--img_eid', default=294, type=int)
# parser.add_argument('--n_jobs', default=8, type=int, help='multiprocessing cores')
# parser.add_argument('--c_sid', default=0, type=int, help='subexp start id')
# parser.add_argument('--c_eid', default=400, type=int, help='subexp end id')

args = parser.parse_args()


# load the detected flake and get features for the flake
def load_one_image(img_name, info_name, fig_name, flake_size_thre=784):
    flake_info = pickle.load(open(info_name, 'rb'))
    image = Image.open(img_name)
    
    # bw_im = Image.open(fig_name)
    # bw_im = np.array(bw_im).astype('uint8')

    im_gray = np.array(image.convert('L', (0.2989, 0.5870, 0.1140, 0))).astype('float')
    imH, imW = im_gray.shape
    # im_hsv = np.array(image.convert('HSV')).astype('float')
    im_rgb = np.array(image).astype('uint8')
    # build a list of flakes
    num_flakes = len(flake_info['flakes'])
    image_labelmap = flake_info['image_labelmap']
    new_image_labelmap = np.zeros(image_labelmap.shape)

    assert num_flakes == image_labelmap.max()
    flakes = flake_info['flakes']
    large_flake_idxs = []
    bw_im = im_rgb.astype(np.uint8)
    
    cnt = 0
    for i in range(num_flakes):
        flake_size = flakes[i]['flake_size']
        if flake_size > flake_size_thre:
            large_flake_idxs.append(i)
            # flake_large_bbox = flakes[i]['flake_large_bbox']
            f_mask_r_min, f_mask_r_max, f_mask_c_min, f_mask_c_max = flakes[i]['flake_exact_bbox']
            f_mask_height = f_mask_r_max - f_mask_r_min
            f_mask_width = f_mask_c_max - f_mask_c_min
            flake_large_bbox = [max(0, f_mask_r_min - int(0.5 * f_mask_height)),
                                min(imH, f_mask_r_max + int(0.5 * f_mask_height)),
                                max(0, f_mask_c_min - int(0.5 * f_mask_width)),
                                min(imW, f_mask_c_max + int(0.5 * f_mask_width))]

            flakes[i]['flake_large_bbox'] = flake_large_bbox
            # flakes[i]['flake_img'] = im_rgb[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3], :].astype(np.uint8)
            flake_contours = flakes[i]['flake_contour_loc']
            flake_contours = np.flip(flake_contours, 1)
            flake_contours = np.expand_dims(flake_contours, 1)
            flake_contours = flake_contours.astype(int)
            # print(flake_contours.shape)
            bw_im = cv2.drawContours(bw_im, flake_contours, -1, (255,0,0), 2)
            cnt += 1
            new_image_labelmap[image_labelmap==(i+1)] = cnt

    flakes = [flakes[j] for j in large_flake_idxs]
    # b,g,r
    bw_im = np.flip(bw_im, 2)

    return im_rgb, bw_im, new_image_labelmap, flakes

def filter_region(image_labelmap, input_map, im_rgb, flakes, rmv=True):
    num_flakes = int(image_labelmap.max())
    new_image_labelmap = np.zeros(image_labelmap.shape)
    new_bw_im = im_rgb.astype(np.uint8)
    cnt = 0

    flake_idxs = []
    for i in range(num_flakes):
        flake_map = image_labelmap == (i+1)
        # remove junk
        if rmv:
            flag = not np.any(np.logical_and(flake_map, input_map))
        # only keep good
        else:
            flag = np.any(np.logical_and(flake_map, input_map))

        if flag:
            flake_idxs.append(i)
            flake_contours = flakes[i]['flake_contour_loc']
            flake_contours = np.flip(flake_contours, 1)
            flake_contours = np.expand_dims(flake_contours, 1)
            flake_contours = flake_contours.astype(int)
            new_bw_im = cv2.drawContours(new_bw_im, flake_contours, -1, (255,0,0), 2)
            cnt += 1
            new_image_labelmap[image_labelmap==(i+1)] = cnt

    flakes = [flakes[j] for j in flake_idxs]
    # b,g,r
    new_bw_im = np.flip(new_bw_im, 2)

    to_save = dict()
    # to_save['res_map'] = res_map
    to_save['image_labelmap'] = new_image_labelmap
    to_save['flakes'] = flakes

    return new_bw_im, to_save


def label_one_img(img_name, subexp_dir, rslt_dir, fig_dir, labelfig_save_path, labelmat_save_path):

    ori_img, bw_img, image_labelmap, flakes = load_one_image(os.path.join(subexp_dir, img_name), os.path.join(rslt_dir, img_name[:-4]+'.p'), os.path.join(fig_dir, img_name[:-4]+'.png'))

    img_dispaly = copy.copy(bw_img)
    ldrawing = True
    rdrawing = True
    def interactive_drawing(event,x,y,flags,params):
        global lix,liy,ldrawing,rix,riy,rdrawing
        if event==cv2.EVENT_LBUTTONDOWN:
            # print("Left click")
            ldrawing=True
            lix,liy=x,y

        if event==cv2.EVENT_RBUTTONDOWN:
            # print("Right click")
            rdrawing=True
            rix,riy=x,y

        elif event==cv2.EVENT_MOUSEMOVE:
            if ldrawing==True:
                if lix and liy:
                    cv2.line(img_dispaly,(lix,liy),(x,y),(255,0,0),2)
                    lix, liy = x, y
            if rdrawing==True:
                if rix and riy:
                    cv2.line(img_dispaly,(rix,riy),(x,y),(0,255,0),2)
                    rix, riy = x, y
            
        elif event==cv2.EVENT_LBUTTONUP:
            ldrawing=False
            cv2.line(img_dispaly,(lix,liy),(x,y),(255,0,0),2)

        elif event==cv2.EVENT_RBUTTONUP:
            rdrawing=False
            cv2.line(img_dispaly,(rix,riy),(x,y),(0,255,0),2)

    # cv2.namedWindow("Labeling", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Labeling")
    cv2.setMouseCallback("Labeling", interactive_drawing)

    while(1):
        cv2.imshow('Labeling', img_dispaly)
        k=cv2.waitKey(1)&0xFF
        if k==27:
            break
    
    # junk
    junk_map = img_dispaly[:,:,1] == 255
    good_map = img_dispaly[:,:,0] == 255
    # # remove junk
    # nojunk_bw_im, _ = filter_region(image_labelmap, junk_map, ori_img, flakes, rmv=True)
    # while(1):
    #     cv2.imshow('Remove Junk', nojunk_bw_im)
    #     k=cv2.waitKey(1)&0xFF
    #     if k==27:
    #         break

    # keep good
    good_bw_im, to_save = filter_region(image_labelmap, good_map, ori_img, flakes, rmv=False)
    while(1):
        cv2.imshow('Keep Good', good_bw_im)
        k=cv2.waitKey(1)&0xFF
        if k==27:
            break

    
    pickle.dump(to_save, open(os.path.join(labelmat_save_path, img_name[:-4] + '.p'), 'wb'))
    cv2.imwrite(os.path.join(labelfig_save_path, img_name[:-4] + '.png'), good_bw_im)
    cv2.destroyAllWindows()


def main():
    data_path = '../../data/data_jan2019'
    result_path = '../../results/data_jan2019_script/mat'
    fig_path = '../../results/data_jan2019_script/fig'
    # cluster_path = '../results/data_jan2019_script/cluster'
    # cluster_path = '../results/data_jan2019_script/cluster_sort'
    labelfig_path = '../../results/data_jan2019_script/labelfig_784'
    labelmat_path = '../../results/data_jan2019_script/labelmat_784'

    exp_names = os.listdir(data_path)
    exp_names.sort()
    exp_names = [ename for ename in exp_names if ename[0] != '.']

    # print(exp_names)
    # exp_names = exp_names[args.exp_sid: args.exp_eid]

    for d in range(args.exp_sid, args.exp_eid):
        exp_name = exp_names[d]
        subexp_names = os.listdir(os.path.join(data_path, exp_name))
        subexp_names = [sname for sname in subexp_names if os.path.isdir(os.path.join(data_path, exp_name, sname))]
        subexp_names.sort()
        # print(subexp_names)

        # process each subexp
        for s_d in range(args.subexp_sid, min(len(subexp_names), args.subexp_eid)):
            sname = subexp_names[s_d]
            img_names = os.listdir(os.path.join(data_path, exp_name, sname))
            img_names.sort()
            img_names = img_names[args.img_sid:args.img_eid]
            # flake_save_path = os.path.join(cluster_path, exp_name+sname)
            labelfig_save_path = os.path.join(labelfig_path, exp_name, sname)
            labelmat_save_path = os.path.join(labelmat_path, exp_name, sname)
            
            if not os.path.exists(labelfig_save_path):
                os.makedirs(labelfig_save_path)
            if not os.path.exists(labelmat_save_path):
                os.makedirs(labelmat_save_path)

            for img_name in img_names:
                label_one_img(img_name, os.path.join(data_path, exp_name, sname), os.path.join(result_path, exp_name, sname), os.path.join(fig_path, exp_name, sname), labelfig_save_path, labelmat_save_path)



if __name__ == '__main__':
    main()






