"""
Add more features to the saved pickles
By: Boyu Wang (boywang@cs.stonybrook.edu)
Created Data: 21 Feb 2020
Last Modified Date: 22 Feb 2020
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

import utils
import robustfit

parser = argparse.ArgumentParser(description='flake segmentation')
parser.add_argument('--exp_sid', default=0, type=int, help='exp start id')
parser.add_argument('--exp_eid', default=5, type=int, help='exp end id')
parser.add_argument('--subexp_sid', default=0, type=int, help='subexp start id')
parser.add_argument('--subexp_eid', default=15, type=int, help='subexp end id')
parser.add_argument('--n_jobs', default=40, type=int, help='multiprocessing cores')


args = parser.parse_args()

hyperparams = { 'im_thre': 10, # given the residulal map, about this threshould is identified as foreground.
                'size_thre': 100, # after detect foreground regions, filter them based on its size. (0 means all of them are kept)
                # 'n_clusters': 3, # number of cluster to segment each flake
                'n_clusters': 3, # number of cluster to segment each flake
                'loc_cluster': 0,
              }


# process one image
def process_one_image(img_name, save_name, fig_save_name, im_i):
    if not os.path.exists(save_name + '.p'):
        return
    print('process %s' %(img_name))

    image = Image.open(img_name)
    im_rgb = np.array(image).astype('float')
    im_gray = np.array(image.convert('L', (0.2989, 0.5870, 0.1140, 0))).astype('float')
    imH, imW = im_gray.shape
    # im_hsv = np.array(image.convert('HSV')).astype('float')
    # to have same result as matlab
    im_hsv = color.rgb2hsv(im_rgb)
    im_hsv[:,:,2] = im_hsv[:,:,2]/255.0
    try:
        flakes = pickle.load(open(save_name+'.p', 'rb'))
    except:
        print('error', save_name)

    image_labelmap = flakes['image_labelmap']
    res_map = flakes['res_map']
    bg_rgb = flakes['bg_rgb']

    flakes = flakes['flakes']
    num_flakes = len(flakes)

    # # get bg
    # bg_rgb = []
    # [C, R] = np.meshgrid(np.arange(0, imW), np.arange(0, imH))
    # Y = np.reshape(R, [-1]) / (imH - 1) - 0.5
    # X = np.reshape(C, [-1]) / (imW - 1) - 0.5
    # A = np.stack([np.ones([imH*imW]), X*X, Y*Y, X*Y, X, Y, ], axis=1)
    # for c in range(3):
    #     brob_rlm_model = robustfit.RLM()
    #     brob_rlm_model.fit(A, np.reshape(im_rgb[:,:,c], [-1]))
    #     pred_map = np.reshape(brob_rlm_model.predict(A), [imH, imW])
    #     bg_rgb.append(pred_map)
    # bg_rgb = np.stack(bg_rgb, axis=2).astype('float')
    bg_hsv = color.rgb2hsv(bg_rgb)
    bg_hsv[:,:,2] = bg_hsv[:,:,2]/255.0
    bg_gray = color.rgb2gray(bg_rgb)
    # get contrast color features
    contrast_gray = im_gray - bg_gray
    contrast_hsv = im_hsv - bg_hsv
    contrast_rgb = im_rgb - bg_rgb

    kernel = np.ones((5,5),np.uint8)
    im_bgr_tosave = np.flip(im_rgb, 2)
    im_bgr_tosave = im_bgr_tosave.astype(np.uint8)

    # get features for each flake
    for i in range(num_flakes):
        bwmap = (image_labelmap == i + 1).astype(np.uint8)
        # inner_bwmap = cv2.erode(bwmap, kernel, iterations=1)
        # # add inner contrast features
        # flake_inner_contrast_color_fea = [0] * 16
        # flake_inner_contrast_color_entropy = 0
        # if inner_bwmap.sum() > 0:
        #     flake_inner_contrast_color_entropy = cv2.calcHist([contrast_gray[inner_bwmap>0].astype('uint8')],[0],None,[256],[0,256])
        #     flake_inner_contrast_color_entropy = entropy(flake_inner_contrast_color_entropy, base=2)
        #     flake_inner_contrast_color_fea = [contrast_gray[inner_bwmap>0].mean(), 
        #                  contrast_hsv[inner_bwmap>0, 2].mean()] + \
        #                  [contrast_gray[inner_bwmap>0].std()] + \
        #                  list(contrast_hsv[inner_bwmap>0].mean(0)) + list(contrast_hsv[inner_bwmap>0].std(0)) + \
        #                  list(contrast_rgb[inner_bwmap>0].mean(0)) + list(contrast_rgb[inner_bwmap>0].std(0)) + list(flake_inner_contrast_color_entropy)
        #     # print(flake_inner_contrast_color_fea)

        flake_i = flakes[i]
        # flake_i['flake_innercontrast_color_fea'] = np.array(flake_inner_contrast_color_fea)

        # _, flake_contoursx, _ = cv2.findContours(bwmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # print(flake_contoursx[0], flake_contoursx[0].shape)
        # cv2.drawContours(im_bgr_tosave, flake_contoursx[0], -1, (0,0,255), 2)

        flake_contours = flake_i['flake_contour_loc']
        flake_contours = np.flip(flake_contours, 1)
        flake_contours = np.expand_dims(flake_contours, 1)
        flake_contours = flake_contours.astype(np.int32)
        # print('new', flake_contours.shape)
        im_bgr_tosave = cv2.drawContours(im_bgr_tosave, [flake_contours], -1, (0,0,255), 2)
        # print('done')
        # cluster each flake 
        if flake_i['flake_size'] > hyperparams['size_thre']:
            n_clusters = hyperparams['n_clusters']
            flake_rgb = im_rgb[bwmap>0]
            flake_gray = im_gray[bwmap>0]
            flake_hsv = im_hsv[bwmap>0]

            flake_contrast_rgb = im_rgb[bwmap>0] - bg_rgb[bwmap>0]
            flake_contrast_gray = im_gray[bwmap>0] - bg_gray[bwmap>0]
            flake_contrast_hsv = im_hsv[bwmap>0] - bg_hsv[bwmap>0]
            
            loc_cluster = hyperparams['loc_cluster']
            if loc_cluster>0:
                flake_x, flake_y = np.nonzero(bwmap>0)
                pixel_features = np.concatenate([flake_rgb, flake_hsv, np.expand_dims(flake_gray, 1), np.expand_dims(flake_x, 1), np.expand_dims(flake_y, 1)], 1)
            else:
                pixel_features = np.concatenate([flake_rgb, flake_hsv, np.expand_dims(flake_gray, 1)], 1)
            pixel_features = StandardScaler().fit_transform(pixel_features)
            n_pixel = pixel_features.shape[0]
            cluster_rslt = KMeans(n_clusters=n_clusters, random_state=0, n_jobs=-1).fit(pixel_features)
            assignment = cluster_rslt.labels_
            # get the overlayed image
            overlay = np.zeros([n_pixel, 3], dtype=np.uint8)
            overlay[assignment==0] = (255,0,0)
            overlay[assignment==1] = (0,255,0)
            overlay[assignment==2] = (0,0,255)
            ori_bgr = im_bgr_tosave[bwmap>0]
            overlay_bgr = cv2.addWeighted(np.expand_dims(ori_bgr,0), 0.75, np.expand_dims(overlay,0), 0.25, 0)
            im_bgr_tosave[bwmap>0] = overlay_bgr[0,:,:]
            all_subsegment_features = []
            all_subsegment_keys = []
            # print(sum(assignment==0))
            # print(sum(assignment==1))
            # print(sum(assignment==2))
            for ci in range(n_clusters):
                subseg_gray = flake_gray[assignment==ci]
                subseg_contrast_gray = flake_contrast_gray[assignment==ci]
                # print(len(subseg_gray))
                subseg_hsv = flake_hsv[assignment==ci]
                subseg_rgb = flake_rgb[assignment==ci]
                subseg_contrast_hsv = flake_contrast_hsv[assignment==ci]
                subseg_contrast_rgb = flake_contrast_rgb[assignment==ci]

                sub_flake_color_entropy = cv2.calcHist([subseg_gray.astype('uint8')],[0],None,[256],[0,256])
                sub_flake_color_entropy = entropy(sub_flake_color_entropy, base=2)[0]
                sub_flake_contrast_color_entropy = cv2.calcHist([subseg_contrast_gray.astype('uint8')],[0],None,[256],[0,256])
                sub_flake_contrast_color_entropy = entropy(sub_flake_contrast_color_entropy, base=2)[0]

                sub_flake_color_fea = [subseg_gray.mean(), 
                         subseg_hsv[:, 2].mean()] + \
                         [subseg_hsv.std()] + \
                         list(subseg_hsv.mean(0)) + list(subseg_hsv.std(0)) + \
                         list(subseg_rgb.mean(0)) + list(subseg_rgb.std(0)) + [sub_flake_color_entropy] + \
                         [subseg_contrast_gray.mean(), 
                         subseg_contrast_hsv[:, 2].mean()] + \
                         [subseg_contrast_gray.std()] + \
                         list(subseg_contrast_hsv.mean(0)) + list(subseg_contrast_hsv.std(0)) + \
                         list(subseg_contrast_rgb.mean(0)) + list(subseg_contrast_rgb.std(0)) + [sub_flake_contrast_color_entropy]
                # all_subsegment_features[sub_flake_color_fea[0]] = sub_flake_color_fea
                all_subsegment_features.append(sub_flake_color_fea)
                all_subsegment_keys.append(sub_flake_color_fea[0])
                # print(sub_flake_color_fea[0])
            # sort based on gray values
            subsegment_features = []
            # for key in sorted(all_subsegment_features.keys()):
            #     subsegment_features.extend(all_subsegment_features[key])
            key_ids = np.argsort(all_subsegment_keys)
            for key_id in key_ids:
                # print(key_id, all_subsegment_keys[key_id])
                subsegment_features.extend(all_subsegment_features[key_id])


            subsegment_features = np.array(subsegment_features)
            if subsegment_features.shape[0] != 32 * n_clusters:
                print('wrong', save_name, i, subsegment_features.shape)
            assert subsegment_features.shape[0] == 32 * n_clusters
            # print(subsegment_features.shape)
            # print(subsegment_features)
            # print(subsegment_features.shape)
            # flake_i['subsegment_features'] = subsegment_features
            # flake_i['subsegment_features_2'] = subsegment_features
            flake_i['subsegment_features_%d_loc_%d'%(n_clusters, loc_cluster)] = subsegment_features
            flake_i['subsegment_assignment_%d_loc_%d'%(n_clusters, loc_cluster)] = assignment
            # flake_i['subsegment_features_4'] = subsegment_features
            # flake_i['subsegment_features_3'] = subsegment_features

            # return 
    # if subsegment_features is not None:
    #     print(subsegment_features.shape)

    # save mat and images
    to_save = dict()
    to_save['bg_rgb'] = bg_rgb
    to_save['res_map'] = res_map
    to_save['image_labelmap'] = image_labelmap
    to_save['flakes'] = flakes

    pickle.dump(to_save, open(save_name + '.p', 'wb'))
    # cv2.imwrite(fig_save_name + '.png', im_bgr_tosave)

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
    data_path = '../data/data_111x_individual/'
    result_path = '../results/data_111x_individual_script/mat'
    result_fig_path = '../results/data_111x_individual_script/fig_subsegment'

    # data_path = '../data/data_mar2020/'
    # result_path = '../results/data_mar2020_script/mat'
    # result_fig_path = '../results/data_mar2020_script/fig_subsegment'
    
    exp_names = os.listdir(data_path)
    # exp_names = [ename for ename in exp_names if ename[0]  not in ['.', '_']]
    # exp_names = ['laminator', 'PDMS-QPress 6s']
    # exp_names = ['PDMS-QPress 6s']
    # exp_names = ['laminator']
    exp_names = ['PDMS-QPress 60s']
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
            
            if not os.path.exists(os.path.join(result_fig_path, exp_name, sname)):
                os.makedirs(os.path.join(result_fig_path, exp_name, sname))

            print('num of images: %d'%(len(img_names)))
            Parallel(n_jobs=args.n_jobs)(delayed(process_one_image)(os.path.join(data_path, exp_name, sname, img_names[i]),
                            os.path.join(result_path, exp_name, sname, img_names[i][:-4]), 
                            os.path.join(result_fig_path, exp_name, sname, img_names[i][:-4]), i) for i in range(len(img_names)))
            

            
def test():
    data_path = '../data/data_111x_individual/'
    result_path = '../results/data_111x_individual_script/mat'
    result_fig_path = '../results/data_111x_individual_script/fig_subsegment'

    img_names = ['laminator/home-pdms/home-pdmspdms_tile__ix004_iy019.tiff']
    for i in range(len(img_names)):
        process_one_image(os.path.join(data_path,img_names[i]),
                    os.path.join(result_path, img_names[i][:-4]), 
                    os.path.join(result_fig_path, img_names[i][:-4]), i)
            

if __name__ == '__main__':
    main()
    # test()






