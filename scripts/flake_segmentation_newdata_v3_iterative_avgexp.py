"""
Flake segmentation on new data. for unknow bg, first fit the bg image, and extract features.
Flake segmentation using robustfit. 
Given a image, the algorithm tries to fit a function as background. 
Everything doesn't fit the background function well are identified as outlier (flake/glue).

By: Boyu Wang (boywang@cs.stonybrook.edu)
Created Data: 21 Feb 2019
Last Modified Date: 3 Nov 2019
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
parser.add_argument('--subexp_eid', default=15, type=int, help='subexp end id')
parser.add_argument('--n_jobs', default=40, type=int, help='multiprocessing cores')


args = parser.parse_args()

hyperparams = { 'im_thre_high': [3,4,4], # given the residulal map, about this threshould is identified as foreground.
                'im_thre_low': [3,2,3], # given the residulal map, about this threshould is identified as foreground.
                'ori_im_thre_pairs': [(3,3), (4,2), (4,3), (4,4)],
                'avg_im_thre_pairs': [(2,0), (3,0), (3,1.5), (4,2)],
                'size_thre': 30, # after detect foreground regions, filter them based on its size. (0 means all of them are kept).
                # 'size_thre': 100, # after detect foreground regions, filter them based on its size. (0 means all of them are kept).
                'n_clusters': 3, # number of cluster to segment each flake.
                }

# process one image
def process_one_image_step1(img_name, save_name, fig_save_name):
    image = Image.open(img_name)
    im_rgb = np.array(image).astype('float')
    im_gray = np.array(image.convert('L', (0.2989, 0.5870, 0.1140, 0))).astype('float')
    imH, imW = im_gray.shape
    # im_hsv = np.array(image.convert('HSV')).astype('float')
    # to have same result as matlab
    im_hsv = color.rgb2hsv(im_rgb)
    im_hsv[:,:,2] = im_hsv[:,:,2]/255.0
    im_ghs = np.concatenate([np.expand_dims(im_gray,2), im_hsv[:,:,:2]], axis=2)
    
    if os.path.exists(save_name + '.p'):
        rslt = pickle.load(open(save_name+'.p', 'rb'))
        bg_ghs = rslt['bg_ghs']
        bg_rgb = rslt['bg_rgb']
        # return bg_ghs, bg_rgb
        return bg_ghs, bg_rgb, im_ghs, im_rgb
    print('process %s' %(img_name))
    
    # res_map, image_labelmap, flake_centroids, flake_sizes, num_flakes = utils.perform_robustfit_multichannel(im_hsv, im_gray, hyperparams['im_thre'], hyperparams['size_thre'])
    all_res_map, all_image_labelmap, all_flake_centroids, all_flake_sizes, all_num_flakes, bg_ghs = utils.perform_robustfit_multichannel_v4(im_hsv, im_gray, hyperparams['ori_im_thre_pairs'], hyperparams['size_thre'])

    # if num_flakes != len(flake_sizes):
    #     print(len(flake_sizes), num_flakes, img_name)
    #     return

    # get bg
    bg_rgb = []
    [C, R] = np.meshgrid(np.arange(0, imW), np.arange(0, imH))
    Y = np.reshape(R, [-1]) / (imH - 1) - 0.5
    X = np.reshape(C, [-1]) / (imW - 1) - 0.5
    A = np.stack([np.ones([imH*imW]), X*X, Y*Y, X*Y, X, Y, ], axis=1)
    for c in range(3):
        brob_rlm_model = robustfit.RLM()
        brob_rlm_model.fit(A, np.reshape(im_rgb[:,:,c], [-1]))
        pred_map = np.reshape(brob_rlm_model.predict(A), [imH, imW])
        bg_rgb.append(pred_map)
    bg_rgb = np.stack(bg_rgb, axis=2).astype('float')
    bg_hsv = color.rgb2hsv(bg_rgb)
    bg_hsv[:,:,2] = bg_hsv[:,:,2]/255.0
    bg_gray = color.rgb2gray(bg_rgb)
    # # get contrast color features
    # contrast_gray = im_gray - bg_gray
    # contrast_hsv = im_hsv - bg_hsv
    # contrast_rgb = im_rgb - bg_rgb

    # flakes = []

    # kernel = np.ones((5,5),np.uint8)

    cv2.imwrite(fig_save_name + '_ori_bg.png', np.flip(bg_rgb.astype(np.uint8), 2))

    all_im_tosave = []
    for pi, thre_pair in enumerate(hyperparams['ori_im_thre_pairs']):
        im_tosave = im_rgb.astype(np.uint8)
    
        # get features for each flake
        for i in range(all_num_flakes[pi]):
            bwmap = (all_image_labelmap[pi] == i + 1).astype(np.uint8)
            _, flake_contours, _ = cv2.findContours(bwmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            im_tosave = cv2.drawContours(im_tosave, flake_contours[0], -1, (255,0,0), 2)
        all_im_tosave.append(im_tosave)

        cv2.imwrite(fig_save_name + '_ori_thre-' + str(thre_pair[0]) + '-' + str(thre_pair[1]) + '.png', np.flip(im_tosave, 2))

    # save mat and images
    to_save = dict()
    to_save['bg_rgb'] = bg_rgb
    to_save['bg_ghs'] = bg_ghs
    to_save['all_res_map'] = all_res_map
    to_save['all_image_labelmap'] = all_image_labelmap
    to_save['all_im_tosave'] = all_im_tosave
    # to_save['flakes'] = flakes

    pickle.dump(to_save, open(save_name + '.p', 'wb'))
    
    return bg_ghs, bg_rgb, im_ghs, im_rgb


def process_one_image_step2(avg_bg_ghs, img_name, save_name, fig_save_name, bg_method_name='avg_est_bg'):
    if os.path.exists(save_name + '.p'):
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

    # res_map, image_labelmap, flake_centroids, flake_sizes, num_flakes = utils.perform_robustfit_multichannel(im_hsv, im_gray, hyperparams['im_thre'], hyperparams['size_thre'])
    all_res_map, all_image_labelmap, all_flake_centroids, all_flake_sizes, all_num_flakes = utils.perform_robustfit_multichannel_v5(avg_bg_ghs, im_hsv, im_gray, hyperparams['avg_im_thre_pairs'], hyperparams['size_thre'])
    
    # print(all_num_flakes)
    all_im_tosave = []
    for pi, thre_pair in enumerate(hyperparams['avg_im_thre_pairs']):
        im_tosave = im_rgb.astype(np.uint8)
    
        # get features for each flake
        for i in range(all_num_flakes[pi]):
            bwmap = (all_image_labelmap[pi] == i + 1).astype(np.uint8)
            _, flake_contours, _ = cv2.findContours(bwmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            im_tosave = cv2.drawContours(im_tosave, flake_contours[0], -1, (255,0,0), 2)
        all_im_tosave.append(im_tosave)

        cv2.imwrite(fig_save_name + '_' + bg_method_name+ '_thre-' + str(thre_pair[0]) + '-' + str(thre_pair[1]) + '.png', np.flip(im_tosave, 2))

    # save mat and images
    to_save = dict()
    to_save['all_res_map'] = all_res_map
    to_save['all_image_labelmap'] = all_image_labelmap
    to_save['all_im_tosave'] = all_im_tosave

    pickle.dump(to_save, open(save_name + '.p', 'wb'))


def plot_all(step1_p_name, step2_p_name, avg_bg_rgb, fig_save_name):
    step1_rslt = pickle.load(open(step1_p_name+'.p', 'rb'))
    step2_rslt = pickle.load(open(step2_p_name+'.p', 'rb'))

    bg_rgb = step1_rslt['bg_rgb']

    fig = plt.figure()
    ax = fig.add_subplot(2,5,1)
    ax.imshow(bg_rgb.astype(np.uint8))
    ax.set_title('ori bg rgb')
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    

    for pi, thre_pair in enumerate(hyperparams['ori_im_thre_pairs']):
        ax = fig.add_subplot(2, 5, 2+pi)
        ax.imshow(step1_rslt['all_im_tosave'][pi].astype(np.uint8))
        ax.set_title('ori: ' + str(thre_pair[0]) + ', ' + str(thre_pair[1]))
        ax.set_xticks([], [])
        ax.set_yticks([], [])
    

    ax = fig.add_subplot(2,5,6)
    ax.imshow(avg_bg_rgb.astype(np.uint8))
    ax.set_title('avg bg rgb')
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    
    for pi, thre_pair in enumerate(hyperparams['avg_im_thre_pairs']):
        ax = fig.add_subplot(2, 5, 5+2+pi)
        ax.imshow(step2_rslt['all_im_tosave'][pi].astype(np.uint8))
        ax.set_title('avg: ' + str(thre_pair[0]) + ', ' + str(thre_pair[1]))
        ax.set_xticks([], [])
        ax.set_yticks([], [])
    

    plt.savefig(fig_save_name+'.png')
    plt.close()


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

    data_path = '../data/data_mar2020/'
    ori_result_path = '../results/data_mar2020_script/mat_ori'
    ori_avg_result_fig_path = '../results/data_mar2020_script/fig_ori_avg'
    avg_estbg_result_path = '../results/data_mar2020_script/mat_avg_estbg'
    median_estbg_result_path = '../results/data_mar2020_script/mat_median_estbg'
    median_raw_result_path = '../results/data_mar2020_script/mat_median_raw'
    avg_raw_result_path = '../results/data_mar2020_script/mat_avg_raw'
    final_result_fig_path = '../results/data_mar2020_script/final_vis'

    # data_path = '../data/data_111x_individual/'
    # ori_result_path = '../results/data_111x_individual_script/ori_mat'
    # # ori_result_fig_path = '../results/data_111x_individual_script/ori_fig'
    # ori_avg_result_fig_path = '../results/data_111x_individual_script/ori_avg_fig'
    # avg_result_path = '../results/data_111x_individual_script/avg_mat'
    # # avg_result_fig_path = '../results/data_111x_individual_script/avg_fig'
    # final_result_fig_path = '../results/data_111x_individual_script/final_vis'

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
            # img_names = [n_i for n_i in img_names if n_i[0] not in ['.', '_']]
            # img_names.sort()
            try:
                os.makedirs(os.path.join(ori_result_path, exp_name, sname))
            except:
                pass
            try:    
                os.makedirs(os.path.join(ori_avg_result_fig_path, exp_name, sname))
            except:
                pass
            try:    
                os.makedirs(os.path.join(avg_estbg_result_path, exp_name, sname))
            except:
                pass
            try:    
                os.makedirs(os.path.join(median_estbg_result_path, exp_name, sname))
            except:
                pass
            try:    
                os.makedirs(os.path.join(median_raw_result_path, exp_name, sname))
            except:
                pass
            try:    
                os.makedirs(os.path.join(avg_raw_result_path, exp_name, sname))
            except:
                pass
            # try:    
            #     os.makedirs(os.path.join(avg_result_fig_path, exp_name, sname))
            # except:
            #     pass
            try:    
                os.makedirs(os.path.join(final_result_fig_path, exp_name, sname))
            except:
                pass
            # if not os.path.exists(os.path.join(ori_result_path, exp_name, sname)):
            #     os.makedirs(os.path.join(ori_result_path, exp_name, sname))
            # if not os.path.exists(os.path.join(ori_result_fig_path, exp_name, sname)):
            #     os.makedirs(os.path.join(ori_result_fig_path, exp_name, sname))


            print('num of images: %d'%(len(img_names)))
            all_bgghs_bgrgb_rawghs_rawrgb = Parallel(n_jobs=args.n_jobs)(delayed(process_one_image_step1)(os.path.join(data_path, exp_name, sname, img_names[i]),
                            os.path.join(ori_result_path, exp_name, sname, img_names[i][:-4]), 
                            os.path.join(ori_avg_result_fig_path, exp_name, sname, img_names[i][:-4])) for i in range(len(img_names)))
            all_bgghs = [bgghs_bgrgb_rawghs_rawrgb[0] for bgghs_bgrgb_rawghs_rawrgb in all_bgghs_bgrgb_rawghs_rawrgb]
            avg_bgghs = np.mean(np.stack(all_bgghs, 0), 0)
            median_bgghs = np.median(np.stack(all_bgghs, 0), 0)
            
            all_bgrgb = [bgghs_bgrgb_rawghs_rawrgb[1] for bgghs_bgrgb_rawghs_rawrgb in all_bgghs_bgrgb_rawghs_rawrgb]
            avg_bgrgb = np.mean(np.stack(all_bgrgb, 0), 0)
            median_bgrgb = np.median(np.stack(all_bgrgb, 0), 0)

            all_rawghs = [bgghs_bgrgb_rawghs_rawrgb[2] for bgghs_bgrgb_rawghs_rawrgb in all_bgghs_bgrgb_rawghs_rawrgb]
            avg_rawghs = np.mean(np.stack(all_rawghs, 0), 0)
            median_rawghs = np.median(np.stack(all_rawghs, 0), 0)
            
            all_rawrgb = [bgghs_bgrgb_rawghs_rawrgb[3] for bgghs_bgrgb_rawghs_rawrgb in all_bgghs_bgrgb_rawghs_rawrgb]
            avg_rawrgb = np.mean(np.stack(all_rawrgb, 0), 0)
            median_rawrgb = np.median(np.stack(all_rawrgb, 0), 0)
            

            cv2.imwrite( os.path.join(ori_avg_result_fig_path, exp_name, sname, 'bg_avg_estbg.png'), np.flip(avg_bgrgb.astype(np.uint8), 2))
            cv2.imwrite( os.path.join(ori_avg_result_fig_path, exp_name, sname, 'bg_median_estbg.png'), np.flip(median_bgrgb.astype(np.uint8), 2))
            cv2.imwrite( os.path.join(ori_avg_result_fig_path, exp_name, sname, 'bg_avg_raw.png'), np.flip(avg_rawrgb.astype(np.uint8), 2))
            cv2.imwrite( os.path.join(ori_avg_result_fig_path, exp_name, sname, 'bg_median_raw.png'), np.flip(median_rawrgb.astype(np.uint8), 2))

            fig = plt.figure()
            ax = fig.add_subplot(2,2,1)
            ax.imshow(avg_bgrgb.astype(np.uint8), cmap='jet')
            ax.set_title('avg estimate bg')
            ax.set_xticks([], [])
            ax.set_yticks([], [])
            ax = fig.add_subplot(2,2,2)
            ax.imshow(median_bgrgb.astype(np.uint8), cmap='jet')
            ax.set_title('median estimate bg')
            ax.set_xticks([], [])
            ax.set_yticks([], [])
            ax = fig.add_subplot(2,2,3)
            ax.imshow(avg_rawrgb.astype(np.uint8), cmap='jet')
            ax.set_title('avg raw image')
            ax.set_xticks([], [])
            ax.set_yticks([], [])
            ax = fig.add_subplot(2,2,4)
            ax.imshow(median_rawrgb.astype(np.uint8), cmap='jet')
            ax.set_title('median raw image')
            ax.set_xticks([], [])
            ax.set_yticks([], [])
            plt.savefig(os.path.join(ori_avg_result_fig_path, exp_name, sname, 'bg_comparison.png'))
            plt.close()



            Parallel(n_jobs=args.n_jobs)(delayed(process_one_image_step2)(avg_bgghs, os.path.join(data_path, exp_name, sname, img_names[i]),
                            os.path.join(avg_estbg_result_path, exp_name, sname, img_names[i][:-4]), 
                            os.path.join(ori_avg_result_fig_path, exp_name, sname, img_names[i][:-4]), bg_method_name='avg_estbg') for i in range(len(img_names)))

            Parallel(n_jobs=args.n_jobs)(delayed(process_one_image_step2)(median_bgghs, os.path.join(data_path, exp_name, sname, img_names[i]),
                            os.path.join(median_estbg_result_path, exp_name, sname, img_names[i][:-4]), 
                            os.path.join(ori_avg_result_fig_path, exp_name, sname, img_names[i][:-4]), bg_method_name='median_estbg') for i in range(len(img_names)))

            Parallel(n_jobs=args.n_jobs)(delayed(process_one_image_step2)(avg_rawghs, os.path.join(data_path, exp_name, sname, img_names[i]),
                            os.path.join(avg_raw_result_path, exp_name, sname, img_names[i][:-4]), 
                            os.path.join(ori_avg_result_fig_path, exp_name, sname, img_names[i][:-4]), bg_method_name='avg_raw') for i in range(len(img_names)))

            Parallel(n_jobs=args.n_jobs)(delayed(process_one_image_step2)(median_rawghs, os.path.join(data_path, exp_name, sname, img_names[i]),
                            os.path.join(median_raw_result_path, exp_name, sname, img_names[i][:-4]), 
                            os.path.join(ori_avg_result_fig_path, exp_name, sname, img_names[i][:-4]), bg_method_name='median_raw') for i in range(len(img_names)))


            # for i in range(len(img_names)):
            #     # process_one_image_step2(avg_ghs, os.path.join(data_path, exp_name, sname, img_names[i]), os.path.join(avg_result_path, exp_name, sname, img_names[i][:-4]), os.path.join(ori_avg_result_fig_path, exp_name, sname, img_names[i][:-4]) )
            #     plot_all(os.path.join(ori_result_path, exp_name, sname, img_names[i][:-4]), os.path.join(avg_result_path, exp_name, sname, img_names[i][:-4]), avg_rgb, os.path.join(final_result_fig_path, exp_name, sname, img_names[i][:-4]))



def debug():
    # img_name = '../data/data_mar2020/EXP/G03132020UK50X/G_tile__ix000_iy000.tif'
    # save_name = '../results/data_mar2020_script/mat/EXP/G03132020UK50X/G_tile__ix000_iy000_debug'
    # fig_save_name = '../results/data_mar2020_script/fig/EXP/G03132020UK50X/G_tile__ix000_iy000_debug'
    
    # img_name = '../data/data_mar2020/EXP/G03132020UK50X_2/G_tile__ix000_iy000.tif'
    img_name = '../data/data_mar2020/EXP/G03132020UK50X/G_tile__ix000_iy000.tif'
    save_name = '../results/data_mar2020_script/mat/' + img_name.split('/', 3)[-1] +  '_debug_iterative_%.2f_%.2f'%(hyperparams['im_thre_high'], hyperparams['im_thre_low'])
    fig_save_name = '../results/data_mar2020_script/fig/' + img_name.split('/', 3)[-1] +  '_debug_iterative_%.2f_%.2f'%(hyperparams['im_thre_high'], hyperparams['im_thre_low'])
    
    # print(fig_save_name)

    # img_name = '../data/data_111x_individual/PDMS-QPress 6s/1/6s-1_tile__ix006_iy005.tiff'
    # save_dir = os.path.join('../results/data_111x_individual_script/mat/', img_name.split('/')[3], img_name.split('/')[4])
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # save_dir = os.path.join('../results/data_111x_individual_script/fig/', img_name.split('/')[3], img_name.split('/')[4])
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # save_name = '../results/data_111x_individual_script/mat/' + img_name.split('/', 3)[-1] +  '_debug_iterative_%.2f_%.2f'%(hyperparams['im_thre_high'], hyperparams['im_thre_low'])
    # fig_save_name = '../results/data_111x_individual_script/fig/' + img_name.split('/', 3)[-1] +  '_debug_iterative_%.2f_%.2f'%(hyperparams['im_thre_high'], hyperparams['im_thre_low'])
    
    image = Image.open(img_name)
    im_rgb = np.array(image).astype('float')
    im_gray = np.array(image.convert('L', (0.2989, 0.5870, 0.1140, 0))).astype('float')
    imH, imW = im_gray.shape
    # im_hsv = np.array(image.convert('HSV')).astype('float')
    # to have same result as matlab
    im_hsv = color.rgb2hsv(im_rgb)
    im_hsv[:,:,2] = im_hsv[:,:,2]/255.0

    # res_map, image_labelmap, flake_centroids, flake_sizes, num_flakes = utils.perform_robustfit_multichannel_v2(im_hsv, im_gray, 3, 0)
    process_one_image(img_name, save_name, fig_save_name, 0)


if __name__ == '__main__':
    main()
    # debug()





