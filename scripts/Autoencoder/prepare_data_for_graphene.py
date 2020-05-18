"""
Crop flake and put it in the center of image for autoencoder. To classify graphene and non-graphene

By: Boyu Wang (boywang@cs.stonybrook.edu)
"""

import os
import numpy as np
import cv2
import pickle
import argparse
from PIL import Image
from multiprocessing import Pool
from joblib import Parallel, delayed
from skimage import io, color
import itertools
import random
import sys
sys.path.append('../')
import robustfit


parser = argparse.ArgumentParser(description='flake segmentation')
parser.add_argument('--exp_sid', default=0, type=int, help='exp start id')
parser.add_argument('--exp_eid', default=6, type=int, help='exp end id')
parser.add_argument('--subexp_sid', default=0, type=int, help='subexp start id')
parser.add_argument('--subexp_eid', default=10, type=int, help='subexp end id')
parser.add_argument('--n_jobs', default=15, type=int, help='multiprocessing cores')
# parser.add_argument('--c_sid', default=0, type=int, help='subexp start id')
# parser.add_argument('--c_eid', default=400, type=int, help='subexp end id')

args = parser.parse_args()
hyperparams = { 'size_thre': 784, # after detect foreground regions, filter them based on its size. (784=28*28 corresponds to 5 around um regions)
                }


def process_one_img(ins_name, img_dir, rslt_dir, ae_img_path, ae_mask_path, ae_contrast_path, ae_bg_path, output_img_size=256):
    ins_name = ins_name.split('.')[0]
    # load the detected flake and get features for the flake
    if not os.path.exists(os.path.join(rslt_dir, ins_name+'..p')):
        return
    flake_info = pickle.load(open(os.path.join(rslt_dir, ins_name+'..p'), 'rb'))
    image = Image.open(os.path.join(img_dir, ins_name+'.tiff'))
    # im_gray = np.array(image.convert('L', (0.2989, 0.5870, 0.1140, 0))).astype('float')
    # imH, imW = im_gray.shape
    # im_hsv = np.array(image.convert('HSV')).astype('float')
    # im_rgb = np.array(image).astype(np.uint8)

    im_rgb = np.array(image).astype('float')
    im_hsv = color.rgb2hsv(im_rgb)
    im_hsv[:,:,2] = im_hsv[:,:,2]/255.0
    im_gray = color.rgb2gray(im_rgb)
    imH, imW = im_gray.shape
    
    # build a list of flakes
    num_flakes = len(flake_info['flakes'])
    image_labelmap = flake_info['image_labelmap']
    assert num_flakes == image_labelmap.max()
    flakes = flake_info['flakes']
    large_flake_idxs = []
    cnt = 0

    # get bg image
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

    for i in range(num_flakes):
        flake_size = flakes[i]['flake_size']
        flake_centroids = flakes[i]['flake_center'].astype('int')
        if flake_size > hyperparams['size_thre']:
        # if flake_size > hyperparams['size_thre'] and flake_centroids[0] - output_img_size/2 >=0 and flake_centroids[0] + output_img_size/2 < imH and flake_centroids[1] - output_img_size/2 >=0 and flake_centroids[1] + output_img_size/2 < imW:
            large_flake_idxs.append(i)
            flake_large_bbox = [max(0, flake_centroids[0] - output_img_size//2),
                                min(imH, flake_centroids[0] + output_img_size // 2),
                                max(0, flake_centroids[1] - output_img_size // 2),
                                min(imW, flake_centroids[1] + output_img_size // 2)]

            f_img = np.zeros([output_img_size, output_img_size, 3], dtype=np.uint8)
            r_min = output_img_size//2 - (flake_centroids[0] - flake_large_bbox[0])
            r_max = r_min + (flake_large_bbox[1] - flake_large_bbox[0])
            c_min = output_img_size//2 - (flake_centroids[1] - flake_large_bbox[2])
            c_max = c_min + (flake_large_bbox[3] - flake_large_bbox[2])
            # print(r_min, r_max, c_min, c_max)
            f_img[r_min:r_max, c_min:c_max, :] = im_rgb[flake_large_bbox[0]: flake_large_bbox[1],
                                     flake_large_bbox[2]:flake_large_bbox[3], :]
            f_img = f_img.astype(np.uint8)

            f_bg_img = np.zeros([output_img_size, output_img_size, 3], dtype=np.uint8)
            f_bg_img[r_min:r_max, c_min:c_max, :] = bg_rgb[flake_large_bbox[0]: flake_large_bbox[1],
                                     flake_large_bbox[2]:flake_large_bbox[3], :]
            f_bg_img = f_bg_img.astype(np.uint8)

            # get mask
            mask = (image_labelmap == i + 1).astype(np.uint8)
            mask[mask==1] = 255
            f_mask = np.zeros([output_img_size, output_img_size], dtype=np.uint8)
            f_mask[r_min:r_max, c_min:c_max] = mask[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3]]
            f_contrast = np.zeros([output_img_size, output_img_size, 3], dtype=np.uint8)
            f_contrast[r_min:r_max, c_min:c_max, 0] = np.abs(im_gray[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3]] - \
                                                        bg_gray[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3]] )
            f_contrast[r_min:r_max, c_min:c_max, 1] = np.abs(im_hsv[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3], 0] - \
                                                        bg_hsv[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3], 0])
            f_contrast = f_contrast.astype(np.uint8)

            cv2.imwrite(os.path.join(ae_img_path, '%s-%d.png' % (ins_name, cnt)), np.flip(f_img, 2))
            cv2.imwrite(os.path.join(ae_mask_path, '%s-%d.png' % (ins_name, cnt)), f_mask)
            cv2.imwrite(os.path.join(ae_contrast_path, '%s-%d.png' % (ins_name, cnt)), f_contrast)
            cv2.imwrite(os.path.join(ae_bg_path, '%s-%d.png' % (ins_name, cnt)), np.flip(f_bg_img, 2))
            cnt += 1

# process all the flakes
def main():
    # data_path = '../../data/data_sep2019'
    # result_path = '../../results/data_sep2019_script/mat'
    # ae_img_path = '../../results/data_sep2019_script/ae_img/'
    # ae_mask_path = '../../results/data_sep2019_script/ae_mask/'
    # ae_contrast_path = '../../results/data_sep2019_script/ae_contrast/'
    # exp_names = os.listdir(data_path)
    # exp_names = [ename for ename in exp_names if ename[0] not in ['.', '_']]
    # exp_names.sort()

    # data_path = '../../data/10222019G wtih Suji'
    # result_path = '../../results/10222019G wtih Suji_script/center_patch_500_500/mat'
    # ae_img_path = '../../results/10222019G wtih Suji_script/ae_img/'
    # ae_mask_path = '../../results/10222019G wtih Suji_script/ae_mask/'
    # ae_contrast_path = '../../results/10222019G wtih Suji_script/ae_contrast/'
    # exp_names = ['center_patch_500_500']

    data_path = '../../data/data_111x_individual'
    result_path = '../../results/data_111x_individual_script/mat'
    ae_img_path = '../../results/data_111x_individual_script/ae_img/'
    ae_mask_path = '../../results/data_111x_individual_script/ae_mask/'
    ae_contrast_path = '../../results/data_111x_individual_script/ae_contrast/'
    ae_bg_path = '../../results/data_111x_individual_script/ae_bg/'
    # exp_names = os.listdir(data_path)
    # exp_names = [ename for ename in exp_names if ename[0] not in ['.', '_']]
    exp_names = ['laminator', 'PDMS-QPress 6s', 'PDMS-QPress 60s']
    exp_names.sort()

    if not os.path.exists(ae_img_path):
        os.makedirs(ae_img_path)

    if not os.path.exists(ae_mask_path):
        os.makedirs(ae_mask_path)

    if not os.path.exists(ae_contrast_path):
        os.makedirs(ae_contrast_path)

    if not os.path.exists(ae_bg_path):
        os.makedirs(ae_bg_path)

    for d in range(args.exp_sid, min(args.exp_eid, len(exp_names))):
        exp_name = exp_names[d]
        subexp_names = os.listdir(os.path.join(data_path, exp_name))
        subexp_names = [sname for sname in subexp_names if os.path.isdir(os.path.join(data_path, exp_name, sname))]
        subexp_names.sort()

        # process each subexp
        for s_d in range(args.subexp_sid, min(len(subexp_names), args.subexp_eid)):
            sname = subexp_names[s_d]

            img_dir_name = os.path.join(data_path, exp_name, sname)
            # for sep
            rslt_dir_name = os.path.join(result_path, exp_name, sname)
            # for oct
            # rslt_dir_name = os.path.join(result_path, sname)

            sub_ae_img_path = os.path.join(ae_img_path, exp_name, sname)
            sub_ae_mask_path = os.path.join(ae_mask_path, exp_name, sname)
            sub_ae_contrast_path = os.path.join(ae_contrast_path, exp_name, sname)
            sub_ae_bg_path = os.path.join(ae_bg_path, exp_name, sname)

            if not os.path.exists(sub_ae_img_path):
                os.makedirs(sub_ae_img_path)
            if not os.path.exists(sub_ae_mask_path):
                os.makedirs(sub_ae_mask_path)
            if not os.path.exists(sub_ae_contrast_path):
                os.makedirs(sub_ae_contrast_path)
            if not os.path.exists(sub_ae_bg_path):
                os.makedirs(sub_ae_bg_path)

            img_names = os.listdir(img_dir_name)
            img_names.sort()
            # for i_d in range(len(img_names)):
            #     process_one_img(img_names[i_d], img_dir_name, rslt_dir_name, sub_ae_img_path, sub_ae_mask_path, output_img_size=256)

            Parallel(n_jobs=args.n_jobs)(delayed(process_one_img)(img_names[i_d], img_dir_name, rslt_dir_name, sub_ae_img_path, sub_ae_mask_path, sub_ae_contrast_path, sub_ae_bg_path, output_img_size=256)
                                                      for i_d in range(len(img_names)))
            #


# load the detected flake and get features for the flake
def load_one_image(fname, flake_path, data_path, output_img_size=256):
    flake_info = pickle.load(open(os.path.join(flake_path, fname), 'rb'))
    img_name = fname.split('.')[0] + '.tiff'
    image = Image.open(os.path.join(data_path, img_name))

    im_rgb = np.array(image).astype('float')
    im_hsv = color.rgb2hsv(im_rgb)
    im_hsv[:,:,2] = im_hsv[:,:,2]/255.0
    im_gray = color.rgb2gray(im_rgb)
    imH, imW = im_gray.shape
    
    # get bg image
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

    # build a list of flakes
    num_flakes = len(flake_info['flakes'])
    image_labelmap = flake_info['image_labelmap']
    flakes = flake_info['flakes']
    large_flake_idxs = []
    for i in range(num_flakes):
        flake_size = flakes[i]['flake_size']
        if flake_size > hyperparams['size_thre']:
            large_flake_idxs.append(i)
            flake_centroids = flakes[i]['flake_center'].astype('int')
            flake_large_bbox = [max(0, flake_centroids[0] - output_img_size//2),
                                min(imH, flake_centroids[0] + output_img_size // 2),
                                max(0, flake_centroids[1] - output_img_size // 2),
                                min(imW, flake_centroids[1] + output_img_size // 2)]

            f_img = np.zeros([output_img_size, output_img_size, 3], dtype=np.uint8)
            r_min = output_img_size//2 - (flake_centroids[0] - flake_large_bbox[0])
            r_max = r_min + (flake_large_bbox[1] - flake_large_bbox[0])
            c_min = output_img_size//2 - (flake_centroids[1] - flake_large_bbox[2])
            c_max = c_min + (flake_large_bbox[3] - flake_large_bbox[2])
            # print(r_min, r_max, c_min, c_max)
            f_img[r_min:r_max, c_min:c_max, :] = im_rgb[flake_large_bbox[0]: flake_large_bbox[1],
                                     flake_large_bbox[2]:flake_large_bbox[3], :]
            f_img = f_img.astype(np.uint8)

            # get mask
            mask = (image_labelmap == flakes[i]['flake_id']).astype(np.uint8)
            mask[mask==1] = 255
            f_mask = np.zeros([output_img_size, output_img_size], dtype=np.uint8)
            f_mask[r_min:r_max, c_min:c_max] = mask[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3]]
            f_contrast = np.zeros([output_img_size, output_img_size, 3], dtype=np.uint8)
            f_contrast[r_min:r_max, c_min:c_max, 0] = np.abs(im_gray[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3]] - \
                                                            bg_gray[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3]])
            f_contrast[r_min:r_max, c_min:c_max, 1] = np.abs(im_hsv[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3], 0] - \
                                                            bg_hsv[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3], 0])
            f_contrast = f_contrast.astype(np.uint8)

            flakes[i]['flake_img'] = f_img
            flakes[i]['flake_mask'] = f_mask
            flakes[i]['flake_contrast'] = f_contrast

    flakes = [flakes[j] for j in large_flake_idxs]
    return flakes

# load graphene flakes, or non-graphene flakes.
def load_one_class(data_path, flake_path):
    flakes = []
    fnames = os.listdir(flake_path)
    flakes = Parallel(n_jobs=20)(delayed(load_one_image)(fname, flake_path, data_path)
                        for fname in fnames)
    flakes = list(itertools.chain.from_iterable(flakes))
    
    return flakes


# only process those with annotations
def main_label():
    # data_path = '../../data/data_sep2019/EXP1/09192019 Graphene'
    # graphene_path = '../../results/data_sep2019_script/labelmat_graphene/EXP1/09192019 Graphene'
    # thick_path = '../../results/data_sep2019_script/labelmat_thick/EXP1/09192019 Graphene'
    # glue_path = '../../results/data_sep2019_script/labelmat_glue/EXP1/09192019 Graphene'
    # output_path = '../../results/data_sep2019_script/ae_complete_label_img'
    # output_mask_path = '../../results/data_sep2019_script/ae_complete_label_mask'
    # output_contrast_path = '../../results/data_sep2019_script/ae_complete_label_contrast'
    # exp_names = os.listdir(data_path)
    # exp_names.sort()
    # feature_path = '../../results/data_sep2019_script/graphene_classifier_binary_fea-%s/feature.p'%'contrast-bg-shape'
    # feats = pickle.load(open(feature_path, 'rb'))
    # graphene_feats = feats['graphene_feats']
    # other_feats = feats['other_feats']

    
    data_path = '../../data/10222019G wtih Suji/center_patch_500_500'
    graphene_path = '../../results/10222019G wtih Suji_script/center_patch_500_500/labelmat_graphene'
    others_path = '../../results/10222019G wtih Suji_script/center_patch_500_500/labelmat_others'
    output_path = '../../results/10222019G wtih Suji_script/ae_complete_label_img'
    output_mask_path = '../../results/10222019G wtih Suji_script/ae_complete_label_mask'
    output_contrast_path = '../../results/10222019G wtih Suji_script/ae_complete_label_contrast'
    feature_path = '../../results/10222019G wtih Suji_script/center_patch_500_500/graphene_classifier_binary_fea-%s/feature.p'%'contrast-bg-shape'
    feats = pickle.load(open(feature_path, 'rb'))
    graphene_feats = feats['graphene_feats']
    other_feats = feats['other_feats']


    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(os.path.join(output_path, 'train'))
        os.makedirs(os.path.join(output_path, 'val'))

    if not os.path.exists(output_mask_path):
        os.makedirs(output_mask_path)
        os.makedirs(os.path.join(output_mask_path, 'train'))
        os.makedirs(os.path.join(output_mask_path, 'val'))

    if not os.path.exists(output_contrast_path):
        os.makedirs(output_contrast_path)
        os.makedirs(os.path.join(output_contrast_path, 'train'))
        os.makedirs(os.path.join(output_contrast_path, 'val'))

    # # for sep data
    # graphene_flakes = load_one_class(data_path, graphene_path)
    # thick_flakes = load_one_class(data_path, thick_path)
    # glue_flakes = load_one_class(data_path, glue_path)
    # other_flakes = list(thick_flakes) + list(glue_flakes)

    # for oct data
    exp_names = os.listdir(data_path)
    exp_names.sort()
    exp_names = [ename for ename in exp_names if ename[0] != '.']
    graphene_flakes = []
    other_flakes = []
    for exp_name in exp_names:
        graphene_flakes.extend(load_one_class(os.path.join(data_path, exp_name), os.path.join(graphene_path, exp_name)))
        other_flakes.extend(load_one_class(os.path.join(data_path, exp_name), os.path.join(others_path, exp_name)))

    assert graphene_feats.shape[0] == len(graphene_flakes)
    assert other_feats.shape[0] == len(other_flakes)

    num_graphene = len(graphene_flakes)
    num_others = len(other_flakes)

    train_ratio = 0.8
    graphene_idxs = np.arange(num_graphene)
    random.Random(123).shuffle(graphene_idxs)
    others_idxs = np.arange(num_others)
    random.Random(123).shuffle(others_idxs)
    
    tr_graphene_idxs = graphene_idxs[:int(train_ratio*num_graphene)]
    val_graphene_idxs = graphene_idxs[int(train_ratio*num_graphene):]
    
    tr_others_idxs = others_idxs[:int(train_ratio*num_others)]
    val_others_idxs = others_idxs[int(train_ratio*num_others):]

    for i in tr_graphene_idxs:
        cv2.imwrite(os.path.join(output_path, 'train', 'graphene_%d.png' % (i)), np.flip(graphene_flakes[i]['flake_img'], 2))
        cv2.imwrite(os.path.join(output_mask_path, 'train', 'graphene_%d.png' % (i)), graphene_flakes[i]['flake_mask'])
        cv2.imwrite(os.path.join(output_contrast_path, 'train', 'graphene_%d.png' % (i)), graphene_flakes[i]['flake_contrast'])

    for i in val_graphene_idxs:
        cv2.imwrite(os.path.join(output_path, 'val', 'graphene_%d.png' % (i)), np.flip(graphene_flakes[i]['flake_img'], 2))
        cv2.imwrite(os.path.join(output_mask_path, 'val', 'graphene_%d.png' % (i)), graphene_flakes[i]['flake_mask'])
        cv2.imwrite(os.path.join(output_contrast_path, 'val', 'graphene_%d.png' % (i)), graphene_flakes[i]['flake_contrast'])

    for i in tr_others_idxs:
        cv2.imwrite(os.path.join(output_path, 'train', 'others_%d.png' % (i)), np.flip(other_flakes[i]['flake_img'], 2))
        cv2.imwrite(os.path.join(output_mask_path, 'train', 'others_%d.png' % (i)), other_flakes[i]['flake_mask'])
        cv2.imwrite(os.path.join(output_contrast_path, 'train', 'others_%d.png' % (i)), other_flakes[i]['flake_contrast'])

    for i in val_others_idxs:
        cv2.imwrite(os.path.join(output_path, 'val', 'others_%d.png' % (i)), np.flip(other_flakes[i]['flake_img'], 2))
        cv2.imwrite(os.path.join(output_mask_path, 'val', 'others_%d.png' % (i)), other_flakes[i]['flake_mask'])
        cv2.imwrite(os.path.join(output_contrast_path, 'val', 'others_%d.png' % (i)), other_flakes[i]['flake_contrast'])


def load_one_flake(data_path, result_path, labeled_flake_name_ids, output_img_size=256):
    # if 'mos2-gelpak' in labeled_flake_name_ids:
    #     exp_name = 'laminator'
    #     subexp_name = 'mos2-gelpak'
    # elif 'mos2-pdms' in labeled_flake_name_ids:
    #     exp_name = 'laminator'
    #     subexp_name = 'mos2-pdms'
    # elif 'home-pdms' in labeled_flake_name_ids:
    #     exp_name = 'laminator'
    #     subexp_name = 'home-pdms'
    # elif '6s-1' in labeled_flake_name_ids:
    #     exp_name = 'PDMS-QPress 6s'
    #     subexp_name = '1'
    # elif '6s-2' in labeled_flake_name_ids:
    #     exp_name = 'PDMS-QPress 6s'
    #     subexp_name = '2'
    # elif '6s-3' in labeled_flake_name_ids:
    #     exp_name = 'PDMS-QPress 6s'
    #     subexp_name = '3'
    # elif '6s-4' in labeled_flake_name_ids:
    #     exp_name = 'PDMS-QPress 6s'
    #     subexp_name = '4'
    # else:
    #     raise NotImplementedError
    exp_name, subexp_name, labeled_flake_name_ids = labeled_flake_name_ids

    fname = labeled_flake_name_ids.rsplit('-', 1)[0]
    flake_id = int(labeled_flake_name_ids.rsplit('-', 1)[1])

    flake_info = pickle.load(open(os.path.join(result_path, exp_name, subexp_name, fname), 'rb'))
    img_name = fname.split('.')[0] + '.tiff'
    image = Image.open(os.path.join(data_path, exp_name, subexp_name, img_name))

    im_rgb = np.array(image).astype('float')
    im_hsv = color.rgb2hsv(im_rgb)
    im_hsv[:,:,2] = im_hsv[:,:,2]/255.0
    im_gray = color.rgb2gray(im_rgb)
    imH, imW = im_gray.shape
    
    # get bg image
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

    # build a list of flakes
    image_labelmap = flake_info['image_labelmap']
    flakes = flake_info['flakes']

    # print(labeled_flake_name_ids, flake_id, len(flakes))
    
    flake = flakes[flake_id]

    flake_centroids = flake['flake_center'].astype('int')
    flake_large_bbox = [max(0, flake_centroids[0] - output_img_size//2),
                        min(imH, flake_centroids[0] + output_img_size // 2),
                        max(0, flake_centroids[1] - output_img_size // 2),
                        min(imW, flake_centroids[1] + output_img_size // 2)]

    f_img = np.zeros([output_img_size, output_img_size, 3], dtype=np.uint8)
    r_min = output_img_size//2 - (flake_centroids[0] - flake_large_bbox[0])
    r_max = r_min + (flake_large_bbox[1] - flake_large_bbox[0])
    c_min = output_img_size//2 - (flake_centroids[1] - flake_large_bbox[2])
    c_max = c_min + (flake_large_bbox[3] - flake_large_bbox[2])
    # print(r_min, r_max, c_min, c_max)
    f_img[r_min:r_max, c_min:c_max, :] = im_rgb[flake_large_bbox[0]: flake_large_bbox[1],
                             flake_large_bbox[2]:flake_large_bbox[3], :]
    f_img = f_img.astype(np.uint8)

    # get contour image
    im_tosave_withcontour = im_rgb.astype(np.uint8)
    contour_color = (255, 255, 255)
    contours = flakes[flake_id]['flake_contour_loc']
    contours = np.expand_dims(np.flip(contours), 1).astype(np.int32)
    im_tosave_withcontour = cv2.drawContours(im_tosave_withcontour, contours, -1, contour_color, 2)
    f_img_withcontour = np.zeros([output_img_size, output_img_size, 3], dtype=np.uint8)
    f_img_withcontour[r_min:r_max, c_min:c_max, :] = im_tosave_withcontour[flake_large_bbox[0]: flake_large_bbox[1],
                             flake_large_bbox[2]:flake_large_bbox[3], :]
    # stick withcontour and without contour together
    black_strip = np.zeros([output_img_size, int(output_img_size*0.03), 3], dtype=np.int)
    f_img_withcontour = np.concatenate([f_img_withcontour, black_strip, f_img], 1)

    # get bg image
    f_bg_img = np.zeros([output_img_size, output_img_size, 3], dtype=np.uint8)
    f_bg_img[r_min:r_max, c_min:c_max, :] = bg_rgb[flake_large_bbox[0]: flake_large_bbox[1],
                             flake_large_bbox[2]:flake_large_bbox[3], :]
    f_bg_img = f_bg_img.astype(np.uint8)


    # get mask
    mask = (image_labelmap == flake['flake_id']).astype(np.uint8)
    mask[mask==1] = 255
    f_mask = np.zeros([output_img_size, output_img_size], dtype=np.uint8)
    f_mask[r_min:r_max, c_min:c_max] = mask[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3]]
    f_contrast = np.zeros([output_img_size, output_img_size, 3], dtype=np.uint8)
    f_contrast[r_min:r_max, c_min:c_max, 0] = np.abs(im_gray[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3]] - \
                                                    bg_gray[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3]])
    f_contrast[r_min:r_max, c_min:c_max, 1] = np.abs(im_hsv[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3], 0] - \
                                                    bg_hsv[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3], 0])
    f_contrast = f_contrast.astype(np.uint8)

    flake['flake_img'] = f_img
    flake['flake_mask'] = f_mask
    flake['flake_contrast'] = f_contrast
    flake['flake_img_withcontour'] = f_img_withcontour
    flake['bg_img'] = f_bg_img

    return flake


def load_one_image_v2(data_path, result_path, fname, exp_name, subexp_name, all_labeled_flake_name_ids, output_img_size=256):
    # fname = labeled_flake_name_ids.rsplit('-', 1)[0]
    # flake_id = int(labeled_flake_name_ids.rsplit('-', 1)[1])

    flake_info = pickle.load(open(os.path.join(result_path, exp_name, subexp_name, fname), 'rb'))
    img_name = fname.split('.')[0] + '.tiff'
    image = Image.open(os.path.join(data_path, exp_name, subexp_name, img_name))

    im_rgb = np.array(image).astype('float')
    im_hsv = color.rgb2hsv(im_rgb)
    im_hsv[:,:,2] = im_hsv[:,:,2]/255.0
    im_gray = color.rgb2gray(im_rgb)
    imH, imW = im_gray.shape
    
    # get bg image
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

    # build a list of flakes
    image_labelmap = flake_info['image_labelmap']
    flakes = flake_info['flakes']

    # print(labeled_flake_name_ids, flake_id, len(flakes))
    
    flake = flakes[flake_id]

    flake_centroids = flake['flake_center'].astype('int')
    flake_large_bbox = [max(0, flake_centroids[0] - output_img_size//2),
                        min(imH, flake_centroids[0] + output_img_size // 2),
                        max(0, flake_centroids[1] - output_img_size // 2),
                        min(imW, flake_centroids[1] + output_img_size // 2)]

    f_img = np.zeros([output_img_size, output_img_size, 3], dtype=np.uint8)
    r_min = output_img_size//2 - (flake_centroids[0] - flake_large_bbox[0])
    r_max = r_min + (flake_large_bbox[1] - flake_large_bbox[0])
    c_min = output_img_size//2 - (flake_centroids[1] - flake_large_bbox[2])
    c_max = c_min + (flake_large_bbox[3] - flake_large_bbox[2])
    # print(r_min, r_max, c_min, c_max)
    f_img[r_min:r_max, c_min:c_max, :] = im_rgb[flake_large_bbox[0]: flake_large_bbox[1],
                             flake_large_bbox[2]:flake_large_bbox[3], :]
    f_img = f_img.astype(np.uint8)

    # get contour image
    im_tosave_withcontour = im_rgb.astype(np.uint8)
    contour_color = (255, 255, 255)
    contours = flakes[flake_id]['flake_contour_loc']
    contours = np.expand_dims(np.flip(contours), 1).astype(np.int32)
    im_tosave_withcontour = cv2.drawContours(im_tosave_withcontour, contours, -1, contour_color, 2)
    f_img_withcontour = np.zeros([output_img_size, output_img_size, 3], dtype=np.uint8)
    f_img_withcontour[r_min:r_max, c_min:c_max, :] = im_tosave_withcontour[flake_large_bbox[0]: flake_large_bbox[1],
                             flake_large_bbox[2]:flake_large_bbox[3], :]
    # stick withcontour and without contour together
    black_strip = np.zeros([output_img_size, int(output_img_size*0.03), 3], dtype=np.int)
    f_img_withcontour = np.concatenate([f_img_withcontour, black_strip, f_img], 1)

    # get bg image
    f_bg_img = np.zeros([output_img_size, output_img_size, 3], dtype=np.uint8)
    f_bg_img[r_min:r_max, c_min:c_max, :] = bg_rgb[flake_large_bbox[0]: flake_large_bbox[1],
                             flake_large_bbox[2]:flake_large_bbox[3], :]
    f_bg_img = f_bg_img.astype(np.uint8)


    # get mask
    mask = (image_labelmap == flake['flake_id']).astype(np.uint8)
    mask[mask==1] = 255
    f_mask = np.zeros([output_img_size, output_img_size], dtype=np.uint8)
    f_mask[r_min:r_max, c_min:c_max] = mask[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3]]
    f_contrast = np.zeros([output_img_size, output_img_size, 3], dtype=np.uint8)
    f_contrast[r_min:r_max, c_min:c_max, 0] = np.abs(im_gray[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3]] - \
                                                    bg_gray[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3]])
    f_contrast[r_min:r_max, c_min:c_max, 1] = np.abs(im_hsv[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3], 0] - \
                                                    bg_hsv[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3], 0])
    f_contrast = f_contrast.astype(np.uint8)

    flake['flake_img'] = f_img
    flake['flake_mask'] = f_mask
    flake['flake_contrast'] = f_contrast
    flake['flake_img_withcontour'] = f_img_withcontour
    flake['bg_img'] = f_bg_img

    return flake


# only process those with annotations. load clean annotation of 111x 
def main_label2():
    data_path = '../../data/data_111x_individual'
    result_path = '../../results/data_111x_individual_script/mat'
    
    # output_path = '../../results/data_111x_individual_script/ae_complete_label_img'
    # output_mask_path = '../../results/data_111x_individual_script/ae_complete_label_mask'
    # output_contrast_path = '../../results/data_111x_individual_script/ae_complete_label_contrast'
    # output_bg_path = '../../results/data_111x_individual_script/ae_complete_label_bg'
    # output_contour_path = '../../results/data_111x_individual_script/ae_complete_label_imgcontour'
    # # feature_path = '../../results/data_111x_individual_script/graphene_classifier_with_clean_anno_colorfea-%s/features_withname.p'%'innercontrast-bg-shape'
    # feature_path = '../../results/data_111x_individual_script/graphene_classifier_with_clean_anno_colorfea-%s/features_withname.p'%'innercontrast-bg-shape'
    
    output_path = '../../results/data_111x_individual_script/ae_complete_doublecheck_label_img'
    output_mask_path = '../../results/data_111x_individual_script/ae_complete_doublecheck_label_mask'
    output_contrast_path = '../../results/data_111x_individual_script/ae_complete_doublecheck_label_contrast'
    output_bg_path = '../../results/data_111x_individual_script/ae_complete_doublecheck_label_bg'
    output_contour_path = '../../results/data_111x_individual_script/ae_complete_doublecheck_label_imgcontour'
    feature_path = '../../results/data_111x_individual_script/graphene_classifier_with_clean_anno_doublecheck_colorfea-%s/features_withname.p'%'subsegment-contrast-bg-shape'

    feats = pickle.load(open(feature_path, 'rb'))
    all_labels = feats['labeled_labels']
    all_labeled_feats = feats['labeled_feats']
    all_labeled_flake_name_ids = feats['labeled_flake_name_ids']
    
    # output_path = '../../results/data_111x_individual_script/ae_complete_label_allnegative_img'
    # output_mask_path = '../../results/data_111x_individual_script/ae_complete_label_allnegative_mask'
    # output_contrast_path = '../../results/data_111x_individual_script/ae_complete_label_allnegative_contrast'
    # output_bg_path = '../../results/data_111x_individual_script/ae_complete_label_allnegative_bg'
    # output_contour_path = '../../results/data_111x_individual_script/ae_complete_label_allnegative_imgcontour'
    # feature_path = '../../results/data_111x_individual_script/graphene_classifier_with_clean_anno_allnegative_colorfea-%s/features_withname.p'%'contrast-bg-shape'
    # feats = pickle.load(open(feature_path, 'rb'))
    # all_labels = feats['labeled_labels']
    # all_labeled_feats = feats['labeled_feats']
    # all_unlabeled_feats = feats['unlabeled_feats']
    # all_labeled_flake_name_ids = feats['labeled_flake_name_ids']
    # all_unlabeled_flake_name_ids = feats['unlabeled_flake_name_ids']
    # # include unlabeled flakes
    # all_labels = np.concatenate([all_labels, np.zeros([len(all_unlabeled_feats)])])
    # all_labeled_feats = np.concatenate([all_labeled_feats, all_unlabeled_feats])
    # all_labeled_flake_name_ids = all_labeled_flake_name_ids + all_unlabeled_flake_name_ids
    

    num_labeled = len(all_labels)
    # exp_names = os.listdir(data_path)
    # exp_names = [ename for ename in exp_names if ename[0] not in ['.', '_']]
    # exp_names = ['laminator', 'PDMS-QPress 6s']
    # exp_names.sort()

    graphene_idxs = [l for l in range(num_labeled) if all_labels[l] ==1 ]
    others_idxs = [l for l in range(num_labeled) if all_labels[l] ==0 ]
    graphene_feats = all_labeled_feats[graphene_idxs]
    other_feats = all_labeled_feats[others_idxs]
    # separate_feats = {}
    # separate_feats['graphene_feats'] = graphene_feats
    # separate_feats['other_feats'] = other_feats
    # # separate_feat_save_path = '../../results/data_111x_individual_script/graphene_classifier_with_clean_anno_colorfea-%s/feature_separate.p'%'contrast-bg-shape'
    # separate_feat_save_path = '../../results/data_111x_individual_script/graphene_classifier_with_clean_anno_colorfea-%s/feature_separate.p'%'innercontrast-bg-shape'
    # # separate_feat_save_path = '../../results/data_111x_individual_script/graphene_classifier_with_clean_anno_allnegative_colorfea-%s/feature_separate.p'%'contrast-bg-shape'
    # pickle.dump(separate_feats, open(separate_feat_save_path, 'wb'))

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(os.path.join(output_path, 'train'))
        os.makedirs(os.path.join(output_path, 'val'))
        os.makedirs(os.path.join(output_path, 'all'))

    if not os.path.exists(output_mask_path):
        os.makedirs(output_mask_path)
        os.makedirs(os.path.join(output_mask_path, 'train'))
        os.makedirs(os.path.join(output_mask_path, 'val'))
        os.makedirs(os.path.join(output_mask_path, 'all'))

    if not os.path.exists(output_contrast_path):
        os.makedirs(output_contrast_path)
        os.makedirs(os.path.join(output_contrast_path, 'train'))
        os.makedirs(os.path.join(output_contrast_path, 'val'))
        os.makedirs(os.path.join(output_contrast_path, 'all'))

    if not os.path.exists(output_bg_path):
        os.makedirs(output_bg_path)
        os.makedirs(os.path.join(output_bg_path, 'train'))
        os.makedirs(os.path.join(output_bg_path, 'val'))
        os.makedirs(os.path.join(output_bg_path, 'all'))

    if not os.path.exists(output_contour_path):
        os.makedirs(output_contour_path)
        os.makedirs(os.path.join(output_contour_path, 'train'))
        os.makedirs(os.path.join(output_contour_path, 'val'))
        os.makedirs(os.path.join(output_contour_path, 'all'))

    graphene_flakes = []
    other_flakes = []

    flakes = []
    flakes = Parallel(n_jobs=20)(delayed(load_one_flake)(data_path, result_path, labeled_flake_name_id)
                        for labeled_flake_name_id in all_labeled_flake_name_ids)
    # for exp_name in exp_names:
    #     subexp_names = os.listdir(os.path.join(result_path, exp_name))
    #     for sname in subexp_names:
    #         fnames = os.listdir(os.path.join(result_path, exp_name, sname))
    #         sub_flakes = Parallel(n_jobs=20)(delayed(load_one_image_v2)(data_path, result_path, fname, exp_name, sname, all_labeled_flake_name_ids)
    #                             for fname in fnames)

    #         sub_flakes = list(itertools.chain.from_iterable(sub_flakes))
    #         flakes.extend(sub_flakes)
    
    
    assert len(flakes) == all_labeled_feats.shape[0]
    
    graphene_flakes = [flakes[i] for i in graphene_idxs]
    other_flakes = [flakes[i] for i in others_idxs]

    num_graphene = len(graphene_flakes)
    num_others = len(other_flakes)

    train_ratio = 0.8
    graphene_idxs = np.arange(num_graphene)
    random.Random(123).shuffle(graphene_idxs)
    others_idxs = np.arange(num_others)
    random.Random(123).shuffle(others_idxs)
    
    tr_graphene_idxs = graphene_idxs[:int(train_ratio*num_graphene)]
    val_graphene_idxs = graphene_idxs[int(train_ratio*num_graphene):]
    
    tr_others_idxs = others_idxs[:int(train_ratio*num_others)]
    val_others_idxs = others_idxs[int(train_ratio*num_others):]

    for i in tr_graphene_idxs:
        cv2.imwrite(os.path.join(output_path, 'train', 'graphene_%d.png' % (i)), np.flip(graphene_flakes[i]['flake_img'], 2))
        cv2.imwrite(os.path.join(output_mask_path, 'train', 'graphene_%d.png' % (i)), graphene_flakes[i]['flake_mask'])
        cv2.imwrite(os.path.join(output_contrast_path, 'train', 'graphene_%d.png' % (i)), graphene_flakes[i]['flake_contrast'])
        cv2.imwrite(os.path.join(output_contour_path, 'train', 'graphene_%d.png' % (i)), np.flip(graphene_flakes[i]['flake_img_withcontour'], 2))
        cv2.imwrite(os.path.join(output_bg_path, 'train', 'graphene_%d.png' % (i)), np.flip(graphene_flakes[i]['bg_img'], 2))

    for i in val_graphene_idxs:
        cv2.imwrite(os.path.join(output_path, 'val', 'graphene_%d.png' % (i)), np.flip(graphene_flakes[i]['flake_img'], 2))
        cv2.imwrite(os.path.join(output_mask_path, 'val', 'graphene_%d.png' % (i)), graphene_flakes[i]['flake_mask'])
        cv2.imwrite(os.path.join(output_contrast_path, 'val', 'graphene_%d.png' % (i)), graphene_flakes[i]['flake_contrast'])
        cv2.imwrite(os.path.join(output_contour_path, 'val', 'graphene_%d.png' % (i)), np.flip(graphene_flakes[i]['flake_img_withcontour'], 2))
        cv2.imwrite(os.path.join(output_bg_path, 'val', 'graphene_%d.png' % (i)), np.flip(graphene_flakes[i]['bg_img'], 2))

    for i in tr_others_idxs:
        cv2.imwrite(os.path.join(output_path, 'train', 'others_%d.png' % (i)), np.flip(other_flakes[i]['flake_img'], 2))
        cv2.imwrite(os.path.join(output_mask_path, 'train', 'others_%d.png' % (i)), other_flakes[i]['flake_mask'])
        cv2.imwrite(os.path.join(output_contrast_path, 'train', 'others_%d.png' % (i)), other_flakes[i]['flake_contrast'])
        cv2.imwrite(os.path.join(output_contour_path, 'train', 'others_%d.png' % (i)), np.flip(other_flakes[i]['flake_img_withcontour'], 2))
        cv2.imwrite(os.path.join(output_bg_path, 'train', 'others_%d.png' % (i)), np.flip(other_flakes[i]['bg_img'], 2))

    for i in val_others_idxs:
        cv2.imwrite(os.path.join(output_path, 'val', 'others_%d.png' % (i)), np.flip(other_flakes[i]['flake_img'], 2))
        cv2.imwrite(os.path.join(output_mask_path, 'val', 'others_%d.png' % (i)), other_flakes[i]['flake_mask'])
        cv2.imwrite(os.path.join(output_contrast_path, 'val', 'others_%d.png' % (i)), other_flakes[i]['flake_contrast'])
        cv2.imwrite(os.path.join(output_contour_path, 'val', 'others_%d.png' % (i)), np.flip(other_flakes[i]['flake_img_withcontour'], 2))
        cv2.imwrite(os.path.join(output_bg_path, 'val', 'others_%d.png' % (i)), np.flip(other_flakes[i]['bg_img'], 2))


    os.system('cp %s/%s/* %s/%s/'%(output_path, 'train', output_path, 'all'))
    os.system('cp %s/%s/* %s/%s/'%(output_path, 'val', output_path, 'all'))
    os.system('cp %s/%s/* %s/%s/'%(output_mask_path, 'train', output_mask_path, 'all'))
    os.system('cp %s/%s/* %s/%s/'%(output_mask_path, 'val', output_mask_path, 'all'))
    os.system('cp %s/%s/* %s/%s/'%(output_contrast_path, 'train', output_contrast_path, 'all'))
    os.system('cp %s/%s/* %s/%s/'%(output_contrast_path, 'val', output_contrast_path, 'all'))
    os.system('cp %s/%s/* %s/%s/'%(output_contour_path, 'train', output_contour_path, 'all'))
    os.system('cp %s/%s/* %s/%s/'%(output_contour_path, 'val', output_contour_path, 'all'))
    os.system('cp %s/%s/* %s/%s/'%(output_bg_path, 'train', output_bg_path, 'all'))
    os.system('cp %s/%s/* %s/%s/'%(output_bg_path, 'val', output_bg_path, 'all'))

if __name__ == '__main__':
    # main()
    # main_label()
    main_label2()