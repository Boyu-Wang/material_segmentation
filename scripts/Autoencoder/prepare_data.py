"""
Crop flake and put it in the center of image for autoencoder

By: Boyu Wang (boywang@cs.stonybrook.edu)
Created Data: 6 May 2019
Last Modified Date: 6 May 2019
"""

import os
import numpy as np
import cv2
import pickle
import argparse
from PIL import Image
from multiprocessing import Pool
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='flake segmentation')
parser.add_argument('--exp_sid', default=5, type=int, help='exp start id')
parser.add_argument('--exp_eid', default=6, type=int, help='exp end id')
parser.add_argument('--subexp_sid', default=0, type=int, help='subexp start id')
parser.add_argument('--subexp_eid', default=1, type=int, help='subexp end id')
parser.add_argument('--n_jobs', default=8, type=int, help='multiprocessing cores')
# parser.add_argument('--c_sid', default=0, type=int, help='subexp start id')
# parser.add_argument('--c_eid', default=400, type=int, help='subexp end id')

args = parser.parse_args()
hyperparams = { 'size_thre': 784, # after detect foreground regions, filter them based on its size. (784=28*28 corresponds to 5 around um regions)
                }


def process_one_img(ins_name, img_dir, rslt_dir, ae_img_path, ae_mask_path, output_img_size=256):
    ins_name = ins_name.split('.')[0]
    # load the detected flake and get features for the flake
    flake_info = pickle.load(open(os.path.join(rslt_dir, ins_name+'.p'), 'rb'))
    image = Image.open(os.path.join(img_dir, ins_name+'.tif'))
    im_gray = np.array(image.convert('L', (0.2989, 0.5870, 0.1140, 0))).astype('float')
    imH, imW = im_gray.shape
    # im_hsv = np.array(image.convert('HSV')).astype('float')
    im_rgb = np.array(image).astype(np.uint8)
    # build a list of flakes
    num_flakes = len(flake_info['flakes'])
    image_labelmap = flake_info['image_labelmap']
    assert num_flakes == image_labelmap.max()
    flakes = flake_info['flakes']
    large_flake_idxs = []
    cnt = 0
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

            # get mask
            mask = (image_labelmap == i + 1).astype(np.uint8)
            mask[mask==1] = 255
            f_mask = np.zeros([output_img_size, output_img_size], dtype=np.uint8)
            f_mask[r_min:r_max, c_min:c_max] = mask[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3]]

            cv2.imwrite(os.path.join(ae_img_path, '%s-%d.png' % (ins_name, cnt)), np.flip(f_img, 2))
            cv2.imwrite(os.path.join(ae_mask_path, '%s-%d.png' % (ins_name, cnt)), f_mask)
            cnt += 1

# process all the flakes
def main():
    data_path = '../../data/data_jan2019'
    result_path = '../../results/data_jan2019_script/mat'
    ae_img_path = '../../results/data_jan2019_script/ae_img/'
    # ae_img_hsv_path = '../../results/data_jan2019_script/ae_img_hsv'
    ae_mask_path = '../../results/data_jan2019_script/ae_mask/'

    exp_names = os.listdir(data_path)
    exp_names.sort()

    if not os.path.exists(ae_img_path):
        os.makedirs(ae_img_path)
    # if not os.path.exists(ae_img_hsv_path):
    #     os.makedirs(ae_img_hsv_path)
    if not os.path.exists(ae_mask_path):
        os.makedirs(ae_mask_path)

    for d in range(args.exp_sid, args.exp_eid):
        exp_name = exp_names[d]
        subexp_names = os.listdir(os.path.join(data_path, exp_name))
        subexp_names = [sname for sname in subexp_names if os.path.isdir(os.path.join(data_path, exp_name, sname))]
        subexp_names.sort()

        # process each subexp
        for s_d in range(args.subexp_sid, min(len(subexp_names), args.subexp_eid)):
            sname = subexp_names[s_d]

            img_dir_name = os.path.join(data_path, exp_name, sname)
            rslt_dir_name = os.path.join(result_path, exp_name, sname)

            sub_ae_img_path = os.path.join(ae_img_path, exp_name, sname)
            sub_ae_mask_path = os.path.join(ae_mask_path, exp_name, sname)
            if not os.path.exists(sub_ae_img_path):
                os.makedirs(sub_ae_img_path)
            if not os.path.exists(sub_ae_mask_path):
                os.makedirs(sub_ae_mask_path)

            img_names = os.listdir(img_dir_name)
            img_names.sort()
            # for i_d in range(len(img_names)):
            #     process_one_img(img_names[i_d], img_dir_name, rslt_dir_name, sub_ae_img_path, sub_ae_mask_path, output_img_size=256)

            Parallel(n_jobs=args.n_jobs)(delayed(process_one_img)(img_names[i_d], img_dir_name, rslt_dir_name, sub_ae_img_path, sub_ae_mask_path, output_img_size=256)
                                                      for i_d in range(len(img_names)))
            #


# load the detected flake and get features for the flake
def load_one_image(img_name, info_name, output_img_size=256):
    flake_info = pickle.load(open(info_name, 'rb'))
    image = Image.open(img_name)
    im_gray = np.array(image.convert('L', (0.2989, 0.5870, 0.1140, 0))).astype('float')
    imH, imW = im_gray.shape
    # im_hsv = np.array(image.convert('HSV')).astype('float')
    im_rgb = np.array(image).astype('float')
    # build a list of flakes
    num_flakes = len(flake_info['flakes'])
    image_labelmap = flake_info['image_labelmap']
    assert num_flakes == image_labelmap.max()
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

            # get mask
            mask = (image_labelmap == i + 1).astype(np.uint8)
            mask[mask==1] = 255
            f_mask = np.zeros([output_img_size, output_img_size], dtype=np.uint8)
            f_mask[r_min:r_max, c_min:c_max] = mask[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3]]

            # flakes[i]['flake_large_bbox'] = flake_large_bbox
            flakes[i]['flake_img'] = f_img
            flakes[i]['flake_mask'] = f_mask

    flakes = [flakes[j] for j in large_flake_idxs]
    return flakes


def locate_flakes(item_names, img_flakes, img_names):
    num = len(item_names)
    flakes = []
    for i in range(num):
        f_name, f_id = item_names[i].split('-')
        f_id = int(f_id)
        idx = img_names.index(f_name)
        flakes.append(img_flakes[idx][f_id])

    return flakes

# only process those with annotations
def main_label():
    subexp_name = 'YoungJaeShinSamples/4'
    # anno_file = '../../data/data_jan2019_anno/anno_flakeglue_useryoungjae.db'
    data_path = os.path.join('../../data/data_jan2019', subexp_name)
    result_path = os.path.join('../../results/data_jan2019_script/mat', subexp_name)
    output_path = '../../results/data_jan2019_script/ae_label_img'
    output_mask_path = '../../results/data_jan2019_script/ae_label_mask'

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(os.path.join(output_path, 'train'))
        os.makedirs(os.path.join(output_path, 'val'))

    if not os.path.exists(output_mask_path):
        os.makedirs(output_mask_path)
        os.makedirs(os.path.join(output_mask_path, 'train'))
        os.makedirs(os.path.join(output_mask_path, 'val'))

    flake_save_name = '../../results/data_jan2019_script/flakeglue_clf/YoungJaeShinSamples/4/train_val_split.p'
    to_load = pickle.load(open(flake_save_name, 'rb'))
    train_names = to_load['train_names']
    # train_labels = to_load['train_labels']
    val_names = to_load['val_names']
    # val_labels = to_load['val_labels']

    img_names = os.listdir(data_path)
    img_names.sort()
    img_flakes = Parallel(n_jobs=args.n_jobs)(delayed(load_one_image)(os.path.join(data_path, img_names[i]),os.path.join(result_path, img_names[i][:-4] + '.p'))
                                                        for i in range(len(img_names)))
    print('loading done')
    # load corresponding flakes
    train_flakes = locate_flakes(train_names, img_flakes, img_names)
    val_flakes = locate_flakes(val_names, img_flakes, img_names)
    print('locating done')

    for i in range(len(train_flakes)):
        cv2.imwrite(os.path.join(output_path, 'train', '%d.png' % (i)), np.flip(train_flakes[i]['flake_img'], 2))
        cv2.imwrite(os.path.join(output_mask_path, 'train', '%d.png' % (i)), train_flakes[i]['flake_mask'])

    for i in range(len(val_flakes)):
        cv2.imwrite(os.path.join(output_path, 'val', '%d.png' % (i)), np.flip(val_flakes[i]['flake_img'], 2))
        cv2.imwrite(os.path.join(output_mask_path, 'val', '%d.png' % (i)), val_flakes[i]['flake_mask'])


if __name__ == '__main__':
    # main()
    main_label()