"""
Flake classification on new data. 
1. For each image, classify the flakes into 5 different types, and save the result images
    graphene: white
    junk: green
    thin: red
    thick: orange
    multi: pink
2. For a set of images, output top 100 predicted graphenes.

By: Boyu Wang (boywang@cs.stonybrook.edu)
Created Data: 10 May 2020
Last Modified Date: 17 May 2020
"""

import numpy as np
from PIL import Image
import cv2
import argparse
import os
import pickle
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='flake segmentation')
parser.add_argument('--exp_sid', default=0, type=int, help='exp start id')
parser.add_argument('--exp_eid', default=5, type=int, help='exp end id')
parser.add_argument('--subexp_sid', default=0, type=int, help='subexp start id')
parser.add_argument('--subexp_eid', default=15, type=int, help='subexp end id')
parser.add_argument('--n_jobs', default=30, type=int, help='multiprocessing cores')
parser.add_argument('--color_fea', default='threesub-contrast-bg-shape', type=str, help='which color feature to use: threesub-contrast-bg-shape, firstcluster-contrast-bg-shape')
parser.add_argument('--topk', default=100, type=int, help='number of top predicted graphenes to visualize')
args = parser.parse_args()

labelmaps = {0: 'junk', 1: 'thick', 2: 'thin', 3: 'multi', 4: 'graphene'}
# labelmaps = {'thin': 2, 'thick': 1, 'junk': 0, 'multi': 3, 'graphene': 4}
rgb_colormaps = {0: (0,255,0), 1: (255,0, 0), 2: (255,128,0) , 3: (255,0,127), 4: (255,255,255)}

def classify_one_image(img_name, info_name, classifier, norm_fea, new_img_save_path, color_fea):
    if not os.path.exists(info_name):
        print('not exist: ', info_name)
        return [], [], []
    flake_info = pickle.load(open(info_name, 'rb'))
    image = Image.open(img_name)
    im_gray = np.array(image.convert('L', (0.2989, 0.5870, 0.1140, 0))).astype('float')
    imH, imW = im_gray.shape
    im_rgb = np.array(image).astype('float')
    # # build a list of flakes
    num_flakes = len(flake_info['flakes'])
    image_labelmap = flake_info['image_labelmap']
    assert num_flakes == image_labelmap.max()
    flakes = flake_info['flakes']
    large_flake_idxs = []
    im_tosave = im_rgb.astype(np.uint8)
    distance_flake_ids = []
    distances = []
    flake_sizes = []
    for i in range(num_flakes):
        flake_size = flakes[i]['flake_size']
        color = (255, 255, 255)
        contours = flakes[i]['flake_contour_loc']
        contours = np.expand_dims(np.flip(contours), 1).astype(np.int32)
        flake_shape_fea = flakes[i]['flake_shape_fea']

        if flake_size > 100 and flake_size< 50000:
            large_flake_idxs.append(i)
            if 'ori' in color_fea:
                img_fea = flakes[i]['flake_color_fea']
            elif 'innercontrast' in color_fea:
                # include both inner and outer contrast features
                # flatten the inner features
                inner_fea = list(flakes[i]['flake_innercontrast_color_fea'])
                if isinstance(inner_fea[-1], np.ndarray):
                    inner_fea = inner_fea[:-1] + list(inner_fea[-1])
                contrast_fea = list(flakes[i]['flake_contrast_color_fea'])
                if isinstance(contrast_fea[-1], np.ndarray):
                    contrast_fea = contrast_fea[:-1] + list(contrast_fea[-1])
                img_fea = np.array(contrast_fea + inner_fea)
            elif 'contrast' in color_fea:
                img_fea = list(flakes[i]['flake_contrast_color_fea'])
                if isinstance(img_fea[-1], np.ndarray):
                    img_fea = img_fea[:-1] + list(img_fea[-1])
                img_fea = np.array(img_fea)
                # len_1 = len(img_fea)
            elif 'both' in color_fea:
                img_fea = np.concatenate([flakes[i]['flake_color_fea'], flakes[i]['flake_contrast_color_fea']])
            else:
                img_fea = np.empty([0])
                # raise NotImplementedError

            if 'subsegment' in color_fea:
                img_fea = np.concatenate([img_fea, flakes[i]['subsegment_features']])
            elif 'threesub' in color_fea:
                img_fea = np.concatenate([img_fea, flakes[i]['subsegment_features_3']])
            elif 'firstcluster' in color_fea:
                tmp_fea = flakes[i]['subsegment_features_3']
                assert len(tmp_fea) == 32*3
                tmp_fea = tmp_fea[:32]
                img_fea = np.concatenate([img_fea,tmp_fea])
            elif 'locsub3' in color_fea:
                img_fea = np.concatenate([img_fea, flakes[i]['subsegment_features_3_loc_1']])
            elif 'twosub' in color_fea:
                img_fea = np.concatenate([img_fea, flakes[i]['subsegment_features_2']])
            elif 'foursub' in color_fea:
                img_fea = np.concatenate([img_fea, flakes[i]['subsegment_features_4']])

            if 'bg' in color_fea:
                img_fea = np.concatenate([img_fea, flakes[i]['flake_bg_color_fea']])
            if 'shape' in color_fea:
                img_fea = np.concatenate([img_fea, np.array([flake_shape_fea[0], flake_shape_fea[-1]])])
            img_fea = np.expand_dims(img_fea, 0)
            img_fea -= norm_fea['mean']
            img_fea /= (norm_fea['std'] + 1e-10)
            # sub features may contain NaN values due to only have one cluster.
            if np.any(np.isnan(img_fea)):
                continue
            pred_cls = classifier.predict(img_fea)[0]

            color = rgb_colormaps[pred_cls]

            pred_distance = classifier.decision_function(img_fea)[0, 4]
            distances.append(pred_distance)
            distance_flake_ids.append(i)
            flake_sizes.append(flake_size)
            im_tosave = cv2.drawContours(im_tosave, contours, -1, color, 2)

    cv2.imwrite(new_img_save_path, np.flip(im_tosave, 2))
    return distances, distance_flake_ids, flake_sizes



# process one sub exp, read all the data, and classify each image. In the end, visualize top 100 predicted graphenes
def classify_one_subexp(subexp_dir, rslt_dir, result_classify_save_path, result_classify_topk_save_path, norm_fea, classifier):
    img_names = os.listdir(subexp_dir)
    img_names = [n_i for n_i in img_names if n_i[0]  not in ['.', '_']]
    img_names.sort()
    print('n images: %d'%(len(img_names)))

    distances_all = []
    distance_flake_ids = []
    flake_sizes = []
    distance_image_ids = []
    all_returns = Parallel(n_jobs=args.n_jobs)(delayed(classify_one_image)(os.path.join(subexp_dir, img_names[i]), os.path.join(rslt_dir, img_names[i][:-4] + '.p'),
                       classifier, norm_fea, os.path.join(result_classify_save_path, img_names[i]), args.color_fea) for i in range(len(img_names)))
    
    if args.topk > 0:
        tmp_distances_all, tmp_distance_flake_ids, tmp_flake_sizes = zip(*all_returns)
        for i in range(len(img_names)):
            distances_all.extend(tmp_distances_all[i])
            distance_flake_ids.extend(tmp_distance_flake_ids[i])
            flake_sizes.extend(tmp_flake_sizes[i])
            distance_image_ids.extend([i]*len(tmp_distances_all[i]))

        # sort the distance 
        distances_all = np.array(distances_all)
        distances_all = np.reshape(distances_all, [-1])
        sorted_ids = np.argsort(distances_all)
        # from high to low
        sorted_ids = sorted_ids[::-1]
        topk = min(args.topk, len(sorted_ids))
        for k in range(topk):
            tmp_idx = sorted_ids[k]
            image_id = distance_image_ids[tmp_idx]
            flake_id = distance_flake_ids[tmp_idx]
            flake_size = flake_sizes[tmp_idx]
            distance = distances_all[tmp_idx]
            img_name = os.path.join(subexp_dir, img_names[image_id])
            info_name = os.path.join(rslt_dir, img_names[image_id][:-4]+'.p')
            flake_info = pickle.load(open(info_name, 'rb'))
            image = Image.open(img_name)
            im_gray = np.array(image.convert('L', (0.2989, 0.5870, 0.1140, 0))).astype('float')
            imH, imW = im_gray.shape
            im_rgb = np.array(image).astype('float')
            # build a list of flakes
            num_flakes = len(flake_info['flakes'])
            image_labelmap = flake_info['image_labelmap']
            assert num_flakes == image_labelmap.max()
            flakes = flake_info['flakes']
            im_tosave = im_rgb.astype(np.uint8)
            im_tosave_withcontour = im_rgb.astype(np.uint8)
            color = (255, 255, 255)
            contours = flakes[flake_id]['flake_contour_loc']
            contours = np.expand_dims(np.flip(contours), 1).astype(np.int32)
            im_tosave_withcontour = cv2.drawContours(im_tosave_withcontour, contours, -1, color, 2)

            # get patch of the image
            flake_large_bbox = flakes[flake_id]['flake_large_bbox']
            flake_r = flake_large_bbox[1] - flake_large_bbox[0]
            flake_c = flake_large_bbox[3] - flake_large_bbox[2]
            flake_r = int(max(1.2 * max(flake_r, flake_c), 100))
            flake_large_bbox[0] = max(0, flake_large_bbox[0]-flake_r)
            flake_large_bbox[1] = min(imH, flake_large_bbox[1]+flake_r)
            flake_large_bbox[2] = max(0, flake_large_bbox[2]-flake_r)
            flake_large_bbox[3] = min(imW, flake_large_bbox[3]+flake_r)
            im_tosave_withcontour = im_tosave_withcontour[flake_large_bbox[0]:flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3], :]
            im_tosave = im_tosave[flake_large_bbox[0]:flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3], :]

            # stick withcontour and without contour together
            imH, imW, _ = im_tosave.shape
            black_strip = np.zeros([imH, max(2, int(imW*0.03)), 3], dtype=np.int)
            im_tosave = np.concatenate([im_tosave_withcontour, black_strip, im_tosave], 1)

            # flake_shape_fea = flakes[flake_id]['flake_shape_fea']
            save_name = os.path.join(result_classify_topk_save_path, 'id-%d_distance_%.4f_flakeid-%d_size-%d_%s'%(k, distance, flake_id, flake_size, img_names[image_id]))

            cv2.imwrite(save_name, np.flip(im_tosave, 2))


def main():
    data_path = '../data/data_111x_individual/'
    result_path = '../results/data_111x_individual_result/mat_2.0_100'
    result_classify_path = '../results/data_111x_individual_result/classify_colorfea-%s_2.0_100'%(args.color_fea)
    result_classify_topk_path = '../results/data_111x_individual_result/classify_colorfea-%s_2.0_100_top-%d'%(args.color_fea, args.topk)

    norm_fea = pickle.load(open('../results/pretrained_clf/graphene_classifier_with_moreanno_v3_colorfea-%s/normfea.p'%args.color_fea, 'rb'))
    classifier = pickle.load(open('../results/pretrained_clf/graphene_classifier_with_moreanno_v3_colorfea-%s/feanorm_weighted_classifier-linearsvm-10.000000.p'%args.color_fea, 'rb'))

    exp_names = os.listdir(data_path)
    exp_names = [ename for ename in exp_names if ename[0] not in ['.', '_']]
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
            print('classify for images under directory: %s' %(os.path.join(data_path, exp_name, sname)))
            result_classify_save_path = os.path.join(result_classify_path, exp_name, sname)
            if not os.path.exists(result_classify_save_path):
                os.makedirs(result_classify_save_path)
            result_classify_topk_save_path = os.path.join(result_classify_topk_path, exp_name, sname)
            if not os.path.exists(result_classify_topk_save_path):
                os.makedirs(result_classify_topk_save_path)

            classify_one_subexp(os.path.join(data_path, exp_name, sname), os.path.join(result_path, exp_name, sname), result_classify_save_path, result_classify_topk_save_path, norm_fea, classifier)

if __name__ == '__main__':
    main()