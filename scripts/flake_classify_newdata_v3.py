"""
Visualization based on scores
Test the classifier (thick/thin/glue) on other set of experiments
For classification, only consider regions larger than 28*28=784.
Red boundary means thick, gree boundary means thin, black boundary means glue, white bounday means regions smaller than 784
"""


import numpy as np
from PIL import Image
import cv2
import argparse
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='flake segmentation')
parser.add_argument('--exp_sid', default=0, type=int, help='exp start id')
parser.add_argument('--exp_eid', default=4, type=int, help='exp end id')
# parser.add_argument('--subexp_sid', default=0, type=int, help='subexp start id')
# parser.add_argument('--subexp_eid', default=1, type=int, help='subexp end id')
# parser.add_argument('--img_sid', default=0, type=int)
# parser.add_argument('--img_eid', default=294, type=int)
parser.add_argument('--n_jobs', default=30, type=int, help='multiprocessing cores')
# parser.add_argument('--c_sid', default=0, type=int, help='subexp start id')
# parser.add_argument('--c_eid', default=400, type=int, help='subexp end id')
parser.add_argument('--color_fea', default='contrast', type=str, help='which color feature to use: contrast, ori, both')
parser.add_argument('--clf', default='linear', type=str, help='classifier type: linear, poly')

args = parser.parse_args()

labelmaps = {'thin': 1, 'thick': 0, 'glue': 2, 'mixed cluster': 3, 'others': 4}

hyperparams = { 'size_thre': 100, # after detect foreground regions, filter them based on its size. (784=28*28 corresponds to around 5 um regions)
                'clf_method': 'linearsvm', # which classifier to use (linear): 'rigde', 'linearsvm'
                }

def classify_one_image(img_name, info_name, classifier, norm_fea, new_img_save_path, color_fea):
    flake_info = pickle.load(open(info_name, 'rb'))
    # image = Image.open(img_name)
    # im_gray = np.array(image.convert('L', (0.2989, 0.5870, 0.1140, 0))).astype('float')
    # imH, imW = im_gray.shape
    # # im_hsv = np.array(image.convert('HSV')).astype('float')
    # im_rgb = np.array(image).astype('float')
    # # build a list of flakes
    num_flakes = len(flake_info['flakes'])
    image_labelmap = flake_info['image_labelmap']
    assert num_flakes == image_labelmap.max()
    flakes = flake_info['flakes']
    large_flake_idxs = []
    # im_tosave = im_rgb.astype(np.uint8)
    distance_flake_ids = []
    distances = []
    for i in range(num_flakes):
        flake_size = flakes[i]['flake_size']
        color = (255, 255, 255)
        contours = flakes[i]['flake_contour_loc']
        contours = np.expand_dims(np.flip(contours), 1).astype(np.int32)
        if flake_size > hyperparams['size_thre']:
            large_flake_idxs.append(i)
            if color_fea == 'ori':
                img_fea = flakes[i]['flake_color_fea']
            elif color_fea == 'contrast':
                img_fea = flakes[i]['flake_contrast_color_fea']
            elif color_fea == 'both':
                img_fea = np.concatenate([flakes[i]['flake_color_fea'], flakes[i]['flake_contrast_color_fea']])
            # img_fea = np.concatenate([flakes[i]['flake_shape_fea'], flakes[i]['flake_color_fea']])
            img_fea = np.expand_dims(img_fea, 0)
            img_fea -= norm_fea['mean']
            img_fea /= norm_fea['std']
            pred_cls = classifier.predict(img_fea)
            pred_distance = classifier.decision_function(img_fea)
            distances.append(pred_distance)
            distance_flake_ids.append(i)
            # if pred_cls == 0:
            #     # thick, red
            #     color = (255, 0, 0)
            # elif pred_cls == 1:
            #     # graphene, white
            #     color = (255, 255, 255)

            # im_tosave = cv2.drawContours(im_tosave, contours, -1, color, 2)

    # cv2.imwrite(new_img_save_path, np.flip(im_tosave, 2))

    # plt.close()

    return distances, distance_flake_ids


# process one sub exp, read all the data, and do clustering
def classify_one_subexp(subexp_dir, rslt_dir, result_classify_save_path, norm_fea, classifier):
    img_names = os.listdir(subexp_dir)
    img_names = [n_i for n_i in img_names if n_i[0]  not in ['.', '_']]
    img_names.sort()
    # print('process ' + subexp_dir)
    print('n images: %d'%(len(img_names)))

    # Parallel(n_jobs=args.n_jobs)(delayed(classify_one_image)(os.path.join(subexp_dir, img_names[i]), os.path.join(rslt_dir, img_names[i][:-4] + '.p'),
                       # classifier, norm_fea, os.path.join(result_classify_save_path, img_names[i]), args.color_fea) for i in range(len(img_names)))
    distances_all = []
    distance_flake_ids = []
    distance_image_ids = []
    for i in range(len(img_names)):
        distances_i, distance_flake_ids_i = classify_one_image(os.path.join(subexp_dir, img_names[i]), os.path.join(rslt_dir, img_names[i][:-4]+'.p'), classifier, norm_fea, os.path.join(result_classify_save_path, img_names[i]), args.color_fea)
        distances_all.extend(distances_i)
        distance_flake_ids.extend(distance_flake_ids_i)
        distance_image_ids.extend([i]*len(distances_i))

    # sort the distance 
    distances_all = np.array(distances_all)
    distances_all = np.reshape(distances_all, [-1])
    # print(distances_all.shape)
    sorted_ids = np.argsort(distances_all)
    # from high to low
    sorted_ids = sorted_ids[::-1]
    topk = 100
    print(sorted_ids)

    for k in range(topk):
        tmp_idx = sorted_ids[k]
        # print(tmp_idx)
        image_id = distance_image_ids[tmp_idx]
        flake_id = distance_flake_ids[tmp_idx]
        distance = distances_all[tmp_idx]
        save_name = os.path.join(result_classify_save_path, 'id-%d_distance_%.4f_%s'%(k, distance, img_names[image_id]))
        # print(save_name)
        # print(img_names[image_id])
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
        color = (255, 255, 255)
        contours = flakes[flake_id]['flake_contour_loc']
        contours = np.expand_dims(np.flip(contours), 1).astype(np.int32)
        im_tosave = cv2.drawContours(im_tosave, contours, -1, color, 2)

        cv2.imwrite(save_name, np.flip(im_tosave, 2))


def main():
    data_path = '../data/10222019G wtih Suji/center_patch_500_500/'
    result_path = '../results/10222019G wtih Suji_script/center_patch_500_500/mat'
    result_classify_path = '../results/10222019G wtih Suji_script/center_patch_500_500/test_sepclassifier_classify_graphene_colorfea-%s_clf-%s_topvisualize'%(args.color_fea, args.clf)

    # data_path = '../data/10222019G wtih Suji/patch_500_500/'
    # result_path = '../results/10222019G wtih Suji_script/patch_500_500/mat'
    # result_classify_path = '../results/10222019G wtih Suji_script/patch_500_500/classify_graphene_colorfea-%s'%args.color_fea

    norm_fea = pickle.load(open('../results/data_sep2019_script/graphene_classifier_binary_fea-%s/normfea.p'%args.color_fea, 'rb'))
    if args.clf == 'linear':
        classifier = pickle.load(open('../results/data_sep2019_script/graphene_classifier_binary_fea-%s/feanorm_weighted_classifier-linearsvm-1.000000.p'%args.color_fea, 'rb'))
    elif args.clf == 'poly':
        classifier = pickle.load(open('../results/data_sep2019_script/graphene_classifier_binary_fea-%s/feanorm_weighted_classifier-poly-1.000000.p'%args.color_fea, 'rb'))

    exp_names = os.listdir(data_path)
    exp_names = [ename for ename in exp_names if ename[0] not in ['.', '_']]
    exp_names.sort()

    for d in range(args.exp_sid, args.exp_eid):
        exp_name = exp_names[d]
        # subexp_names = os.listdir(os.path.join(data_path, exp_name))
        # subexp_names = [sname for sname in subexp_names if sname[0] not in ['.', '_']]
        # subexp_names = [sname for sname in subexp_names if os.path.isdir(os.path.join(data_path, exp_name, sname))]
        # subexp_names.sort()
        # # print(subexp_names)

        # # process each subexp
        # for s_d in range(args.subexp_sid, min(len(subexp_names), args.subexp_eid)):
        #     sname = subexp_names[s_d]
            # print('classify for images under directory: %s' %(os.path.join(data_path, exp_name, sname)))
        result_classify_save_path = os.path.join(result_classify_path, exp_name)
        if not os.path.exists(result_classify_save_path):
            os.makedirs(result_classify_save_path)

        classify_one_subexp(os.path.join(data_path, exp_name), os.path.join(result_path, exp_name), result_classify_save_path, norm_fea, classifier)




if __name__ == '__main__':
    main()






