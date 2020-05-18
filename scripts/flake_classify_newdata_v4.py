"""
Class on data_111x_individual
Visualization based on scores
Test the classifier (graphene / non-graphane) on other set of experiments
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='flake segmentation')
parser.add_argument('--exp_sid', default=0, type=int, help='exp start id')
parser.add_argument('--exp_eid', default=5, type=int, help='exp end id')
parser.add_argument('--subexp_sid', default=0, type=int, help='subexp start id')
parser.add_argument('--subexp_eid', default=15, type=int, help='subexp end id')
# parser.add_argument('--img_sid', default=0, type=int)
# parser.add_argument('--img_eid', default=294, type=int)
parser.add_argument('--n_jobs', default=30, type=int, help='multiprocessing cores')
# parser.add_argument('--c_sid', default=0, type=int, help='subexp start id')
# parser.add_argument('--c_eid', default=400, type=int, help='subexp end id')
parser.add_argument('--topk', default=100, type=int, help='number of detected graphene')
parser.add_argument('--color_fea', default='contrast-bg-shape', type=str, help='which color feature to use: contrast, ori, both, contrast-bg, ori-bg, both-bg, contrast-bg-shape, threesub-contrast-bg-shape')
parser.add_argument('--clf', default='linear', type=str, help='classifier type: linear, poly')
parser.add_argument('--train_data', default='cleananno', type=str, help='what is training data: sep, sep-oct, cleananno')
parser.add_argument('--flake_size_down', default=100, type=int, help='size lower bound')
parser.add_argument('--flake_size_up', default=20000, type=int, help='size upper bound')
parser.add_argument('--len_area_ratio', default=0.5, type=float, help='upper bound for len area ratio')
parser.add_argument('--fracdim', default=0.9, type=float, help='upper bound for fracdim')

args = parser.parse_args()

labelmaps = {'thin': 1, 'thick': 0, 'glue': 2, 'mixed cluster': 3, 'others': 4}

hyperparams = { 'size_thre': 100, # after detect foreground regions, filter them based on its size. (784=28*28 corresponds to around 5 um regions)
                'clf_method': 'linearsvm', # which classifier to use (linear): 'rigde', 'linearsvm'
                }

def classify_one_image(img_name, info_name, classifier, norm_fea, new_img_save_path, color_fea, topk, flake_size_down, flake_size_up, len_area_ratio, fracdim):
    if not os.path.exists(info_name):
        print('not exist: ', info_name)
        return [], [], []
    flake_info = pickle.load(open(info_name, 'rb'))
    image = Image.open(img_name)
    im_gray = np.array(image.convert('L', (0.2989, 0.5870, 0.1140, 0))).astype('float')
    imH, imW = im_gray.shape
    # im_hsv = np.array(image.convert('HSV')).astype('float')
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
    has_graphene = False
    for i in range(num_flakes):
        flake_size = flakes[i]['flake_size']
        color = (255, 255, 255)
        contours = flakes[i]['flake_contour_loc']
        contours = np.expand_dims(np.flip(contours), 1).astype(np.int32)
        flake_shape_fea = flakes[i]['flake_shape_fea']

        if flake_size > hyperparams['size_thre'] and flake_size< flake_size_up and flake_shape_fea[0] < len_area_ratio and flake_shape_fea[-1] < fracdim: 
            large_flake_idxs.append(i)
            # if 'ori' in color_fea:
            #     img_fea = flakes[i]['flake_color_fea']
            # elif 'contrast' in color_fea:
            #     img_fea = flakes[i]['flake_contrast_color_fea']
            # elif 'both' in color_fea:
            #     img_fea = np.concatenate([flakes[i]['flake_color_fea'], flakes[i]['flake_contrast_color_fea']])
            # else:
            #     raise NotImplementedError

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
            elif 'locsub3' in color_fea:
                img_fea = np.concatenate([img_fea, flakes[i]['subsegment_features_3_loc_1']])
            elif 'twosub' in color_fea:
                img_fea = np.concatenate([img_fea, flakes[i]['subsegment_features_2']])
            elif 'foursub' in color_fea:
                img_fea = np.concatenate([img_fea, flakes[i]['subsegment_features_4']])

            # print('len: ', len_1, img_fea.shape, color_fea)
            if 'bg' in color_fea:
                img_fea = np.concatenate([img_fea, flakes[i]['flake_bg_color_fea']])
                # len_2 = img_fea.shape
            if 'shape' in color_fea:
                img_fea = np.concatenate([img_fea, np.array([flake_shape_fea[0], flake_shape_fea[-1]])])
                # len_3 = img_fea.shape
                
            img_fea = np.expand_dims(img_fea, 0)
            # im_fea_cp = img_fea.copy()
            img_fea -= norm_fea['mean']
            img_fea /= (norm_fea['std'] + 1e-10)
            # sub features may contain NaN values due to only have one cluster.
            if np.any(np.isnan(img_fea)):
                continue
            
            pred_cls = classifier.predict(img_fea)
            # except:
            #     bwmap = (image_labelmap == i + 1).astype(np.uint8)
            #     print(im_rgb[bwmap>0])
            #     print(im_rgb[bwmap>0].mean())
            #     print(len_1, len_2, len_3)
            #     print(flakes[i]['flake_contrast_color_fea'])
            #     print('threesub')
            #     print(flakes[i]['subsegment_features_3'])
            #     print(flakes[i]['subsegment_assignment_3'])
            #     print('bg')
            #     print(flakes[i]['flake_bg_color_fea'])
            #     print('shape')
            #     print(flake_shape_fea)
            #     print(img_fea.shape)
            #     print(norm_fea['mean'].shape)
            #     print(im_fea_cp)
            #     print(img_fea)
            #     print(img_fea.sum())
           
            pred_distance = classifier.decision_function(img_fea)
            distances.append(pred_distance)
            distance_flake_ids.append(i)
            flake_sizes.append(flake_size)
            # if pred_cls == 0:
            #     # thick, red
            #     color = (255, 0, 0)
            if pred_cls == 1:
                # graphene, white
                color = (255, 255, 255)
                has_graphene = True
                im_tosave = cv2.drawContours(im_tosave, contours, -1, color, 2)


    if topk > 0:
        return distances, distance_flake_ids, flake_sizes

    elif has_graphene:
            cv2.imwrite(new_img_save_path, np.flip(im_tosave, 2))

    

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
    flake_sizes = []
    distance_image_ids = []
    all_returns = Parallel(n_jobs=args.n_jobs)(delayed(classify_one_image)(os.path.join(subexp_dir, img_names[i]), os.path.join(rslt_dir, img_names[i][:-4] + '.p'),
                       classifier, norm_fea, os.path.join(result_classify_save_path, img_names[i]), args.color_fea, args.topk, args.flake_size_down, 
                       args.flake_size_up, args.len_area_ratio, args.fracdim) for i in range(len(img_names)))
    
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
        # get all the flakes with distance > 0
        num_positive = len(distances_all[distances_all>0])
        print(num_positive)
        print(distances_all)
        topk = min(args.topk, len(sorted_ids))
        if num_positive > topk:
            topk = num_positive
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
            save_name = os.path.join(result_classify_save_path, 'id-%d_distance_%.4f_flakeid-%d_size-%d_%s'%(k, distance, flake_id, flake_size, img_names[image_id]))
            
            cv2.imwrite(save_name, np.flip(im_tosave, 2))


def main():
    # data_path = '../data/10222019G wtih Suji/center_patch_500_500/'
    # result_path = '../results/10222019G wtih Suji_script/center_patch_500_500/mat'
    # result_classify_path = '../results/10222019G wtih Suji_script/center_patch_500_500/test_sepclassifier_classify_graphene_colorfea-%s_clf-%s_topvisualize'%(args.color_fea, args.clf)

    data_path = '../data/data_111x_individual/'
    # result_path = '../results/data_111x_individual_script/mat'
    # result_path = '../results/data_111x_individual_script/mat_1.5_100'
    result_path = '../results/data_111x_individual_script/mat_2.0_100'
    # result_classify_path = '../results/data_111x_individual_script/test_classify_graphene_colorfea-%s_clf-%s_data-%s_topvisualize-%d_size_from_%d_to_%d'%(args.color_fea, args.clf, args.train_data, args.topk, args.flake_size_down, args.flake_size_up)
    result_classify_path = '../results/data_111x_individual_script/test_classify_graphene_colorfea-%s_clf-%s_data-%s_topvisualize-%d_patch-withori_2.0_100'%(args.color_fea, args.clf, args.train_data, args.topk)


    if args.train_data == 'sep':
        norm_fea = pickle.load(open('../results/data_sep2019_script/graphene_classifier_binary_fea-%s/normfea.p'%args.color_fea, 'rb'))
        if args.clf == 'linear':
            classifier = pickle.load(open('../results/data_sep2019_script/graphene_classifier_binary_fea-%s/feanorm_weighted_classifier-linearsvm-1.000000.p'%args.color_fea, 'rb'))
        elif args.clf == 'poly':
            classifier = pickle.load(open('../results/data_sep2019_script/graphene_classifier_binary_fea-%s/feanorm_weighted_classifier-poly-1.000000.p'%args.color_fea, 'rb'))
    
    elif args.train_data == 'sep-oct':
        norm_fea = pickle.load(open('../results/10222019G wtih Suji_script/center_patch_500_500/graphene_classifier_binary_fea-%s/normfea.p'%args.color_fea, 'rb'))
        if args.clf == 'linear':
            classifier = pickle.load(open('../results/10222019G wtih Suji_script/center_patch_500_500/graphene_classifier_binary_fea-%s/feanorm_weighted_classifier-linearsvm-1.000000.p'%args.color_fea, 'rb'))
        elif args.clf == 'poly':
            classifier = pickle.load(open('../results/10222019G wtih Suji_script/center_patch_500_500/graphene_classifier_binary_fea-%s/feanorm_weighted_classifier-poly-1.000000.p'%args.color_fea, 'rb'))
    
    elif args.train_data == 'cleananno':
        norm_fea = pickle.load(open('../results/data_111x_individual_script/graphene_classifier_with_clean_anno_colorfea-%s/normfea.p'%args.color_fea, 'rb'))
        if args.clf == 'linear':
            classifier = pickle.load(open('../results/data_111x_individual_script/graphene_classifier_with_clean_anno_colorfea-%s/feanorm_weighted_classifier-linearsvm-1.000000.p'%args.color_fea, 'rb'))
        elif args.clf == 'poly':
            classifier = pickle.load(open('../results/data_111x_individual_script/graphene_classifier_with_clean_anno_colorfea-%s/feanorm_weighted_classifier-poly-1.000000.p'%args.color_fea, 'rb'))
    
    elif args.train_data == 'doublecheck':
        norm_fea = pickle.load(open('../results/data_111x_individual_script/graphene_classifier_with_clean_anno_doublecheck_colorfea-%s/normfea.p'%args.color_fea, 'rb'))
        if args.clf == 'linear':
            classifier = pickle.load(open('../results/data_111x_individual_script/graphene_classifier_with_clean_anno_doublecheck_colorfea-%s/feanorm_weighted_classifier-linearsvm-1.000000.p'%args.color_fea, 'rb'))
        elif args.clf == 'poly':
            classifier = pickle.load(open('../results/data_111x_individual_script/graphene_classifier_with_clean_anno_doublecheck_colorfea-%s/feanorm_weighted_classifier-poly-1.000000.p'%args.color_fea, 'rb'))
    
    else:
        raise NotImplementedError


    exp_names = os.listdir(data_path)
    exp_names = [ename for ename in exp_names if ename[0] not in ['.', '_']]
    exp_names.sort()

    for d in range(args.exp_sid, args.exp_eid):
        exp_name = exp_names[d]
        subexp_names = os.listdir(os.path.join(data_path, exp_name))
        subexp_names = [sname for sname in subexp_names if sname[0] not in ['.', '_']]
        subexp_names = [sname for sname in subexp_names if os.path.isdir(os.path.join(data_path, exp_name, sname))]
        subexp_names.sort()
        # print(subexp_names)

        # process each subexp
        for s_d in range(args.subexp_sid, min(len(subexp_names), args.subexp_eid)):
            sname = subexp_names[s_d]
            print('classify for images under directory: %s' %(os.path.join(data_path, exp_name, sname)))
            result_classify_save_path = os.path.join(result_classify_path, exp_name, sname)
            if not os.path.exists(result_classify_save_path):
                os.makedirs(result_classify_save_path)

            classify_one_subexp(os.path.join(data_path, exp_name, sname), os.path.join(result_path, exp_name, sname), result_classify_save_path, norm_fea, classifier)




if __name__ == '__main__':
    main()






