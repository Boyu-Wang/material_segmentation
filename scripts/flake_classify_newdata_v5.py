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
parser.add_argument('--exp_eid', default=1, type=int, help='exp end id')
parser.add_argument('--subexp_sid', default=0, type=int, help='subexp start id')
parser.add_argument('--subexp_eid', default=15, type=int, help='subexp end id')
# parser.add_argument('--img_sid', default=0, type=int)
# parser.add_argument('--img_eid', default=294, type=int)
parser.add_argument('--n_jobs', default=30, type=int, help='multiprocessing cores')
# parser.add_argument('--c_sid', default=0, type=int, help='subexp start id')
# parser.add_argument('--c_eid', default=400, type=int, help='subexp end id')
parser.add_argument('--color_fea', default='threesub-contrast-bg-shape', type=str, help='which color feature to use: contrast, ori, both, contrast-bg, ori-bg, both-bg, contrast-bg-shape, threesub-contrast-bg-shape')
parser.add_argument('--clf', default='linear', type=str, help='classifier type: linear, poly')
parser.add_argument('--train_data', default='doublecheck', type=str, help='what is training data: sep, sep-oct, cleananno, doublecheck_allnegative')

args = parser.parse_args()

hyperparams = { 'size_thre': 100, # after detect foreground regions, filter them based on its size. (784=28*28 corresponds to around 5 um regions)
                'clf_method': 'linearsvm', # which classifier to use (linear): 'rigde', 'linearsvm'
                }

def classify_one_image(img_name, info_name, classifier, norm_fea, new_img_save_path, color_fea):
    if not os.path.exists(info_name):
        print('not exist: ', info_name)
        return [], [], []
    img_postfix = img_name.rsplit('/', 1)[1]

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
    
    
    for i in range(num_flakes):
        bwmap = (image_labelmap == i + 1).astype(np.uint8)
        flake_size = flakes[i]['flake_size']
        color = (255, 255, 255)
        contours = flakes[i]['flake_contour_loc']
        contours = np.expand_dims(np.flip(contours), 1).astype(np.int32)
        flake_shape_fea = flakes[i]['flake_shape_fea']

        # print(flakes[i].keys())

        if flake_size > hyperparams['size_thre']:
            im_tosave_withcontour = im_rgb.astype(np.uint8)
            im_tosave_overlay = im_rgb.astype(np.uint8)

            large_flake_idxs.append(i)
            fea_len_contrast = 0
            fea_len_threesub = 0

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
                fea_len_contrast = len(img_fea)
            elif 'both' in color_fea:
                img_fea = np.concatenate([flakes[i]['flake_color_fea'], flakes[i]['flake_contrast_color_fea']])
            else:
                img_fea = np.empty([0])


            if 'subsegment' in color_fea:
                img_fea = np.concatenate([img_fea, flakes[i]['subsegment_features']])
            elif 'threesub' in color_fea:
                img_fea = np.concatenate([img_fea, flakes[i]['subsegment_features_3']])
                fea_len_threesub = len(flakes[i]['subsegment_features_3'])
                assignment = flakes[i]['subsegment_assignment_3']

            elif 'locsub3' in color_fea:
                img_fea = np.concatenate([img_fea, flakes[i]['subsegment_features_3_loc_1']])
                fea_len_threesub = len(flakes[i]['subsegment_features_3_loc_1'])
                assignment = flakes[i]['subsegment_assignment_3_loc_1']
            elif 'twosub' in color_fea:
                img_fea = np.concatenate([img_fea, flakes[i]['subsegment_features_2']])
            elif 'foursub' in color_fea:
                img_fea = np.concatenate([img_fea, flakes[i]['subsegment_features_4']])
         
            # if 'ori' in color_fea:
            #     img_fea = flakes[i]['flake_color_fea']
            # elif 'contrast' in color_fea:
            #     img_fea = flakes[i]['flake_contrast_color_fea']
            # elif 'both' in color_fea:
            #     img_fea = np.concatenate([flakes[i]['flake_color_fea'], flakes[i]['flake_contrast_color_fea']])
            # else:
            #     raise NotImplementedError

            if 'bg' in color_fea:
                img_fea = np.concatenate([img_fea, flakes[i]['flake_bg_color_fea']])
                fea_len_bg = len(flakes[i]['flake_bg_color_fea'])
            else:
                fea_len_bg = 0

            if 'shape' in color_fea:
                img_fea = np.concatenate([img_fea, np.array([flake_shape_fea[0], flake_shape_fea[-1]])])
                fea_len_shape = len(np.array([flake_shape_fea[0], flake_shape_fea[-1]]))
            else:
                fea_len_shape = 0
                
            img_fea = np.expand_dims(img_fea, 0)
            # print('fea', img_fea.shape, img_fea)
            # print('mean', norm_fea['mean'].shape, norm_fea['mean'])
            img_fea -= norm_fea['mean']
            img_fea /= norm_fea['std']
            classifier_coef = classifier.coef_
            # print('contrast', fea_len_contrast)
            # print(classifier_coef[0, :fea_len_contrast])
            # print('threesub', fea_len_contrast+fea_len_threesub)
            # print(classifier_coef[0, fea_len_contrast:fea_len_contrast+fea_len_threesub])
            # print('bg', fea_len_contrast+fea_len_threesub+fea_len_bg)
            # print(classifier_coef[0, fea_len_contrast+fea_len_threesub: fea_len_contrast+fea_len_threesub+fea_len_bg])
            # print('shape', fea_len_contrast+fea_len_threesub+fea_len_bg+fea_len_shape)
            # print(classifier_coef[0, fea_len_contrast+fea_len_threesub+fea_len_bg: fea_len_contrast+fea_len_threesub+fea_len_bg+fea_len_shape])

            pred_cls = classifier.predict(img_fea)
            pred_distance = classifier.decision_function(img_fea)
            pred_scores = classifier_coef * img_fea
            # pred_scores = pred_scores[0]
            # print(pred_scores.shape)
            pred_scores_contrast = np.sum(pred_scores[0, :fea_len_contrast])
            pred_scores_threesub = np.sum(pred_scores[0, fea_len_contrast:fea_len_contrast+fea_len_threesub])
            pred_scores_bg = np.sum(pred_scores[0, fea_len_contrast+fea_len_threesub: fea_len_contrast+fea_len_threesub+fea_len_bg])
            pred_scores_shape = np.sum(pred_scores[0, fea_len_contrast+fea_len_threesub+fea_len_bg: fea_len_contrast+fea_len_threesub+fea_len_bg+fea_len_shape ])
            pred_scores_bias = classifier.intercept_
            # print(pred_distance, np.sum(classifier_coef * img_fea) + classifier.intercept_)
            distances.append(pred_distance)
            distance_flake_ids.append(i)
            flake_sizes.append(flake_size)

            # get patch of the image
            color = (255, 255, 255)
            contours = flakes[i]['flake_contour_loc']
            contours = np.expand_dims(np.flip(contours), 1).astype(np.int32)
            patch_im_tosave_withcontour = cv2.drawContours(im_tosave_withcontour, contours, -1, color, 2)
            # print('get contour')

            if fea_len_threesub > 0:
                n_pixel = len(assignment)
                overlay = np.zeros([n_pixel, 3], dtype=np.uint8)
                overlay[assignment==0] = (255,0,0)
                overlay[assignment==1] = (0,255,0)
                overlay[assignment==2] = (0,0,255)
                patch_rgb= im_tosave_overlay[bwmap>0]
                # print(patch_rgb.shape)
                # print(overlay.shape)
                # print(cv2.addWeighted(np.expand_dims(patch_rgb,0), 0.75, np.expand_dims(overlay,0), 0.25, 0))
                overlay_rgb = cv2.addWeighted(np.expand_dims(patch_rgb,0), 0.75, np.expand_dims(overlay,0), 0.25, 0)
                im_tosave_overlay[bwmap>0] = overlay_rgb[0,:,:]
            # print('get overlay')

            flake_large_bbox = flakes[i]['flake_large_bbox']
            flake_r = flake_large_bbox[1] - flake_large_bbox[0]
            flake_c = flake_large_bbox[3] - flake_large_bbox[2]
            flake_r = max(2 * max(flake_r, flake_c), 100)
            flake_large_bbox[0] = max(0, flake_large_bbox[0]-flake_r)
            flake_large_bbox[1] = min(imH, flake_large_bbox[1]+flake_r)
            flake_large_bbox[2] = max(0, flake_large_bbox[2]-flake_r)
            flake_large_bbox[3] = min(imW, flake_large_bbox[3]+flake_r)
            # print(patch_im_tosave_withcontour.shape)
            # print(im_tosave.shape)
            # print(im_tosave_overlay.shape)
            patch_im_tosave_withcontour = patch_im_tosave_withcontour[flake_large_bbox[0]:flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3], :]
            patch_im_tosave = im_tosave[flake_large_bbox[0]:flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3], :]
            patch_im_tosave_overlay = im_tosave_overlay[flake_large_bbox[0]:flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3], :]

            # stick withcontour and without contour together
            patchH, patchW, _ = patch_im_tosave.shape
            black_strip = np.zeros([patchH, max(2, int(patchW*0.03)), 3], dtype=np.int)
            # print(patch_im_tosave_withcontour.shape)
            # print(black_strip.shape)
            # print(patch_im_tosave.shape)
            # print(patch_im_tosave_overlay.shape)
            final_im_tosave = np.concatenate([patch_im_tosave_withcontour, black_strip, patch_im_tosave, black_strip, patch_im_tosave_overlay], 1)

            # font 
            font = cv2.FONT_HERSHEY_SIMPLEX 
            # org 
            org = (patchH//2, 2*patchW) 
            # fontScale 
            fontScale = 1
            # Blue color in BGR 
            color = (255, 0, 0) 
            # Line thickness of 2 px 
            thickness = 2
            final_im_tosave = np.ascontiguousarray(final_im_tosave, dtype=np.uint8)
            # final_im_tosave = cv2.putText(final_im_tosave, 'score_%.4f\ncontrast_%.4f\nthreesub_%.4f\nbg_%.4f\nshape_%.4f\nbias_%.4f'.format(pred_distance, pred_scores_contrast, pred_scores_threesub, pred_scores_bg, pred_scores_shape, pred_scores_bias), org, font,  
            #        fontScale, color, thickness, cv2.LINE_AA) 
            cv2.putText(final_im_tosave, 'OpenCV', org, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 10) 

            # flake_shape_fea = flakes[flake_id]['flake_shape_fea']
            # save_name = os.path.join(new_img_save_path, '%s_score_%.4f_flakeid-%d_size-%d.png'%(img_postfix, pred_distance, i, flake_size))
            save_name = os.path.join(new_img_save_path, '%s_score_%.4f_contrast_%.4f_threesub_%.4f_bg_%.4f_shape_%.4f_bias_%.4f_flakeid-%d_size-%d.png'%(img_postfix, pred_distance, pred_scores_contrast, pred_scores_threesub, pred_scores_bg, pred_scores_shape, pred_scores_bias, i, flake_size))
            cv2.imwrite(save_name, np.flip(final_im_tosave, 2))

            save_name2 = os.path.join(new_img_save_path, 'score_%.4f_contrast_%.4f_threesub_%.4f_bg_%.4f_shape_%.4f_bias_%.4f_flakeid-%d_size-%d_%s'%(pred_distance, pred_scores_contrast, pred_scores_threesub, pred_scores_bg, pred_scores_shape, pred_scores_bias, i, flake_size, img_postfix))
            cv2.imwrite(save_name2, np.flip(final_im_tosave, 2))



# process one sub exp, read all the data, and do clustering
def classify_one_subexp(subexp_dir, rslt_dir, result_classify_save_path, norm_fea, classifier):
    img_names = os.listdir(subexp_dir)
    img_names = [n_i for n_i in img_names if n_i[0]  not in ['.', '_']]
    img_names.sort()
    # print('process ' + subexp_dir)
    print('n images: %d'%(len(img_names)))

    # Parallel(n_jobs=args.n_jobs)(delayed(classify_one_image)(os.path.join(subexp_dir, img_names[i]), os.path.join(rslt_dir, img_names[i][:-4] + '.p'),
    #                    classifier, norm_fea, result_classify_save_path, args.color_fea) for i in range(len(img_names)))
    
    for i in range(10):
        classify_one_image(os.path.join(subexp_dir, img_names[i]), os.path.join(rslt_dir, img_names[i][:-4] + '.p'),
                       classifier, norm_fea, result_classify_save_path, args.color_fea)


def main():
    
    data_path = '../data/data_mar2020/'
    # result_path = '../results/data_mar2020_script/mat'
    result_path = '../results/data_mar2020_script/mat_1.5_100'
    result_classify_path = '../results/data_mar2020_script/test_classify_graphene_colorfea-%s_clf-%s_data-%s'%(args.color_fea, args.clf, args.train_data)


    if args.train_data == 'doublecheck':
        norm_fea = pickle.load(open('../results/data_111x_individual_script/graphene_classifier_with_clean_anno_doublecheck_colorfea-%s/normfea.p'%args.color_fea, 'rb'))
        if args.clf == 'linear':
            classifier = pickle.load(open('../results/data_111x_individual_script/graphene_classifier_with_clean_anno_doublecheck_colorfea-%s/feanorm_weighted_classifier-linearsvm-1.000000.p'%args.color_fea, 'rb'))
        elif args.clf == 'poly':
            classifier = pickle.load(open('../results/data_111x_individual_script/graphene_classifier_with_clean_anno_doublecheck_colorfea-%s/feanorm_weighted_classifier-poly-1.000000.p'%args.color_fea, 'rb'))
    elif args.train_data == 'doublecheck_allnegative':
        norm_fea = pickle.load(open('../results/data_111x_individual_script/graphene_classifier_with_clean_anno_doublecheck_allnegative_colorfea-%s/normfea.p'%args.color_fea, 'rb'))
        if args.clf == 'linear':
            classifier = pickle.load(open('../results/data_111x_individual_script/graphene_classifier_with_clean_anno_doublecheck_allnegative_colorfea-%s/feanorm_weighted_classifier-linearsvm-1.000000.p'%args.color_fea, 'rb'))
        elif args.clf == 'poly':
            classifier = pickle.load(open('../results/data_111x_individual_script/graphene_classifier_with_clean_anno_doublecheck_allnegative_colorfea-%s/feanorm_weighted_classifier-poly-1.000000.p'%args.color_fea, 'rb'))
    else:
        raise NotImplementedError


    exp_names = os.listdir(data_path)
    exp_names = [ename for ename in exp_names if ename[0] not in ['.', '_']]
    exp_names.sort()

    for d in range(args.exp_sid, args.exp_eid):
        print(d)
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






