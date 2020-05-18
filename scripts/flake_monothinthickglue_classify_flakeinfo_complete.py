"""
Complete annotation
classify graphene, thick, thin, glue
use the detailed labeled dataset, create a train/validation set
train different classifiers. test and visualize their performance.

By: Boyu Wang (boywang@cs.stonybrook.edu)
Created Data: 8 May 2019
Last Modified Date: 9 May 2019
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
import sqlite3
from multiprocessing import Pool
from joblib import Parallel, delayed
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import RidgeClassifier
import random


# labelmaps = {'thin': 1, 'thick': -1, 'others': 0}
labelmaps = {'thin': 1, 'thick': 0, 'glue': 2, 'mixed cluster': 3, 'others': 4, 'graphene': 5}
labelmapsback = { 0: 'thick', 1: 'thin', 2: 'glue', 3: 'graphene'}

hyperparams = { 'clf_method': 'linearsvm', # which classifier to use (linear): 'ridge', 'linearsvm', 'rbfkernelsvm', 'polysvm'
                'C': 5, # parameter to tune for SVM
                }

# split into train and val
def split_trainval(itemname_labels):
    num_total = len(itemname_labels)
    train_ratio = 0.8
    all_labels = [itemname_labels[i][1] for i in range(num_total)]
    thick_idxs = [i for i in range(num_total) if all_labels[i]==0]
    thin_idxs = [i for i in range(num_total) if all_labels[i] ==1]
    glue_idxs = [i for i in range(num_total) if all_labels[i]==2]
    num_thick = len(thick_idxs)
    num_thin = len(thin_idxs)
    num_glue = len(glue_idxs)
    print('total: %d, thick: %d, thin: %d, glue: %d'%(num_total, num_thick, num_thin, num_glue))

    random.seed(0)
    random.shuffle(thick_idxs)
    random.shuffle(thin_idxs)
    random.shuffle(glue_idxs)

    tr_thick_idxs = thick_idxs[:int(train_ratio*num_thick)]
    tr_thin_idxs = thin_idxs[:int(train_ratio*num_thin)]
    tr_glue_idxs = glue_idxs[:int(train_ratio*num_glue)]

    val_thick_idxs = thick_idxs[int(train_ratio * num_thick):]
    val_thin_idxs = thin_idxs[int(train_ratio * num_thin):]
    val_glue_idxs = glue_idxs[int(train_ratio * num_glue):]

    tr_idxs = tr_thick_idxs + tr_thin_idxs + tr_glue_idxs
    train_names = [itemname_labels[i][0] for i in tr_idxs]
    train_labels = [itemname_labels[i][1] for i in tr_idxs]

    val_idxs = val_thick_idxs + val_thin_idxs + val_glue_idxs
    val_names = [itemname_labels[i][0] for i in val_idxs]
    val_labels = [itemname_labels[i][1] for i in val_idxs]

    return train_names, train_labels, val_names, val_labels


# load the detected flake and get features for the flake
def load_one_image(img_name, info_name):
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
        if flake_size > 784:
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
            flakes[i]['flake_img'] = im_rgb[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3], :].astype(np.uint8)

    flakes = [flakes[j] for j in large_flake_idxs]
    return flakes


def vis_error(pred_cls, pred_scores, gt_cls, flakes, img_save_path, item_names, prefix):
    num = len(pred_cls)

    for i in range(num):
        if item_names[i] == '9graphene-3..p-graphene-0':
            print(gt_cls[i], pred_cls[i])

        if pred_cls[i] != gt_cls[i]:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            contours = flakes[i]['flake_contour_loc']
            contours[:,0] = contours[:,0] - flakes[i]['flake_large_bbox'][0]
            contours[:,1] = contours[:,1] - flakes[i]['flake_large_bbox'][2]
            contours = np.expand_dims(np.flip(contours), 1).astype(np.int32)
            color = None
            if pred_cls[i] == 1:
                # thin, red
                color = (255,0,0)
                # contour_img = cv2.drawContours(flakes[i]['flake_img'], contours, -1, (255,0,0), 2)
            if pred_cls[i] == 0:
                # thick, green
                color = (0,255,0)
                # contour_img = cv2.drawContours(flakes[i]['flake_img'], contours, -1, (0,255,0), 2)
            if pred_cls[i] == 2:
                # glue, black
                color = (0, 0, 0)
                # contour_img = cv2.drawContours(flakes[i]['flake_img'], contours, -1, (0, 0, 0), 2)
            if pred_cls[i] == 3:
                # graphene, white
                color = (255, 255, 255)
                # contour_img = cv2.drawContours(flakes[i]['flake_img'], contours, -1, (255, 255, 255), 2)

            contour_img = cv2.drawContours(flakes[i]['flake_img'], contours, -1, color, 2)

            ax.imshow(contour_img)

            ax.axis('off')
            if item_names[i] == '9graphene-3..p-graphene-0':
                print('error', gt_cls[i], pred_cls[i])
                print(os.path.join(img_save_path, '%s_gt_%s_pred_%s_%s.png'%(prefix, labelmapsback[gt_cls[i]], labelmapsback[pred_cls[i]], item_names[i] )))
            # print(prefix, gt_cls[i], pred_scores[i], item_names[i])
            # fig.savefig(os.path.join(img_save_path, '%s_gt_%d_pred_%.2f_%s.png'%(prefix, gt_cls[i], pred_scores[i], item_names[i] )))
            fig.savefig(os.path.join(img_save_path, '%s_gt_%s_pred_%s_%s.png'%(prefix, labelmapsback[gt_cls[i]], labelmapsback[pred_cls[i]], item_names[i] )))

            plt.close()
        elif gt_cls[i] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            contours = flakes[i]['flake_contour_loc']
            contours[:,0] = contours[:,0] - flakes[i]['flake_large_bbox'][0]
            contours[:,1] = contours[:,1] - flakes[i]['flake_large_bbox'][2]
            contours = np.expand_dims(np.flip(contours), 1).astype(np.int32)
            color = None
            if pred_cls[i] == 1:
                # thin, red
                color = (255,0,0)
                # contour_img = cv2.drawContours(flakes[i]['flake_img'], contours, -1, (255,0,0), 2)
            if pred_cls[i] == 0:
                # thick, green
                color = (0,255,0)
                # contour_img = cv2.drawContours(flakes[i]['flake_img'], contours, -1, (0,255,0), 2)
            if pred_cls[i] == 2:
                # glue, black
                color = (0, 0, 0)
                # contour_img = cv2.drawContours(flakes[i]['flake_img'], contours, -1, (0, 0, 0), 2)
            if pred_cls[i] == 3:
                # graphene, white
                color = (255, 255, 255)
                # contour_img = cv2.drawContours(flakes[i]['flake_img'], contours, -1, (255, 255, 255), 2)

            contour_img = cv2.drawContours(flakes[i]['flake_img'], contours, -1, color, 2)
            ax.imshow(contour_img)

            ax.axis('off')
            if item_names[i] == '9graphene-3..p-graphene-0':
                print('same', gt_cls[i], pred_cls[i])
                print(color)
                print(os.path.join(img_save_path, '%s_gt_%s_pred_%s_%s.png'%(prefix, labelmapsback[gt_cls[i]], labelmapsback[pred_cls[i]], item_names[i] )))
            # print(prefix, gt_cls[i], pred_scores[i], item_names[i])
            # fig.savefig(os.path.join(img_save_path, '%s_gt_%d_pred_%.2f_%s.png'%(prefix, gt_cls[i], pred_scores[i], item_names[i] )))
            fig.savefig(os.path.join(img_save_path, '%s_gt_%s_pred_%s_%s.png'%(prefix, labelmapsback[gt_cls[i]], labelmapsback[pred_cls[i]], item_names[i] )))

            plt.close()


def main():
    subexp_name = 'YoungJaeShinSamples/4'
    clf_path = os.path.join('../results/data_jan2019_script/thickthinglue_clf_complete', subexp_name)
    data_path = '../data/data_sep2019/EXP1/09192019 Graphene'
    save_path = os.path.join('../results/data_sep2019_script/EXP1/09192019 Graphene/graphenethickthinglue_clf_complete_v2')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # get the train/val split
    split_name = os.path.join(clf_path, 'train_val_split.p')
    if os.path.exists(split_name):
        to_load = pickle.load(open(split_name, 'rb'))
        train_names = to_load['train_names']
        train_labels = to_load['train_labels']
        val_names = to_load['val_names']
        val_labels = to_load['val_labels']
    else:
        raise NotImplementedError

    # load flakes
    flake_save_name = os.path.join(clf_path, 'train_val_data.p')
    if os.path.exists(flake_save_name):
        to_load = pickle.load(open(flake_save_name, 'rb'))
        train_flakes = to_load['train_flakes']
        train_feats = to_load['train_feats']
        val_flakes = to_load['val_flakes']
        val_feats = to_load['val_feats']
    print('loading done')

    graphene_path = '../results/data_sep2019_script/labelmat_graphene/EXP1/09192019 Graphene'
    # load graphene
    graphene_flakes = []
    graphene_feats = []
    graphene_names = []
    fnames = os.listdir(graphene_path)
    for fname in fnames:
        tmp_flake = pickle.load(open(os.path.join(graphene_path, fname), 'rb'))
        tmp_flake = tmp_flake['flakes']
        # print(tmp_flake)
        # print(len(tmp_flake))
        if len(tmp_flake) > 0:
            # print(os.path.join(data_path, fname[:-2] + 'tiff'))
            image = Image.open(os.path.join(data_path, fname[:-2] + 'tiff'))
            im_rgb = np.array(image).astype('float')
            imH, imW, _ = im_rgb.shape
            
            graphene_flakes.extend(tmp_flake)
            for i in range(len(tmp_flake)):
                graphene_names.append(fname+'-graphene-'+str(i))
                f_mask_r_min, f_mask_r_max, f_mask_c_min, f_mask_c_max = tmp_flake[i]['flake_exact_bbox']
                f_mask_height = f_mask_r_max - f_mask_r_min
                f_mask_width = f_mask_c_max - f_mask_c_min
                flake_large_bbox = [max(0, f_mask_r_min - int(0.5 * f_mask_height)),
                                    min(imH, f_mask_r_max + int(0.5 * f_mask_height)),
                                    max(0, f_mask_c_min - int(0.5 * f_mask_width)),
                                    min(imW, f_mask_c_max + int(0.5 * f_mask_width))]
                tmp_flake[i]['flake_large_bbox'] = flake_large_bbox
                tmp_flake[i]['flake_img'] = im_rgb[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3], :].astype(np.uint8)

    num_graphene = len(graphene_flakes)
    print("number of graphene: {}".format(num_graphene))
    # load features
    for g_flake in graphene_flakes:
        graphene_feats.append(np.concatenate([g_flake['flake_shape_fea'],g_flake['flake_color_fea']]))
    graphene_feats = np.stack(graphene_feats)
    
    # load thick
    thick_path = '../results/data_sep2019_script/labelmat_thick/EXP1/09192019 Graphene'
    thick_flakes = []
    thick_feats = []
    thick_names = []
    fnames = os.listdir(thick_path)
    for fname in fnames:
        tmp_flake = pickle.load(open(os.path.join(thick_path, fname), 'rb'))
        tmp_flake = tmp_flake['flakes']
        # print(tmp_flake)
        # print(len(tmp_flake))
        if len(tmp_flake) > 0:
            # print(os.path.join(data_path, fname[:-2] + 'tiff'))
            image = Image.open(os.path.join(data_path, fname[:-2] + 'tiff'))
            im_rgb = np.array(image).astype('float')
            imH, imW, _ = im_rgb.shape
            
            thick_flakes.extend(tmp_flake)
            for i in range(len(tmp_flake)):
                thick_names.append(fname+'-thick-'+str(i))
                f_mask_r_min, f_mask_r_max, f_mask_c_min, f_mask_c_max = tmp_flake[i]['flake_exact_bbox']
                f_mask_height = f_mask_r_max - f_mask_r_min
                f_mask_width = f_mask_c_max - f_mask_c_min
                flake_large_bbox = [max(0, f_mask_r_min - int(0.5 * f_mask_height)),
                                    min(imH, f_mask_r_max + int(0.5 * f_mask_height)),
                                    max(0, f_mask_c_min - int(0.5 * f_mask_width)),
                                    min(imW, f_mask_c_max + int(0.5 * f_mask_width))]
                tmp_flake[i]['flake_large_bbox'] = flake_large_bbox
                tmp_flake[i]['flake_img'] = im_rgb[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3], :].astype(np.uint8)

    num_thick = len(thick_flakes)
    print("number of thick: {}".format(num_thick))
    # load features
    for g_flake in thick_flakes:
        thick_feats.append(np.concatenate([g_flake['flake_shape_fea'],g_flake['flake_color_fea']]))
    thick_feats = np.stack(thick_feats)

    # load glue
    glue_path = '../results/data_sep2019_script/labelmat_glue/EXP1/09192019 Graphene'
    glue_flakes = []
    glue_feats = []
    glue_names = []
    fnames = os.listdir(glue_path)
    for fname in fnames:
        tmp_flake = pickle.load(open(os.path.join(glue_path, fname), 'rb'))
        tmp_flake = tmp_flake['flakes']
        # print(tmp_flake)
        # print(len(tmp_flake))
        if len(tmp_flake) > 0:
            # print(os.path.join(data_path, fname[:-2] + 'tiff'))
            image = Image.open(os.path.join(data_path, fname[:-2] + 'tiff'))
            im_rgb = np.array(image).astype('float')
            imH, imW, _ = im_rgb.shape
            
            glue_flakes.extend(tmp_flake)
            for i in range(len(tmp_flake)):
                glue_names.append(fname+'-glue-'+str(i))
                f_mask_r_min, f_mask_r_max, f_mask_c_min, f_mask_c_max = tmp_flake[i]['flake_exact_bbox']
                f_mask_height = f_mask_r_max - f_mask_r_min
                f_mask_width = f_mask_c_max - f_mask_c_min
                flake_large_bbox = [max(0, f_mask_r_min - int(0.5 * f_mask_height)),
                                    min(imH, f_mask_r_max + int(0.5 * f_mask_height)),
                                    max(0, f_mask_c_min - int(0.5 * f_mask_width)),
                                    min(imW, f_mask_c_max + int(0.5 * f_mask_width))]
                tmp_flake[i]['flake_large_bbox'] = flake_large_bbox
                tmp_flake[i]['flake_img'] = im_rgb[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3], :].astype(np.uint8)

    num_glue = len(glue_flakes)
    print("number of glue: {}".format(num_glue))
    # load features
    for g_flake in glue_flakes:
        glue_feats.append(np.concatenate([g_flake['flake_shape_fea'],g_flake['flake_color_fea']]))
    glue_feats = np.stack(glue_feats)
    
    # split train/val for graphene
    random.seed(0)
    graphene_idxs = np.arange(num_graphene)
    thick_idxs = np.arange(num_thick)
    glue_idxs = np.arange(num_glue)
    random.shuffle(graphene_idxs)
    random.shuffle(thick_idxs)
    random.shuffle(glue_idxs)
    train_ratio = 0.8
    tr_graphene_idxs = graphene_idxs[:int(train_ratio*num_graphene)]
    val_graphene_idxs = graphene_idxs[int(train_ratio*num_graphene):]
    tr_thick_idxs = thick_idxs[:int(train_ratio*num_thick)]
    val_thick_idxs = thick_idxs[int(train_ratio*num_thick):]
    tr_glue_idxs = glue_idxs[:int(train_ratio*num_glue)]
    val_glue_idxs = glue_idxs[int(train_ratio*num_glue):]


    train_flakes = train_flakes + [graphene_flakes[i] for i in tr_graphene_idxs] + [thick_flakes[i] for i in tr_thick_idxs] + [glue_flakes[i] for i in tr_glue_idxs]
    train_feats = np.concatenate([train_feats, graphene_feats[tr_graphene_idxs,:], thick_feats[tr_thick_idxs,:], glue_feats[tr_glue_idxs,:]])
    train_labels = train_labels + [3 for _ in range(len(tr_graphene_idxs))] + [0 for _ in range(len(tr_thick_idxs))] + [2 for _ in range(len(tr_glue_idxs))]
    train_names = train_names + [graphene_names[i] for i in tr_graphene_idxs] + [thick_names[i] for i in tr_thick_idxs] + [glue_names[i] for i in tr_glue_idxs]

    val_flakes = val_flakes + [graphene_flakes[i] for i in val_graphene_idxs] + [thick_flakes[i] for i in val_thick_idxs] + [glue_flakes[i] for i in val_glue_idxs]
    val_feats = np.concatenate([val_feats, graphene_feats[val_graphene_idxs, :], thick_feats[val_thick_idxs, :], glue_feats[val_glue_idxs, :]])
    val_labels = val_labels + [3 for _ in range(len(val_graphene_idxs))] + [0 for _ in range(len(val_thick_idxs))] + [2 for _ in range(len(val_glue_idxs))]
    val_names = val_names + [graphene_names[i] for i in val_graphene_idxs] + [thick_names[i] for i in val_thick_idxs] + [glue_names[i] for i in val_glue_idxs]


    # normalize data
    mean_feat = np.mean(train_feats, axis=0, keepdims=True)
    std_feat = np.std(train_feats, axis=0, keepdims=True)
    norm_fea = {}
    norm_fea['mean'] = mean_feat
    norm_fea['std'] = std_feat
    pickle.dump(norm_fea, open(os.path.join(clf_path, 'normfea.p'), 'wb'))
    train_feats -= mean_feat
    train_feats = train_feats / std_feat
    # train_feats = train_feats / np.linalg.norm(train_feats, 2, axis=1, keepdims=True)
    val_feats -= mean_feat
    val_feats = val_feats / std_feat
    # val_feats = val_feats / np.linalg.norm(val_feats, 2, axis=1, keepdims=True)

    # run classifier
    # method = 'linearsvm'
    # method = 'ridge'
    # method = 'rbfkernelsvm'
    method = hyperparams['clf_method']
    print(method)
    C = hyperparams['C']
    # C = 10
    Cs = [0.01, 0.1, 1, 10,]# 50, 100]
    for C in Cs:
        clf_save_path = os.path.join(save_path, 'feanorm_weighted_classifier-%s-%f.p'%(method, C))
        if os.path.exists(clf_save_path):
            clf = pickle.load(open(clf_save_path, 'rb'))
        else:
            if method == 'linearsvm':
                # clf = LinearSVC(random_state=0, tol=1e-5, C=C, max_iter=5e4, class_weight='balanced')
                clf = LinearSVC(random_state=0, tol=1e-5, C=C, max_iter=9e4, class_weight={0:1, 1:5, 2:1})#, multi_class='crammer_singer')
                clf.fit(train_feats, train_labels)
            elif method == 'ridge':
                clf = RidgeClassifier(random_state=0, alpha=C)
                clf.fit(train_feats, train_labels)
            elif method == 'rbfkernelsvm':
                clf = SVC(kernel='rbf', C=C, tol=1e-5, gamma=0.03)#, class_weight={0:1, 1:5, 2:1})
                clf.fit(train_feats, train_labels)
            elif method == 'polysvm':
                clf = SVC(kernel='poly', C=C, tol=1e-5, gamma='auto', class_weight={0:1, 1:5, 2:1})
                clf.fit(train_feats, train_labels)

            else:
                raise NotImplementedError

            pickle.dump(clf, open(clf_save_path, 'wb'))

        train_pred_cls = clf.predict(train_feats)
        train_pred_scores = clf.decision_function(train_feats)
        val_pred_cls = clf.predict(val_feats)
        val_pred_scores = clf.decision_function(val_feats)
        clf_vis_path = os.path.join(save_path, 'vis', 'feanorm_weighted_%s-%f'%(method, C))
        if not os.path.exists(clf_vis_path):
            os.makedirs(clf_vis_path)

        from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, precision_recall_curve
        train_acc = accuracy_score(train_labels, train_pred_cls)
        val_acc = accuracy_score(val_labels, val_pred_cls)
        train_conf = confusion_matrix(train_labels, train_pred_cls)
        val_conf = confusion_matrix(val_labels, val_pred_cls)
        print(train_conf)
        print(val_conf)
        val_conf = val_conf / np.sum(val_conf, 1, keepdims=True)
        train_conf = train_conf / np.sum(train_conf, 1, keepdims=True)
        # # val_acc = accuracy_score(, test_pred_cls)
        # print('train acc: %.4f' % (train_acc))
        print(train_conf)
        print(val_conf)
        vis_error(val_pred_cls, val_pred_scores, val_labels, val_flakes, clf_vis_path, val_names, 'val')
        vis_error(train_pred_cls, train_pred_scores, train_labels, train_flakes, clf_vis_path, train_names, 'train')

        # calculate map:
        uniquelabels = [0,1,2,3]
        train_aps = []
        val_aps = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        legends = ['thick', 'thin', 'glue', 'graphene']
        for l in uniquelabels:
            l_train_labels = [_ == l for _ in train_labels]
            l_val_labels = [_ == l for _ in val_labels]
            # if method == 'linearsvm':
            #     clf = LinearSVC(random_state=0, tol=1e-5, C=C)
            #     clf.fit(train_feats, l_train_labels)
            # elif method == 'ridge':
            #     clf = RidgeClassifier(random_state=0, alpha=C)
            #     clf.fit(train_feats, l_train_labels)
            # elif method == 'rbfkernelsvm':
            #     clf = SVC(kernel='rbf', C=C, tol=1e-5, gamma='auto')
            #     clf.fit(train_feats, l_train_labels)
            # else:
            #     raise NotImplementedError

            # l_train_pred_scores = clf.decision_function(train_feats)
            # l_val_pred_scores = clf.decision_function(val_feats)
            train_aps.append(average_precision_score(l_train_labels, train_pred_scores[:, l]))
            val_aps.append(average_precision_score(l_val_labels, val_pred_scores[:, l]))
            # print(val_pred_scores[:, l])
            # print(np.array(l_val_labels, dtype=np.uint8))
            precision_l, recall_l, _ = precision_recall_curve(np.array(l_val_labels, dtype=np.uint8), val_pred_scores[:, l])
            print(l, getPrecisionAtRecall(precision_l, recall_l, 0.90), getPrecisionAtRecall(precision_l, recall_l, 0.95) )
            ax.plot(recall_l, precision_l, label=legends[l])

        plt.legend()
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])

        plt.savefig(os.path.join(save_path, 'feanorm_weighted_%s-%f.png'%(method, C)), dpi=300)
        plt.close(fig)

        # train_labels = (np.array(train_labels) + 1)//2
        # val_labels = (np.array(val_labels) + 1 ) // 2
        # train_ap = average_precision_score(train_labels, train_pred_scores)
        # val_ap = average_precision_score(val_labels, pred_scores)
        print(train_aps)
        print(val_aps)
        print('%s-%f: train: %.4f, val: %.4f, ap train: %4f, ap val: %4f' %(method, C, train_acc, val_acc, np.mean(train_aps), np.mean(val_aps)))


def getPrecisionAtRecall(precision, recall, rate=0.95):
    # find the recall which is the first one that small or equal to rate.
    for id, r in enumerate(recall):
        if r <= rate:
            break
    return precision[id]


if __name__ == '__main__':
    main()