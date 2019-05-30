"""
Complete annotation
classify thick, thin, glue
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
labelmaps = {'thin': 1, 'thick': 0, 'glue': 2, 'mixed cluster': 3, 'others': 4}
labelmapsback = { 0: 'thick', 1: 'thin', 2: 'glue'}

hyperparams = { 'clf_method': 'linearsvm', # which classifier to use (linear): 'ridge', 'linearsvm', 'rbfkernelsvm', 'polysvm'
                'C': 5, # parameter to tune for SVM
                }


# read dataset
def readdb(dbname):
    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    # c.execute('PRAGMA TABLE_INFO({})'.format('annotab'))
    # info = c.fetchall()
    # col_dict = dict()
    # for col in info:
    #     col_dict[col[1]] = 0

    c.execute('SELECT oriname_newflakeid, indlabel, labelfromcluster FROM annotab')
    db = c.fetchall()
    num_empty = 0

    itemname_labels = []
    for i in range(len(db)):
        name_i = db[i][0]
        label_name_i = db[i][1]
        if label_name_i is not None:
            label_i = labelmaps[label_name_i]
            if label_i in [0, 1, 2]:
                itemname_labels.append([name_i, label_i])
        else:
            num_empty += 1

    print('empty annotation: %d' %(num_empty))
    return itemname_labels


# create db
def updatedb(all_dbname, thickthin_oriname_newflakeids):
    conn = sqlite3.connect(all_dbname)
    c = conn.cursor()

    # add the first round labeled thin/thick flake
    num_thickthin = len(thickthin_oriname_newflakeids)
    for i in range(num_thickthin):
        c.execute('''UPDATE annotab SET indlabel='%s' WHERE oriname_newflakeid='%s' ''' % (thickthin_oriname_newflakeids[i][1], thickthin_oriname_newflakeids[i][0]))
        conn.commit()
    print('label from thin/thick: %d' % (num_thickthin))


# load the incomplete thin/thick annotation
def readthickthindb(thickthin_dbname):
    conn = sqlite3.connect(thickthin_dbname)
    c = conn.cursor()
    c.execute('PRAGMA TABLE_INFO({})'.format('annotab'))
    info = c.fetchall()
    col_dict = dict()
    for col in info:
        col_dict[col[1]] = 0

    c.execute('SELECT * FROM annotab')
    db = c.fetchall()

    oriname_newflakeids = []
    for i in range(len(db)):
        oriname_newflakeid = db[i][0]
        ori_name, newflakeid = oriname_newflakeid.split('-')
        newflakeid = int(newflakeid)
        label = db[i][1]
        # if label == 'thin' or label == 'thick':
        oriname_newflakeids.append([oriname_newflakeid, label])

    return oriname_newflakeids

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

def locate_flakes(item_names, img_flakes, img_names):
    num = len(item_names)
    flakes = []
    features = []
    for i in range(num):
        f_name, f_id = item_names[i].split('-')
        f_id = int(f_id)
        idx = img_names.index(f_name)
        flakes.append(img_flakes[idx][f_id])
        features.append(np.concatenate([img_flakes[idx][f_id]['flake_shape_fea'],img_flakes[idx][f_id]['flake_color_fea']]))
    features = np.stack(features)

    return flakes, features


def vis_error(pred_cls, pred_scores, gt_cls, flakes, img_save_path, item_names, prefix):
    num = len(pred_cls)

    for i in range(num):
        if pred_cls[i] != gt_cls[i]:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            contours = flakes[i]['flake_contour_loc']
            contours[:,0] = contours[:,0] - flakes[i]['flake_large_bbox'][0]
            contours[:,1] = contours[:,1] - flakes[i]['flake_large_bbox'][2]
            contours = np.expand_dims(np.flip(contours), 1).astype(np.int32)
            if pred_cls[i] == 1:
                # thin, red
                contour_img = cv2.drawContours(flakes[i]['flake_img'], contours, -1, (255,0,0), 2)
            if pred_cls[i] == 0:
                # thick, green
                contour_img = cv2.drawContours(flakes[i]['flake_img'], contours, -1, (0,255,0), 2)
            if pred_cls[i] == 2:
                # glue, white
                contour_img = cv2.drawContours(flakes[i]['flake_img'], contours, -1, (255, 255, 255), 2)

            ax.imshow(contour_img)

            ax.axis('off')
            # print(prefix, gt_cls[i], pred_scores[i], item_names[i])
            # fig.savefig(os.path.join(img_save_path, '%s_gt_%d_pred_%.2f_%s.png'%(prefix, gt_cls[i], pred_scores[i], item_names[i] )))
            fig.savefig(os.path.join(img_save_path, '%s_gt_%s_pred_%s_%s.png'%(prefix, labelmapsback[gt_cls[i]], labelmapsback[pred_cls[i]], item_names[i] )))

            plt.close()


def main():
    subexp_name = 'YoungJaeShinSamples/4'
    anno_incomplete_file = '../data/data_jan2019_anno/anno_all_incomplete_YoungJaeShinSamples_4_useryoungjae.db'
    anno_complete_file = '../data/data_jan2019_anno/anno_all_YoungJaeShinSamples_4_useryoungjae.db'
    # anno_complete_file = '../data/data_jan2019_anno/anno_all_YoungJaeShinSamples_4_usertest0123.db'
    thickthin_anno_file = '../data/data_jan2019_anno/anno_thickthin_v2_YoungJaeShinSamples_4_useryoungjae.db'
    # thickthin_anno_file = '../data/data_jan2019_anno/anno_thickthin_v2_YoungJaeShinSamples_4_usertest0123.db'

    data_path = os.path.join('../data/data_jan2019', subexp_name)
    result_path = os.path.join('../results/data_jan2019_script/mat', subexp_name)
    clf_path = os.path.join('../results/data_jan2019_script/thickthinglue_clf_complete', subexp_name)
    if not os.path.exists(clf_path):
        os.makedirs(clf_path)

    # merge the annotation
    if not os.path.exists(anno_complete_file):
        os.system('cp %s %s' % (anno_incomplete_file, anno_complete_file))

        # update the annotation into the all annotation
        thickthin_oriname_newflakeids = readthickthindb(thickthin_anno_file)
        n_thickthin_flakes = len(thickthin_oriname_newflakeids)
        print(n_thickthin_flakes)

        updatedb(anno_complete_file, thickthin_oriname_newflakeids)


    # get the train/val split
    split_name = os.path.join(clf_path, 'train_val_split.p')
    if os.path.exists(split_name):
        to_load = pickle.load(open(split_name, 'rb'))
        train_names = to_load['train_names']
        train_labels = to_load['train_labels']
        val_names = to_load['val_names']
        val_labels = to_load['val_labels']
    else:
        itemname_labels = readdb(anno_complete_file)
        train_names, train_labels, val_names, val_labels = split_trainval(itemname_labels)
        to_save = dict()
        to_save['train_names'] = train_names
        to_save['train_labels'] = train_labels
        to_save['val_names'] = val_names
        to_save['val_labels'] = val_labels
        pickle.dump(to_save, open(split_name, 'wb'))

    # load flakes
    flake_save_name = os.path.join(clf_path, 'train_val_data.p')
    if os.path.exists(flake_save_name):
        to_load = pickle.load(open(flake_save_name, 'rb'))
        train_flakes = to_load['train_flakes']
        train_feats = to_load['train_feats']
        val_flakes = to_load['val_flakes']
        val_feats = to_load['val_feats']
    else:
        img_names = os.listdir(data_path)
        img_names.sort()
        img_flakes = Parallel(n_jobs=8)(delayed(load_one_image)(os.path.join(data_path, img_names[i]), os.path.join(result_path, img_names[i][:-4]+'.p'))
                                     for i in range(len(img_names)))
        # pickle.dump(img_flakes, open(flake_save_name, 'wb'))
        # load corresponding flakes
        train_flakes, train_feats = locate_flakes(train_names, img_flakes, img_names)
        val_flakes, val_feats = locate_flakes(val_names, img_flakes, img_names)
        to_save = dict()
        to_save['train_flakes'] = train_flakes
        to_save['train_feats'] = train_feats
        # to_save['train_names'] = train_names
        # to_save['train_labels'] = train_labels
        to_save['val_flakes'] = val_flakes
        to_save['val_feats'] = val_feats
        # to_save['val_names'] = val_names
        # to_save['val_labels'] = val_labels
        pickle.dump(to_save, open(flake_save_name, 'wb'))
    print('loading done')

# Hs = []
# Ws = []
# HoverWs = []
# for i in range(len(train_flakes)):
#     f_mask_r_min, f_mask_r_max, f_mask_c_min, f_mask_c_max = train_flakes[i]['flake_exact_bbox']
#     f_mask_height = f_mask_r_max - f_mask_r_min
#     f_mask_width = f_mask_c_max - f_mask_c_min
#     Hs.append(f_mask_height)
#     Ws.append(f_mask_width)
#     HoverWs.append(f_mask_height / f_mask_width)

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
    Cs = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 10, 50, 100]
    for C in Cs:
        clf_save_path = os.path.join(clf_path, 'feanorm_weighted_classifier-%s-%f.p'%(method, C))
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
        clf_vis_path = os.path.join(clf_path, subexp_name, 'vis', 'feanorm_weighted_%s-%f'%(method, C))
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
        uniquelabels = [0,1,2]
        train_aps = []
        val_aps = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        legends = ['thick', 'thin', 'glue']
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

        plt.savefig(os.path.join(clf_path, 'feanorm_weighted_%s-%f.png'%(method, C)), dpi=300)
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