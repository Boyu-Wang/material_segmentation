"""
Graphene classification using clean annotated data from youngjae.
"""

import numpy as np
from PIL import Image
import cv2
import argparse
import os
from skimage import io, color
import scipy
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from skimage.morphology import disk
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


parser = argparse.ArgumentParser(description='graphene classification')
parser.add_argument('--color_fea', default='threesub-contrast-bg-shape', type=str, help='which color feature to use: contrast, ori, both, contrast-bg, ori-bg, both-bg, contrast-bg-shape, innercontrast-bg-shape, subsegment-contrast-bg-shape, twosub-contrast-bg-shape, threesub-contrast-bg-shape, firstcluster-contrast-bg-shape')
# parser.add_argument('--exp_sid', default=0, type=int, help='exp start id')
# parser.add_argument('--exp_eid', default=1, type=int, help='exp end id')
# parser.add_argument('--subexp_sid', default=0, type=int, help='subexp start id')
# parser.add_argument('--subexp_eid', default=3, type=int, help='subexp end id')
parser.add_argument('--n_jobs', default=30, type=int, help='multiprocessing cores')
parser.add_argument('--multi_class', default=1, type=int, help='whether to use multi classification or binary classification')

args = parser.parse_args()
print(args.color_fea)

#{0: junk, 1: thick, 2: thin, 3: multi, 4: graphene}


def getPrecisionAtRecall(precision, recall, rate=0.95):
    # find the recall which is the first one that small or equal to rate.
    for id, r in enumerate(recall):
        if r <= rate:
            break
    return precision[id]


def main():
    # data_path = '../data/data_111x_individual/'
    # result_path = '../results/data_111x_individual_script/mat'
    feature_path = '../results/data_111x_individual_script/graphene_classifier_with_moreanno_v3_colorfea-%s'%(args.color_fea)
    result_classify_path = '../results/data_111x_individual_script/graphene_classifier_with_moreanno_v3_colorfea-%s_5fold_multiclass-%d'%(args.color_fea, args.multi_class)

    if not os.path.exists(result_classify_path):
        os.makedirs(result_classify_path)

    feat_save_path = os.path.join(feature_path, 'features_withname.p')
    separate_feat_save_path = os.path.join(feature_path, 'feature_separate.p')
    if os.path.exists(feat_save_path):
        feats = pickle.load(open(feat_save_path, 'rb'))
        all_labeled_feats = feats['labeled_feats']
        all_labeled_feats[np.isnan(all_labeled_feats)]= 0
        all_feats = all_labeled_feats
        # all_unlabeled_feats = feats['unlabeled_feats']
        all_labels = feats['labeled_labels']
    else:
        raise NotImplementedError
        
    num_labeled = len(all_labels)
    graphene_idxs = [l for l in range(num_labeled) if all_labels[l] == 4]
    junk_idxs = [l for l in range(num_labeled) if all_labels[l] == 0]
    thick_idxs = [l for l in range(num_labeled) if all_labels[l] == 1]
    thin_idxs = [l for l in range(num_labeled) if all_labels[l] == 2]
    multi_idxs = [l for l in range(num_labeled) if all_labels[l] == 3]
    # graphene_feats = all_labeled_feats[graphene_idxs]
    # junk_feats = all_labeled_feats[junk_idxs]
    # thick_feats = all_labeled_feats[thick_idxs]
    # thin_feats = all_labeled_feats[thin_idxs]
    # multi_feats = all_labeled_feats[multi_idxs]
    
    print(all_labeled_feats.shape)
    # print(all_labeled_feats)
    num_graphene = np.sum(all_labels == 4)
    num_junk = np.sum(all_labels == 0)
    num_thick = np.sum(all_labels == 1)
    num_thin = np.sum(all_labels == 2)
    num_multi = np.sum(all_labels == 3)
    print('junk: %d, thick: %d, thin: %d, multi: %d, graphene: %d'%(num_junk, num_thick, num_thin, num_multi, num_graphene))
    
    mean_feat = np.mean(all_labeled_feats, axis=0, keepdims=True)
    std_feat = np.std(all_labeled_feats, axis=0, keepdims=True)
    norm_fea = {}
    norm_fea['mean'] = mean_feat
    norm_fea['std'] = std_feat
    all_labeled_feats -= mean_feat
    all_labeled_feats = all_labeled_feats / std_feat

    
    # num_data = len(all_feats)
    # cross validation
    n_cross = 5
    shuffle_graphene_idxes = np.random.RandomState(seed=123).permutation(len(graphene_idxs))
    shuffle_junk_idxes = np.random.RandomState(seed=123).permutation(len(junk_idxs))
    shuffle_thick_idxes = np.random.RandomState(seed=123).permutation(len(thick_idxs))
    shuffle_thin_idxes = np.random.RandomState(seed=123).permutation(len(thin_idxs))
    shuffle_multi_idxes = np.random.RandomState(seed=123).permutation(len(multi_idxs))
    val_group_graphene_idxes = []
    val_group_junk_idxes = []
    val_group_thick_idxes = []
    val_group_thin_idxes = []
    val_group_multi_idxes = []
    for ni in range(n_cross):
        tmp_idx = [i*n_cross + ni for i in range(len(graphene_idxs) // n_cross+1) if i*n_cross + ni < len(graphene_idxs)]
        tmp_idx = [shuffle_graphene_idxes[i] for i in tmp_idx]
        val_group_graphene_idxes.append(tmp_idx)
        tmp_idx = [i*n_cross + ni for i in range(len(junk_idxs) // n_cross+1) if i*n_cross + ni < len(junk_idxs)]
        tmp_idx = [shuffle_junk_idxes[i] for i in tmp_idx]
        val_group_junk_idxes.append(tmp_idx)
        tmp_idx = [i*n_cross + ni for i in range(len(thick_idxs) // n_cross+1) if i*n_cross + ni < len(thick_idxs)]
        tmp_idx = [shuffle_thick_idxes[i] for i in tmp_idx]
        val_group_thick_idxes.append(tmp_idx)
        tmp_idx = [i*n_cross + ni for i in range(len(thin_idxs) // n_cross+1) if i*n_cross + ni < len(thin_idxs)]
        tmp_idx = [shuffle_thin_idxes[i] for i in tmp_idx]
        val_group_thin_idxes.append(tmp_idx)
        tmp_idx = [i*n_cross + ni for i in range(len(multi_idxs) // n_cross+1) if i*n_cross + ni < len(multi_idxs)]
        tmp_idx = [shuffle_multi_idxes[i] for i in tmp_idx]
        val_group_multi_idxes.append(tmp_idx)

    if args.multi_class == 1:
        num_class = 5
    else:
        num_class = 2
    methods = ['linearsvm', 'rbf']
    Cs = [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 10, 50]
    # Cs = [2]

    from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, precision_recall_curve
            
    for method in methods:
        for C in Cs:
            train_aps = []
            train_accs = []
            val_aps = []
            val_accs = []
            val_confs = np.zeros([num_class, num_class])
            val_pred_scores_all_fold = []
            l_val_labels_all_fold = []
            # split data
            test_clf_save_dir = os.path.join(result_classify_path, '%s-%f'%(method, C))
            if not os.path.exists(test_clf_save_dir):
                os.makedirs(test_clf_save_dir)

            for ni in range(n_cross):
                val_graphene_feats = np.array([all_feats[graphene_idxs[ij],:] for ij in val_group_graphene_idxes[ni]])
                tr_graphene_feats = np.array([all_feats[graphene_idxs[ij],:] for ij in range(len(graphene_idxs)) if ij not in val_group_graphene_idxes[ni]])
                val_junk_feats = np.array([all_feats[junk_idxs[ij],:] for ij in val_group_junk_idxes[ni]])
                tr_junk_feats = np.array([all_feats[junk_idxs[ij],:] for ij in range(len(junk_idxs)) if ij not in val_group_junk_idxes[ni]])
                
                val_thick_feats = np.array([all_feats[thick_idxs[ij],:] for ij in val_group_thick_idxes[ni]])
                tr_thick_feats = np.array([all_feats[thick_idxs[ij],:] for ij in range(len(thick_idxs)) if ij not in val_group_thick_idxes[ni]])
                val_thin_feats = np.array([all_feats[thin_idxs[ij],:] for ij in val_group_thin_idxes[ni]])
                tr_thin_feats = np.array([all_feats[thin_idxs[ij],:] for ij in range(len(thin_idxs)) if ij not in val_group_thin_idxes[ni]])
                val_multi_feats = np.array([all_feats[multi_idxs[ij],:] for ij in val_group_multi_idxes[ni]])
                tr_multi_feats = np.array([all_feats[multi_idxs[ij],:] for ij in range(len(multi_idxs)) if ij not in val_group_multi_idxes[ni]])
                
                val_graphene_names = ['graphene-%d.png'%(ij) for ij in val_group_graphene_idxes[ni]]
                val_junk_names = ['junk-%d.png'%(ij) for ij in val_group_junk_idxes[ni]]
                val_thick_names = ['thick-%d.png'%(ij) for ij in val_group_thick_idxes[ni]]
                val_thin_names = ['thin-%d.png'%(ij) for ij in val_group_thin_idxes[ni]]
                val_multi_names = ['multi-%d.png'%(ij) for ij in val_group_multi_idxes[ni]]
                val_names = val_graphene_names + val_junk_names + val_thick_names + val_thin_names + val_multi_names

                # print('train num of graphene: %d, number of others: %d'%(len(tr_graphene_feats), len(tr_others_feats)))
                # print('val num of graphene: %d, number of others: %d'%(len(val_graphene_feats), len(val_others_feats)))
                train_feats = np.concatenate([tr_graphene_feats, tr_junk_feats, tr_thick_feats, tr_thin_feats, tr_multi_feats])
                val_feats = np.concatenate([val_graphene_feats, val_junk_feats, val_thick_feats, val_thin_feats, val_multi_feats])
                
                if args.multi_class == 1:
                    train_labels = np.concatenate([4*np.ones([len(tr_graphene_feats)]), 0*np.ones([len(tr_junk_feats)]), 1*np.ones([len(tr_thick_feats)]), 2*np.ones([len(tr_thin_feats)]), 3*np.ones([len(tr_multi_feats)])])
                    val_labels = np.concatenate([4*np.ones([len(val_graphene_feats)]), 0*np.ones([len(val_junk_feats)]), 1*np.ones([len(val_thick_feats)]), 2*np.ones([len(val_thin_feats)]), 3*np.ones([len(val_multi_feats)])])
                    class_weight = {0: 1, 1: len(tr_junk_feats) / len(tr_thick_feats), 2: len(tr_junk_feats) / len(tr_thin_feats), 3: len(tr_junk_feats) / len(tr_multi_feats), 4: len(tr_junk_feats) / len(tr_graphene_feats)}
                else:
                    train_labels = np.concatenate([1*np.ones([len(tr_graphene_feats)]), 0*np.ones([len(tr_junk_feats)]), 0*np.ones([len(tr_thick_feats)]), 0*np.ones([len(tr_thin_feats)]), 0*np.ones([len(tr_multi_feats)])])
                    val_labels = np.concatenate([1*np.ones([len(val_graphene_feats)]), 0*np.ones([len(val_junk_feats)]), 0*np.ones([len(val_thick_feats)]), 0*np.ones([len(val_thin_feats)]), 0*np.ones([len(val_multi_feats)])])
                    class_weight = {0: 1, 1: len(tr_junk_feats) + len(tr_thick_feats) + len(tr_thin_feats) + len(tr_multi_feats) / len(tr_graphene_feats) }

                if method == 'linearsvm':
                    clf = LinearSVC(random_state=0, tol=1e-5, C=C, max_iter=5e4, class_weight=class_weight)
                elif method == 'rbf':
                    # compute gamma,
                    pair_dist = scipy.spatial.distance.pdist(train_feats, metric='euclidean')
                    gamma = 1.0 / np.mean(pair_dist)
                    clf = SVC(kernel='rbf', gamma=gamma, random_state=0, tol=1e-5, C=C, max_iter=5e4, class_weight=class_weight)
                else:
                    raise NotImplementedError
                clf.fit(train_feats, train_labels)

                train_pred_cls = clf.predict(train_feats)
                train_pred_scores = clf.decision_function(train_feats)
                val_pred_cls = clf.predict(val_feats)
                val_pred_scores = clf.decision_function(val_feats)

                # save images
                for imx in range(val_feats.shape[0]):
                    ori_name = os.path.join('../results/data_111x_individual_script/flake_vis_moreanno_v3', val_names[imx])
                    if args.multi_class == 1:
                        if val_pred_cls[imx] == 0:
                            pred_cls_name = 'junk'
                            pred_score = val_pred_scores[imx, 0]
                        elif val_pred_cls[imx] == 1:
                            pred_cls_name = 'thick'
                            pred_score = val_pred_scores[imx, 1]
                        elif val_pred_cls[imx] == 2:
                            pred_cls_name = 'thin'
                            pred_score = val_pred_scores[imx, 2]
                        elif val_pred_cls[imx] == 3:
                            pred_cls_name = 'multi'
                            pred_score = val_pred_scores[imx, 3]
                        elif val_pred_cls[imx] == 4:
                            pred_cls_name = 'graphene'
                            pred_score = val_pred_scores[imx, 4]
                    else:
                        pred_score = val_pred_scores[imx]
                        if val_pred_cls[imx] == 0:
                            pred_cls_name = 'others'
                        elif val_pred_cls[imx] == 1:
                            pred_cls_name = 'graphene'
                            
                    if val_pred_cls[imx] != val_labels[imx]:
                        new_name = os.path.join(, 'gt_' + val_names[imx] + '_predict_%s_score_%.3f.png'%(pred_cls_name, pred_score))
                        os.system('cp %s %s'%(ori_name, new_name))


                from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix
                train_acc = accuracy_score(train_labels, train_pred_cls)
                train_accs.append(train_acc)
                val_acc = accuracy_score(val_labels, val_pred_cls)
                val_accs.append(val_acc)
                # train_conf = confusion_matrix(train_labels, train_pred_cls)
                # train_conf = train_conf / np.sum(train_conf, 1, keepdims=True)
                val_conf = confusion_matrix(val_labels, val_pred_cls)
                # val_conf = val_conf / np.sum(val_conf, 1, keepdims=True)
                val_confs += val_conf

                # calculate map:
                if args.multi_class == 1:
                    l_train_labels = [_ == 4 for _ in train_labels]
                    l_val_labels = [_ == 4 for _ in val_labels]
                    train_aps.append(average_precision_score(l_train_labels, train_pred_scores[:,4]))
                    val_aps.append(average_precision_score(l_val_labels, val_pred_scores[:, 4]))
                    l_val_labels_all_fold.extend(l_val_labels)
                    val_pred_scores_all_fold.append(val_pred_scores)
                else:
                    l_train_labels = [_ == 1 for _ in train_labels]
                    l_val_labels = [_ == 1 for _ in val_labels]
                    train_aps.append(average_precision_score(l_train_labels, train_pred_scores))
                    val_aps.append(average_precision_score(l_val_labels, val_pred_scores))
                    l_val_labels_all_fold.extend(l_val_labels)
                    val_pred_scores_all_fold.append(val_pred_scores)

            # val_pred_scores_all_fold = np.concatenate(val_pred_scores_all_fold)
            # precision_l, recall_l, _ = precision_recall_curve(np.array(l_val_labels_all_fold, dtype=np.uint8), val_pred_scores_all_fold)
            # print(getPrecisionAtRecall(precision_l, recall_l, 0.90), getPrecisionAtRecall(precision_l, recall_l, 0.95) )
            
                # print(train_aps)
                # print(val_aps)
            print(val_confs)
            print('%s-%f: train: %.4f, val: %.4f, graphene ap train: %4f, graphene ap val: %4f' % (method, C, np.mean(train_accs), np.mean(val_accs), np.mean(train_aps), np.mean(val_aps)))



if __name__ == '__main__':
    main()