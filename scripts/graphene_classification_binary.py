"""
Visualize graphene and other flakes.
"""
import numpy as np
from PIL import Image
import cv2
import argparse
import os
from skimage import io, color
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from skimage.morphology import disk
import pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import itertools
import sklearn
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, AffinityPropagation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
import gc
import copy
import sqlite3
from multiprocessing import Pool
from joblib import Parallel, delayed
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import RidgeClassifier
import random
import robustfit


parser = argparse.ArgumentParser(description='graphene classification')
parser.add_argument('--color', default='contrast', type=str, help='which color feature to use: contrast, ori, both, contrast-bg, ori-bg, both-bg, contrast-bg-shape')
# parser.add_argument('--exp_sid', default=0, type=int, help='exp start id')
# parser.add_argument('--exp_eid', default=1, type=int, help='exp end id')
# parser.add_argument('--subexp_sid', default=0, type=int, help='subexp start id')
# parser.add_argument('--subexp_eid', default=3, type=int, help='subexp end id')
# parser.add_argument('--n_jobs', default=30, type=int, help='multiprocessing cores')

args = parser.parse_args()
print(args.color)

def load_one_image(args_color, flake_path, fname, data_path, size_thre):
    tmp_flake = pickle.load(open(os.path.join(flake_path, fname), 'rb'))
    image_labelmap = tmp_flake['image_labelmap']
    tmp_flake = tmp_flake['flakes']

    flakes = []
    feats = []
    if len(tmp_flake) > 0:
        image = Image.open(os.path.join(data_path, fname[:-2] + 'tiff'))
        im_rgb = np.array(image).astype('float')
        im_hsv = color.rgb2hsv(im_rgb)
        im_hsv[:,:,2] = im_hsv[:,:,2]/255.0
        im_gray = color.rgb2gray(im_rgb)
        imH, imW, _ = im_rgb.shape

        # background fitting
        if args_color == 'contrast' or args_color == 'both' or 'bg' in args_color:
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
        
        for i in range(len(tmp_flake)):
            if tmp_flake[i]['flake_size'] > size_thre:
                # names.append(fname+'-'+str(tmp_flake[i]['flake_id']))
                f_mask_r_min, f_mask_r_max, f_mask_c_min, f_mask_c_max = tmp_flake[i]['flake_exact_bbox']
                f_mask_height = f_mask_r_max - f_mask_r_min
                f_mask_width = f_mask_c_max - f_mask_c_min
                flake_large_bbox = [max(0, f_mask_r_min - int(0.5 * f_mask_height)),
                                    min(imH, f_mask_r_max + int(0.5 * f_mask_height)),
                                    max(0, f_mask_c_min - int(0.5 * f_mask_width)),
                                    min(imW, f_mask_c_max + int(0.5 * f_mask_width))]
                tmp_flake[i]['flake_large_bbox'] = flake_large_bbox
                tmp_flake[i]['flake_img'] = im_rgb[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3], :].astype(np.uint8)

                flakes.append(tmp_flake[i])

                tmp_fea_ori = tmp_flake[i]['flake_color_fea']
                bwmap = (image_labelmap == i+1).astype(np.uint8)

                if 'ori' in args_color:
                    tmp_color_fea = list(tmp_fea_ori)
                elif 'contrast' in args_color or 'both' in args_color:
                    # color fea
                    assert bwmap.sum() == tmp_flake[i]['flake_size']
                    contrast_gray = im_gray - bg_gray
                    contrast_hsv = im_hsv - bg_hsv
                    contrast_rgb = im_rgb - bg_rgb
                    flake_color_entropy = cv2.calcHist([contrast_gray[bwmap>0].astype('uint8')],[0],None,[256],[0,256])
                    flake_color_entropy = entropy(flake_color_entropy, base=2)
                    # gray, h, gray std, hsv mean, hsv std, rgb mean, rgb std, gray entropy
                    tmp_fea_contrast = [contrast_gray[bwmap>0].mean(), 
                                     contrast_hsv[bwmap>0, 2].mean()] + \
                                     [contrast_gray[bwmap>0].std()] + \
                                     list(contrast_hsv[bwmap>0].mean(0)) + list(contrast_hsv[bwmap>0].std(0)) + \
                                     list(contrast_rgb[bwmap>0].mean(0)) + list(contrast_rgb[bwmap>0].std(0)) + [flake_color_entropy]

                    if 'contrast' in args_color:
                        tmp_color_fea = list(tmp_fea_contrast)
                    elif 'both' in args_color:
                        tmp_color_fea = list(tmp_fea_ori) + list(tmp_fea_contrast)
                else:
                    raise NotImplementedError

                if 'bg' in args_color:
                    tmp_bg_fea = [bg_gray[bwmap>0].mean()] + \
                         [bg_gray[bwmap>0].std()] + \
                         list(bg_hsv[bwmap>0].mean(0)) + list(bg_hsv[bwmap>0].std(0)) + \
                         list(bg_rgb[bwmap>0].mean(0)) + list(bg_rgb[bwmap>0].std(0))

                    tmp_color_fea = list(tmp_color_fea) + list(tmp_bg_fea)

                if 'shape' in args_color:
                    tmp_shape_fea = tmp_flake[i]['flake_shape_fea']
                    len_area_ratio = tmp_shape_fea[0]
                    fracdim = tmp_shape_fea[-1]
                    tmp_color_fea = list(tmp_color_fea) + [len_area_ratio, fracdim]

                feats.append(tmp_color_fea)

    return flakes, feats

def load_one_class(data_path, flake_path, size_thre=100):
    flakes = []
    feats = []
    # names = []

    fnames = os.listdir(flake_path)
    flake_feats = Parallel(n_jobs=20)(delayed(load_one_image)(args.color, flake_path, fname, data_path, size_thre) 
                        for fname in fnames)
    flakes = [flake_feat[0] for flake_feat in flake_feats]
    feats = [flake_feat[1] for flake_feat in flake_feats]
    flakes = list(itertools.chain.from_iterable(flakes))
    feats = list(itertools.chain.from_iterable(feats))
    feats = np.stack(feats)

    assert len(flakes) == feats.shape[0]
    return feats

def load_data(data_path, graphene_path, thick_path, glue_path):
    size_thre = 100

    graphene_feats = load_one_class(data_path, graphene_path, size_thre)

    num_graphene = graphene_feats.shape[0]
    print("number of graphene: {}".format(num_graphene))
    
    # load thick
    thick_feats = load_one_class(data_path, thick_path, size_thre)
    glue_feats = load_one_class(data_path, glue_path, size_thre)

    other_feats = np.concatenate([thick_feats, glue_feats])

    num_other = other_feats.shape[0]
    print("number of others: {}".format(num_other))

    return graphene_feats, other_feats

def getPrecisionAtRecall(precision, recall, rate=0.95):
    # find the recall which is the first one that small or equal to rate.
    for id, r in enumerate(recall):
        if r <= rate:
            break
    return precision[id]


def classify(graphene_feats, other_feats, classifier_save_path):
    train_feats = np.concatenate([graphene_feats, other_feats])
    train_labels = np.concatenate([np.ones(graphene_feats.shape[0]), np.zeros(other_feats.shape[0])])
    # normalize data
    mean_feat = np.mean(train_feats, axis=0, keepdims=True)
    std_feat = np.std(train_feats, axis=0, keepdims=True)
    norm_fea = {}
    norm_fea['mean'] = mean_feat
    norm_fea['std'] = std_feat
    pickle.dump(norm_fea, open(os.path.join(classifier_save_path, 'normfea.p'), 'wb'))
    train_feats -= mean_feat
    train_feats = train_feats / std_feat
    # train_feats = train_feats / np.linalg.norm(train_feats, 2, axis=1, keepdims=True)
    
    # run classifier
    methods = ['linearsvm', 'poly']
    Cs = [0.01, 0.1, 1, 10,]# 50, 100]
    for method in methods:
        for C in Cs:
            clf_save_path = os.path.join(classifier_save_path, 'feanorm_weighted_classifier-%s-%f.p'%(method, C))
            if os.path.exists(clf_save_path):
                clf = pickle.load(open(clf_save_path, 'rb'))
            else:
                if method == 'linearsvm':
                    # clf = LinearSVC(random_state=0, tol=1e-5, C=C, max_iter=5e4, class_weight='balanced')
                    # clf = LinearSVC(random_state=0, tol=1e-5, C=C, max_iter=9e4, class_weight={0:1, 1:5, 2:1})#, multi_class='crammer_singer')
                    clf = LinearSVC(random_state=0, tol=1e-5, C=C, max_iter=9e4, class_weight={0:1, 1:10})#, multi_class='crammer_singer')
                    clf.fit(train_feats, train_labels)
                elif method == 'poly':
                    clf = SVC(kernel='poly', gamma=2, degree=2, random_state=0, tol=1e-5, C=C, max_iter=9e4, class_weight={0:1, 1:10})#, multi_class='crammer_singer')
                    clf.fit(train_feats, train_labels)
                else:
                    raise NotImplementedError

                pickle.dump(clf, open(clf_save_path, 'wb'))

            train_pred_cls = clf.predict(train_feats)
            train_pred_scores = clf.decision_function(train_feats)
            print(train_pred_scores.shape)
            
            from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, precision_recall_curve
            train_acc = accuracy_score(train_labels, train_pred_cls)
            train_conf = confusion_matrix(train_labels, train_pred_cls)
            print(train_conf)
            train_conf = train_conf / np.sum(train_conf, 1, keepdims=True)
            # print(train_conf)
            
            # calculate map:
            uniquelabels = [1]
            train_aps = []
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # legends = ['thick', 'thin', 'glue', 'graphene']
            legends = ['others', 'graphene']
            for l in uniquelabels:
                l_train_labels = [_ == l for _ in train_labels]
                # train_aps.append(average_precision_score(l_train_labels, train_pred_scores[:, l]))
                train_aps.append(average_precision_score(l_train_labels, train_pred_scores))
                # precision_l, recall_l, _ = precision_recall_curve(np.array(l_train_labels, dtype=np.uint8), train_pred_scores[:, l])
                precision_l, recall_l, _ = precision_recall_curve(np.array(l_train_labels, dtype=np.uint8), train_pred_scores)
                print(l, getPrecisionAtRecall(precision_l, recall_l, 0.90), getPrecisionAtRecall(precision_l, recall_l, 0.95) )
                ax.plot(recall_l, precision_l, label=legends[l])

            plt.legend()
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])

            plt.savefig(os.path.join(classifier_save_path, 'feanorm_weighted_%s-%f.png'%(method, C)), dpi=300)
            plt.close(fig)

            # print(train_aps)
            print('%s-%f: train: %.4f, ap train: %4f' %(method, C, train_acc, np.mean(train_aps)))


def main():
    data_path = '../data/data_sep2019/EXP1/09192019 Graphene'
    graphene_path = '../results/data_sep2019_script/labelmat_graphene/EXP1/09192019 Graphene'
    thick_path = '../results/data_sep2019_script/labelmat_thick/EXP1/09192019 Graphene'
    glue_path = '../results/data_sep2019_script/labelmat_glue/EXP1/09192019 Graphene'

    classifier_save_path = '../results/data_sep2019_script/graphene_classifier_binary_fea-%s'%(args.color)
    if not os.path.exists(classifier_save_path):
        os.makedirs(classifier_save_path)

    feat_save_path = os.path.join(classifier_save_path, 'feature.p')
    if os.path.exists(feat_save_path):
        feats = pickle.load(open(feat_save_path, 'rb'))
        graphene_feats = feats['graphene_feats']
        other_feats = feats['other_feats']
    else:
        graphene_feats, other_feats = load_data(data_path, graphene_path, thick_path, glue_path)
        feats = {}
        feats['graphene_feats'] = graphene_feats
        feats['other_feats'] = other_feats
        pickle.dump(feats, open(feat_save_path, 'wb'))
    print("number of graphene: {}".format(graphene_feats.shape[0]))
    print("number of others: {}".format(other_feats.shape[0]))
    classify(graphene_feats, other_feats, classifier_save_path)

if __name__ == '__main__':
    main()