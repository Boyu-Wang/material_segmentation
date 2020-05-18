"""
Graphene classification using clean annotated data from youngjae.
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
parser.add_argument('--color_fea', default='subsegment-contrast-bg-shape', type=str, help='which color feature to use: contrast, ori, both, contrast-bg, ori-bg, both-bg, contrast-bg-shape, innercontrast-bg-shape, subsegment-contrast-bg-shape')
# parser.add_argument('--exp_sid', default=0, type=int, help='exp start id')
# parser.add_argument('--exp_eid', default=1, type=int, help='exp end id')
# parser.add_argument('--subexp_sid', default=0, type=int, help='subexp start id')
# parser.add_argument('--subexp_eid', default=3, type=int, help='subexp end id')
parser.add_argument('--n_jobs', default=30, type=int, help='multiprocessing cores')

args = parser.parse_args()
print(args.color_fea)


# read from the annotation
def readdb(all_dbname):
    conn = sqlite3.connect(all_dbname)
    c = conn.cursor()
    c.execute('SELECT imflakeid, thicklabel FROM annotab')
    db = c.fetchall()

    num_graphene = 0
    num_others = 0
    oriname_flakeids = []
    for i in range(len(db)):
        imflakeid = db[i][0]
        flake_id = int(imflakeid.split('_')[3].split('-')[1])
        flake_oriname = imflakeid.split('_', 5)[5]
        label = db[i][1]
        if label == 'graphene':
            num_graphene += 1
            label_id = 1
        elif label == 'others':
            num_others += 1
            label_id = 0
        else:
            raise NotImplementedError

        oriname_flakeids.append([flake_oriname+'-'+str(flake_id), flake_oriname, flake_id, label_id])
        
    print('graphene: %d, others: %d'%(num_graphene, num_others))

    return oriname_flakeids


def load_one_image(color_fea, flake_path, fname, data_path, size_thre, annotated_names, annotated_labels):
    tmp_flake = pickle.load(open(os.path.join(flake_path, fname), 'rb'))
    image_labelmap = tmp_flake['image_labelmap']
    tmp_flake = tmp_flake['flakes']

    unlabeled_flakes = []
    labeled_feats = []
    unlabeled_feats = [] 
    labeled_flake_ids = []
    labeled_labels = []

    if len(tmp_flake) > 0:
        image = Image.open(os.path.join(data_path, fname[:-2] + 'tiff'))
        im_rgb = np.array(image).astype('float')
        im_hsv = color.rgb2hsv(im_rgb)
        im_hsv[:,:,2] = im_hsv[:,:,2]/255.0
        im_gray = color.rgb2gray(im_rgb)
        imH, imW, _ = im_rgb.shape

        for i in range(len(tmp_flake)):
            if tmp_flake[i]['flake_size'] > size_thre:
                flake_shape_fea = tmp_flake[i]['flake_shape_fea']
                if 'ori' in color_fea:
                    img_fea = tmp_flake[i]['flake_color_fea']
                elif 'innercontrast' in color_fea:
                    # include both inner and outer contrast features
                    # flatten the inner features
                    inner_fea = list(tmp_flake[i]['flake_innercontrast_color_fea'])
                    if isinstance(inner_fea[-1], np.ndarray):
                        inner_fea = inner_fea[:-1] + list(inner_fea[-1])
                    contrast_fea = list(tmp_flake[i]['flake_contrast_color_fea'])
                    if isinstance(contrast_fea[-1], np.ndarray):
                        contrast_fea = contrast_fea[:-1] + list(contrast_fea[-1])
                    img_fea = np.array(contrast_fea + inner_fea)
                elif 'contrast' in color_fea:
                    img_fea = tmp_flake[i]['flake_contrast_color_fea']
                elif 'both' in color_fea:
                    img_fea = np.concatenate([tmp_flake[i]['flake_color_fea'], tmp_flake[i]['flake_contrast_color_fea']])
                else:
                    img_fea = np.empty([0])
                    # raise NotImplementedError

                if 'subsegment' in color_fea:
                    img_fea = np.concatenate([img_fea, tmp_flake[i]['subsegment_features']])

                if 'bg' in color_fea:
                    img_fea = np.concatenate([img_fea, tmp_flake[i]['flake_bg_color_fea']])

                if 'shape' in color_fea:
                    img_fea = np.concatenate([img_fea, np.array([flake_shape_fea[0], flake_shape_fea[-1]])])

                # print(img_fea[15], type(img_fea[15]))
                # if isinstance(img_fea[15], np.ndarray):
                #     print(img_fea)
                #     print(data_path)
                    # raise NotImplementedError
                if fname[:-2] + 'tiff' + '-' + str(i) in annotated_names:
                    # get features only
                    labeled_feats.append(np.array(list(img_fea)))
                    labeled_flake_ids.append(i)
                    loc = annotated_names.index(fname[:-2] + 'tiff' + '-' + str(i))
                    labeled_labels.append(annotated_labels[loc])
                else:
                    # get original flake as well, for visualization.
                    unlabeled_feats.append(np.array(list(img_fea)))
                    # f_mask_r_min, f_mask_r_max, f_mask_c_min, f_mask_c_max = tmp_flake[i]['flake_exact_bbox']
                    # f_mask_height = f_mask_r_max - f_mask_r_min
                    # f_mask_width = f_mask_c_max - f_mask_c_min
                    # flake_large_bbox = [max(0, f_mask_r_min - int(1 * f_mask_height)),
                    #                     min(imH, f_mask_r_max + int(1 * f_mask_height)),
                    #                     max(0, f_mask_c_min - int(1 * f_mask_width)),
                    #                     min(imW, f_mask_c_max + int(1 * f_mask_width))]
                    # tmp_flake[i]['flake_large_bbox'] = flake_large_bbox
                    # tmp_flake[i]['flake_img'] = im_rgb[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3], :].astype(np.uint8)

                    # unlabeled_flakes.append(tmp_flake[i])

    # return labeled_feats, unlabeled_feats, unlabeled_flakes
    # print(labeled_feats)
    return labeled_feats, unlabeled_feats, labeled_flake_ids, labeled_labels


def load_one_dir(sub_anno_data, sub_data_path, sub_flake_path):
    annotated_names = [anno[0] for anno in sub_anno_data]
    annotated_labels = [anno[3] for anno in sub_anno_data]

    flake_names = os.listdir(sub_flake_path)
    flake_names = [n_i for n_i in flake_names if n_i[0]  not in ['.', '_']]
    flake_names.sort()

    size_thre = 100
    all_feats = Parallel(n_jobs=args.n_jobs)(delayed(load_one_image)(args.color_fea, sub_flake_path, flake_names[i], sub_data_path, size_thre, annotated_names, annotated_labels) for i in range(len(flake_names)))

    labeled_feats = [feat[0] for feat in all_feats]
    unlabeled_feats = [feat[1] for feat in all_feats]
    # unlabeled_flakes = [feat[2] for feat in all_feats]
    labeled_flake_ids = [feat[2] for feat in all_feats]
    labeled_labels = [feat[3] for feat in all_feats]

    labeled_flake_name_ids = []
    for i in range(len(labeled_flake_ids)):
        if len(labeled_flake_ids[i]) > 0:
            labeled_flake_name_ids.extend([flake_names[i] + '-' +str(x) for x in labeled_flake_ids[i]])

    labeled_feats = list(itertools.chain(*labeled_feats))
    unlabeled_feats = list(itertools.chain(*unlabeled_feats))
    # unlabeled_flakes = list(itertools.chain(*unlabeled_flakes))
    labeled_labels = list(itertools.chain(*labeled_labels))

    labeled_feats = np.stack(labeled_feats)
    unlabeled_feats = np.stack(unlabeled_feats)
    labeled_labels = np.array(labeled_labels)

    return labeled_feats, unlabeled_feats, labeled_flake_name_ids, labeled_labels


def getPrecisionAtRecall(precision, recall, rate=0.95):
    # find the recall which is the first one that small or equal to rate.
    for id, r in enumerate(recall):
        if r <= rate:
            break
    return precision[id]


def main():
    data_path = '../data/data_111x_individual/'
    result_path = '../results/data_111x_individual_script/mat'
    result_classify_path = '../results/data_111x_individual_script/graphene_classifier_with_clean_anno_colorfea-%s'%(args.color_fea)
    annotation_path = '../data/anno_graphene_youngjae'

    if not os.path.exists(result_classify_path):
        os.makedirs(result_classify_path)

    feat_save_path = os.path.join(result_classify_path, 'features_withname.p')
    separate_feat_save_path = os.path.join(result_classify_path, 'feature_separate.p')
    # unlabeled_flake_save_path = os.path.join(result_classify_path, 'unlabeled_flakes.p')
    if os.path.exists(feat_save_path):
        feats = pickle.load(open(feat_save_path, 'rb'))
        all_labeled_feats = feats['labeled_feats']
        all_unlabeled_feats = feats['unlabeled_feats']
        all_labels = feats['labeled_labels']
        # unlabeled_flakes = pickle.load(open(unlabeled_flake_save_path, 'rb'))
    else:

        # merge all annotation
        exp_names = os.listdir(data_path)
        exp_names = [ename for ename in exp_names if ename[0] not in ['.', '_']]
        exp_names = ['laminator', 'PDMS-QPress 6s']
        exp_names.sort()

        all_anno_data = []
        all_labeled_feats = []
        all_labels = []
        all_unlabeled_feats = []
        all_labeled_flake_name_ids = []
        for exp_name in exp_names:
            subexp_names = os.listdir(os.path.join(annotation_path, exp_name))
            subexp_names = [sname for sname in subexp_names if sname[0] not in ['.', '_']]
            subexp_names = [sname for sname in subexp_names if os.path.isdir(os.path.join(annotation_path, exp_name, sname))]
            subexp_names.sort()
            for sname in subexp_names:
                sub_anno_path = os.path.join(annotation_path, exp_name, sname, 'anno_user-youngjae.db')
                if not os.path.exists(sub_anno_path):
                    print('not exist annotation! ', sub_anno_path)

                sub_anno_data = readdb(sub_anno_path)
                all_anno_data.extend(sub_anno_data)
        
                # get the features
                labeled_feats, unlabeled_feats, labeled_flake_name_ids, labels = load_one_dir(sub_anno_data, os.path.join(data_path, exp_name, sname), os.path.join(result_path, exp_name, sname))
                all_labeled_feats.append(labeled_feats)
                all_unlabeled_feats.append(unlabeled_feats)
                labeled_flake_name_ids = [(exp_name, sname, labeled_flake_name_ids[i]) for i in range(len(labeled_flake_name_ids))]
                all_labeled_flake_name_ids.extend(labeled_flake_name_ids)
                # labels = np.array([anno[3] for anno in sub_anno_data])
                all_labels.append(labels)

        merge_anno_path = os.path.join(result_classify_path, 'merged_annotations.p')
        pickle.dump(all_anno_data, open(merge_anno_path, 'wb'))

        all_labeled_feats = np.concatenate(all_labeled_feats)
        all_unlabeled_feats = np.concatenate(all_unlabeled_feats)
        all_labels = np.concatenate(all_labels)

        feats ={}
        feats['labeled_feats'] = all_labeled_feats
        feats['unlabeled_feats'] = all_unlabeled_feats
        feats['labeled_labels'] = all_labels
        feats['labeled_flake_name_ids'] = all_labeled_flake_name_ids
    
        pickle.dump(feats, open(feat_save_path, 'wb'))

        num_labeled = len(all_labels)
        graphene_idxs = [l for l in range(num_labeled) if all_labels[l] ==1 ]
        others_idxs = [l for l in range(num_labeled) if all_labels[l] ==0 ]
        graphene_feats = all_labeled_feats[graphene_idxs]
        other_feats = all_labeled_feats[others_idxs]
        separate_feats = {}
        separate_feats['graphene_feats'] = graphene_feats
        separate_feats['other_feats'] = other_feats
        pickle.dump(separate_feats, open(separate_feat_save_path, 'wb'))

    print(all_labels)
    print(all_labeled_feats.shape)
    # print(all_labeled_feats)
    num_graphene = np.sum(all_labels == 1)
    num_others = np.sum(all_labels == 0)
    print('graphene: %d, others: %d'%(num_graphene, num_others))

    mean_feat = np.mean(all_labeled_feats, axis=0, keepdims=True)
    std_feat = np.std(all_labeled_feats, axis=0, keepdims=True)
    norm_fea = {}
    norm_fea['mean'] = mean_feat
    norm_fea['std'] = std_feat
    pickle.dump(norm_fea, open(os.path.join(result_classify_path, 'normfea.p'), 'wb'))
    all_labeled_feats -= mean_feat
    all_labeled_feats = all_labeled_feats / std_feat

    all_unlabeled_feats -= mean_feat
    all_unlabeled_feats = all_unlabeled_feats / std_feat

    # run classifier
    methods = ['linearsvm', 'poly']
    Cs = [0.01, 0.1, 1, 10, 50, 100]
    for method in methods:
        for C in Cs:
            clf_save_path = os.path.join(result_classify_path, 'feanorm_weighted_classifier-%s-%f.p'%(method, C))
            if os.path.exists(clf_save_path):
                clf = pickle.load(open(clf_save_path, 'rb'))
            else:
                if method == 'linearsvm':
                    clf = LinearSVC(random_state=0, tol=1e-5, C=C, max_iter=9e4, class_weight={0:1, 1:10})#, multi_class='crammer_singer')
                    clf.fit(all_labeled_feats, all_labels)
                elif method == 'poly':
                    clf = SVC(kernel='poly', gamma=2, degree=2, random_state=0, tol=1e-5, C=C, max_iter=9e4, class_weight={0:1, 1:10})#, multi_class='crammer_singer')
                    clf.fit(all_labeled_feats, all_labels)
                else:
                    raise NotImplementedError

                pickle.dump(clf, open(clf_save_path, 'wb'))

            train_pred_cls = clf.predict(all_labeled_feats)
            train_pred_scores = clf.decision_function(all_labeled_feats)
            # print(train_pred_scores.shape)
            
            from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, precision_recall_curve
            train_acc = accuracy_score(all_labels, train_pred_cls)
            train_conf = confusion_matrix(all_labels, train_pred_cls)
            print(train_conf)
            train_conf = train_conf / np.sum(train_conf, 1, keepdims=True)
            
            # calculate map:
            uniquelabels = [1]
            train_aps = []
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # legends = ['thick', 'thin', 'glue', 'graphene']
            legends = ['others', 'graphene']
            for l in uniquelabels:
                l_train_labels = [_ == l for _ in all_labels]
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

            plt.savefig(os.path.join(result_classify_path, 'feanorm_weighted_%s-%f.png'%(method, C)), dpi=300)
            plt.close(fig)

            # print(train_aps)
            print('%s-%f: train: %.4f, ap train: %4f' %(method, C, train_acc, np.mean(train_aps)))





if __name__ == '__main__':
    main()