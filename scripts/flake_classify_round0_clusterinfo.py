"""
Train an classifier for each region (flake/glue), according to cluster annotation
And apply the classifier on the mixed cluster regions

By: Boyu Wang (boywang@cs.stonybrook.edu)
Created Data: 4 Apr 2019
Last Modified Date: 4 Apr 2019
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
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
# from multiprocessing import Pool

parser = argparse.ArgumentParser(description='flake segmentation')
parser.add_argument('--exp_sid', default=5, type=int, help='exp start id')
parser.add_argument('--exp_eid', default=6, type=int, help='exp end id')
parser.add_argument('--subexp_sid', default=0, type=int, help='subexp start id')
parser.add_argument('--subexp_eid', default=1, type=int, help='subexp end id')
# parser.add_argument('--img_sid', default=0, type=int)
# parser.add_argument('--img_eid', default=294, type=int)
parser.add_argument('--n_jobs', default=30, type=int, help='multiprocessing cores')
# parser.add_argument('--c_sid', default=0, type=int, help='subexp start id')
# parser.add_argument('--c_eid', default=400, type=int, help='subexp end id')

args = parser.parse_args()

labelmaps = {'thin': 0, 'thick': 0, 'glue': 1, 'mixed cluster': 2, 'others': 3}

hyperparams = { 'size_thre': 784, # after detect foreground regions, filter them based on its size. (784 corresponds to 5 um regions)
                'clf_method': 'rigde', # which classifier to use (linear): 'rigde', 'linearsvm'
                'cluster_fea': 'all', # what feature to use for clustering, could be: 'all', 'shape', 'color'
                'cluster_method': 'affinity', # which clustering method to use, could be: 'affinity', 'kmeans', 'meanshift'
                }


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
        if flake_size > hyperparams['size_thre']:
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


def load_one_cluster(cluster_name, cluster_save_path, all_img_names):
    cluster_id = int(cluster_name.split('-')[1])
    batch_id = int(cluster_name.split('.')[0].split('-')[2])
    num_row = 5
    flake_start_id = num_row * num_row * batch_id
    # cluster_save_path = os.path.join(cluster_save_path, str(cluster_id))
    names_path = os.path.join(cluster_save_path, str(cluster_id), 'names.txt')
    with open(names_path, 'r') as f:
        names = f.readlines()
    num_flakes = len(names)
    flake_end_id = min(num_row * num_row * (batch_id+1), num_flakes)

    ids = []
    for fid in range(flake_start_id, flake_end_id):
        # find the original image name, and flake id
        img_name = names[fid].split('-')[1]
        img_id = [ nn for nn in range(len(all_img_names)) if img_name in all_img_names[nn]]
        assert len(img_id) == 1
        img_id = img_id[0]
        flake_id = int(names[fid].split('-')[2].split('.')[0])
        ids.append([img_id, flake_id])

    return ids


# read the cluster level annotation file
def readdb(cluster_dbname):
    conn = sqlite3.connect(cluster_dbname)
    c = conn.cursor()
    c.execute('PRAGMA TABLE_INFO({})'.format('annotab'))
    info = c.fetchall()
    col_dict = dict()
    for col in info:
        col_dict[col[1]] = 0

    c.execute('SELECT * FROM annotab')
    db = c.fetchall()

    clustername_labels = []
    for i in range(len(db)):
        # print(db[i][1])
        # print(labelmaps[db[i][1]])
        clustername_labels.append([db[i][0], labelmaps[db[i][1]]])

    return clustername_labels


def map_flake_ids(img_flakes, cluster_ids):
    n_img = len(cluster_ids)
    for i in range(n_img):
        ori_id = cluster_ids[i][1]
        fks = img_flakes[cluster_ids[i][0]]
        for k in range(len(fks)):
            if ori_id == fks[k]['flake_id']:
                new_id = k
                print(ori_id, k, fks[k]['flake_id'])
                break
        cluster_ids[i][1] = new_id
        
    return cluster_ids


def load_cluster_fea(img_flakes, cluster_ids):
    n_img = len(cluster_ids)
    img_feas = []
    for i in range(n_img):
        # print(cluster_ids[i][0], cluster_ids[i][1])
        # print(len(img_flakes[cluster_ids[i][0]]))
        # print(img_flakes[cluster_ids[i][0]][cluster_ids[i][1]])
        img_feas.append(np.concatenate([ img_flakes[cluster_ids[i][0]][cluster_ids[i][1]]['flake_shape_fea'],
                                        img_flakes[cluster_ids[i][0]][cluster_ids[i][1]]['flake_color_fea']]))

    img_feas = np.array(img_feas)
    return img_feas


def test_one_cluster(cluster_ids, cluster_feas, mean_fea, classifier, img_flakes, new_cluster_save_path, cluster_imgname):
    n_img = len(cluster_feas)

    # normalize feature (subtract mean, divide by l2 norm)
    cluster_feas = cluster_feas - mean_fea
    cluster_feas = cluster_feas / np.linalg.norm(cluster_feas, axis=1, keepdims=True)
    cluster_feas = np.concatenate([cluster_feas, np.ones([cluster_feas.shape[0], 1])], axis=1)

    p_val = classifier.decision_function(cluster_feas)
    # p_val = p_val > 0

    flakes = []
    for i in range(n_img):
        flakes.append(img_flakes[cluster_ids[i][0]][cluster_ids[i][1]])

    # plot this cluster
    num_row = 5
    fig = plt.figure()
    cnt = 0
    for i in range(n_img):
        if i % (num_row*num_row) == 0:
            fig.clear()
            # print(i)
        ax = fig.add_subplot(num_row, num_row, i %(num_row*num_row) + 1)
        # # also plot contour
        contours = flakes[i]['flake_contour_loc']
        contours[:,0] = contours[:,0] - flakes[i]['flake_large_bbox'][0]
        contours[:,1] = contours[:,1] - flakes[i]['flake_large_bbox'][2]
        contours = np.expand_dims(np.flip(contours), 1).astype(np.int32)
        if p_val[i] > 0:
            # glue, red
            contour_img = cv2.drawContours(flakes[i]['flake_img'], contours, -1, (255,0,0), 2)
        if p_val[i] <= 0:
            # flake, green
            contour_img = cv2.drawContours(flakes[i]['flake_img'], contours, -1, (0,255,0), 2)
        ax.imshow(contour_img)
        # ax.imshow(flakes[i]['flake_img'])

        # print(flakes[i]['flake_img'].shape)
        # print(flakes[i]['flake_bbox'])
        ax.axis('off')
        if (i+1) % (num_row*num_row) == 0 or i == n_img-1:
            fig.savefig(os.path.join(new_cluster_save_path, cluster_imgname))
            cnt += 1

    plt.close()


def test_one_image(classifier, img_flakes, mean_fea, ori_img_path, ori_img_name, new_img_save_path):
    image = Image.open(os.path.join(ori_img_path, ori_img_name))

    im_gray = np.array(image.convert('L', (0.2989, 0.5870, 0.1140, 0))).astype('float')
    imH, imW = im_gray.shape
    # im_hsv = np.array(image.convert('HSV')).astype('float')
    im_rgb = np.array(image).astype(np.uint8)

    n_img = len(img_flakes)
    # get features
    img_feas = []
    for i in range(n_img):
        img_feas.append(np.concatenate([ img_flakes[i]['flake_shape_fea'],
                                            img_flakes[i]['flake_color_fea']]))

    img_feas = np.array(img_feas)

    # normalize feature (subtract mean, divide by l2 norm)
    img_feas = img_feas - mean_fea
    img_feas = img_feas / np.linalg.norm(img_feas, axis=1, keepdims=True)
    img_feas = np.concatenate([img_feas, np.ones([img_feas.shape[0], 1])], axis=1)

    p_val = classifier.decision_function(img_feas)

    fig = plt.figure()
    fig.clear()
    ax = fig.add_subplot(1, 1, 1)

    contour_img = im_rgb
    for i in range(n_img):
        contours = img_flakes[i]['flake_contour_loc']
        contours[:,0] = contours[:,0]
        contours[:,1] = contours[:,1]
        contours = np.expand_dims(np.flip(contours), 1).astype(np.int32)
        if p_val[i] > 0:
            # glue red
            contour_img = cv2.drawContours(contour_img, contours, -1, (255,0,0), 2)
        else:
            # flake, green
            contour_img = cv2.drawContours(contour_img, contours, -1, (0,255,0), 2)
        # contour_img = cv2.drawContours(im_rgb, contours, -1, (255,0,0), 2)
    ax.imshow(contour_img)
    ax.axis('off')
    fig.savefig(os.path.join(new_img_save_path, '%s.png'%(ori_img_name)))

        # # also show the location of the flake in ori image
        # # fig.clear()
        # # ax = fig.add_subplot(num_row, num_row, i %(num_row*num_row) + 1)
        # flake_large_bbox = flakes[i]['flake_large_bbox']
        # ori_img = Image.open(flakes[i]['img_name'])
        # ori_img = np.array(ori_img).astype(np.uint8)
        # # imH, imW, _ = ori_img.shape
        # # print(ori_img.shape, flake_large_bbox[2]-1, flake_large_bbox[0], flake_large_bbox[3]-1, flake_large_bbox[1])
        # bbox_img = cv2.rectangle(ori_img, (flake_large_bbox[2]-1, flake_large_bbox[0]), (flake_large_bbox[3]-1, flake_large_bbox[1]),(255,0,0),2)
        # cv2.imwrite(os.path.join(cluster_save_path, str(cluster_id), 'flake%04d-%s-%d_bbox.png'%(i, ori_name, ori_flakeid)), cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR))
        # # ax.imshow(bbox_img)
        # # ax.axis('off')
        # # fig.savefig(os.path.join(cluster_save_path, str(cluster_id), 'flake%04d-%s-%d_bbox.png'%(i, ori_name, ori_flakeid)))

    plt.close()
        # gc.collect()


# process one sub exp, read all the data, and do clustering
def classify_one_subexp(anno_file, subexp_dir, rslt_dir, flake_save_path, cluster_save_path, new_cluster_save_path, new_img_save_path, method, online_save_path):
    img_names = os.listdir(subexp_dir)
    img_names.sort()
    print('process ' + subexp_dir)
    print('n images: %d'%(len(img_names)))

    if os.path.exists(flake_save_path + 'flakes.p'):
        img_flakes = pickle.load(open(flake_save_path + 'flakes.p', 'rb'))
    else:
        img_flakes = Parallel(n_jobs=args.n_jobs)(delayed(load_one_image)(os.path.join(subexp_dir, img_names[i]), os.path.join(rslt_dir, img_names[i][:-4]+'.p'))
                                     for i in range(len(img_names)))
        pickle.dump(img_flakes, open(flake_save_path + 'flakes.p', 'wb'))
    print('loading done')

    clustername_labels = readdb(anno_file)
    n_total_clusters = len(clustername_labels)

    if os.path.exists(flake_save_path + 'imgflake_ids.p'):
        cluster_imgflake_ids = pickle.load(open(flake_save_path + 'imgflake_ids.p', 'rb'))
    else:
        cluster_imgflake_ids = Parallel(n_jobs=args.n_jobs)(delayed(load_one_cluster)(clustername_labels[i][0], cluster_save_path, img_names) for i in range(len(clustername_labels)))
        cluster_imgflake_ids = Parallel(n_jobs=args.n_jobs)(delayed(map_flake_ids)(img_flakes, cluster_imgflake_ids[i]) for i in range(n_total_clusters))
        pickle.dump(cluster_imgflake_ids, open(flake_save_path + 'imgflake_ids.p', 'wb'))
    print('id loading done')

    cluster_nflakes = [len(cluster_imgflake_ids[i]) for i in range(n_total_clusters)]
    cluster_labels = [clustername_labels[i][1] for i in range(n_total_clusters)]
    
    if os.path.exists(flake_save_path + 'feats.p'):
        cluster_feas = pickle.load(open(flake_save_path + 'feats.p', 'rb'))
    else:
        cluster_feas = Parallel(n_jobs=args.n_jobs)(delayed(load_cluster_fea)(img_flakes, cluster_imgflake_ids[i]) for i in range(n_total_clusters))
        pickle.dump(cluster_feas, open(flake_save_path + 'feats.p', 'wb'))
    print('fea loading done')

    cluster_train_ids = [i for i in range(n_total_clusters) if cluster_labels[i]==0 or cluster_labels[i]==1]
    cluster_test_ids = [i for i in range(n_total_clusters) if cluster_labels[i]==2]

    cluster_train_feas = [cluster_feas[i] for i in range(n_total_clusters) if i in cluster_train_ids]
    # size: [num_example, fea_dim]
    cluster_train_feas = np.concatenate(cluster_train_feas, axis=0)
    
    cluster_train_labels = [cluster_labels[i] * np.ones([cluster_nflakes[i],1]) for i in range(n_total_clusters) if i in cluster_train_ids]
    cluster_train_labels = np.concatenate(cluster_train_labels, axis=0)
    cluster_train_labels = cluster_train_labels[:, 0]

    cluster_train_labels = cluster_train_labels * 2 - 1
    cluster_train_labels = cluster_train_labels.astype('int')

    # normalize feature (subtract mean, divide by l2 norm)
    if os.path.exists(flake_save_path + 'meanfea.p'):
        mean_fea = pickle.load(open(flake_save_path + 'meanfea.p', 'rb'))
    else:
        mean_fea = np.mean(cluster_train_feas, axis=0, keepdims=True)
        pickle.dump(mean_fea, open(flake_save_path + 'meanfea.p', 'wb'))
    cluster_train_feas = cluster_train_feas - mean_fea
    cluster_train_feas = cluster_train_feas / np.linalg.norm(cluster_train_feas, axis=1, keepdims=True)
    # size: [num_example, fea_dim + 1]
    cluster_train_feas = np.concatenate([cluster_train_feas, np.ones([cluster_train_feas.shape[0], 1])], axis=1)


    if not os.path.exists(os.path.join(online_save_path, 'all_test_feas.p')):
        all_test_feas = []
        all_test_imgflake = []
        for c in cluster_test_ids:
            img_nflake = len(img_flakes[c])
            for i in range(img_nflake):
                all_test_feas.append(np.concatenate([ img_flakes[c][i]['flake_shape_fea'],
                                                img_flakes[c][i]['flake_color_fea']]))
                all_test_imgflake.append('%s-%d'%(img_names[c], i))
        all_test_feas = np.array(all_test_feas)
        # normalize feature (subtract mean, divide by l2 norm)
        all_test_feas = all_test_feas - mean_fea
        all_test_feas = all_test_feas / np.linalg.norm(all_test_feas, axis=1, keepdims=True)
        all_test_feas = np.concatenate([all_test_feas, np.ones([all_test_feas.shape[0], 1])], axis=1)

        to_save = dict()
        for i in range(len(all_test_feas)):
            to_save[all_test_imgflake[i]] = all_test_feas[i]
        pickle.dump(to_save, open(os.path.join(online_save_path, 'all_test_feas.p'), 'wb'))#,  protocol=0)
        with open(os.path.join(online_save_path, 'all_test_names.txt'), 'w') as f:
            f.write('\n'.join(all_test_imgflake))


    # save classifier:
    if os.path.exists(flake_save_path + 'classifier-%s.p'%(method)):
        clf = pickle.load(open(flake_save_path + 'classifier-%s.p'%(method), 'rb'))['clf']
    else:
        if method == 'linearsvm':
            clf = LinearSVC(random_state=0, tol=1e-5, fit_intercept=False)
            clf.fit(cluster_train_feas, cluster_train_labels)
        elif method == 'ridge':
            clf = RidgeClassifier(random_state=0, alpha=1, fit_intercept=False)
            clf.fit(cluster_train_feas, cluster_train_labels)
        else:
            raise NotImplementedError

        # pickle.dump(clf, open(flake_save_path + 'classifier-%s.p'%(method), 'wb'))
        Cn = np.matmul(np.transpose(cluster_train_feas), cluster_train_feas) + 1.0 * np.eye(cluster_train_feas.shape[1])
        Cn_inv = np.linalg.inv(Cn)
        XiYi = np.sum(cluster_train_feas * np.expand_dims(cluster_train_labels, 1), axis=0)
        to_save = dict()
        to_save['clf'] = clf
        to_save['coef'] = clf.coef_.transpose()
        to_save['Cn_inv'] = Cn_inv

        pickle.dump(to_save, open(flake_save_path + 'classifier-%s.p'%(method), 'wb'))
        pickle.dump(to_save, open(os.path.join(online_save_path, 'classifier-%s.p'%(method)), 'wb'))#,  protocol=0)
        pickle.dump(to_save, open(os.path.join(online_save_path, 'classifier-%s_b--1.p'%(method)), 'wb'))#,  protocol=0)

        # print(clf.coef_)
        print(np.sum(np.abs(np.matmul(Cn_inv, XiYi) - clf.coef_)))
    # test on each cluster

    # Parallel(n_jobs=args.n_jobs)(delayed(test_one_cluster)(cluster_imgflake_ids[c], cluster_feas[c], clf, img_flakes, new_cluster_save_path, clustername_labels[c][0])
    #                             for c in cluster_test_ids)

    # for c in cluster_test_ids:
    #     test_one_cluster(cluster_imgflake_ids[c], cluster_feas[c], mean_fea, clf, img_flakes, new_cluster_save_path, clustername_labels[c][0])
    

    # for i in range(len(img_names)):
    #     test_one_image(clf, img_flakes[i], mean_fea, subexp_dir, img_names[i], new_img_save_path)

def main():
    data_path = '../data/data_jan2019'
    result_path = '../results/data_jan2019_script/mat'
    # fig_path = '../results/data_jan2019_script/fig'
    anno_file = '../data/data_jan2019_anno/anno_cluster_YoungJaeShinSamples_4_useryoungjae.db'
    cluster_path = '../results/data_jan2019_script/cluster_sort_%d'%(hyperparams['size_thre'])
    testclf_cluster_path = '../results/data_jan2019_script/cluster_sort_%d_testclf0_cluster'%(hyperparams['size_thre'])
    online_path = '../results/data_jan2019_script/cluster_sort_%d_online'%(hyperparams['size_thre'])
    testclf_img_path = '../results/data_jan2019_script/cluster_sort_%d_testclf0_oriimg'%(hyperparams['size_thre'])

    exp_names = os.listdir(data_path)
    exp_names.sort()
    exp_names = [ename for ename in exp_names if ename[0] != '.']

    # print(exp_names)
    # exp_names = exp_names[args.exp_sid: args.exp_eid]

    # method = 'linearsvm'
    # method = 'ridge'
    method = hyperparams['clf_method']

    for d in range(args.exp_sid, args.exp_eid):
        exp_name = exp_names[d]
        subexp_names = os.listdir(os.path.join(data_path, exp_name))
        subexp_names = [sname for sname in subexp_names if os.path.isdir(os.path.join(data_path, exp_name, sname))]
        subexp_names.sort()
        # print(subexp_names)

        # process each subexp
        for s_d in range(args.subexp_sid, min(len(subexp_names), args.subexp_eid)):
            sname = subexp_names[s_d]
            # img_names = os.listdir(os.path.join(data_path, exp_name, sname))
            # img_names.sort()
            # img_names = img_names[args.img_sid:args.img_eid]
            
            flake_save_path = os.path.join(testclf_cluster_path, exp_name+sname)
            cluster_save_path = os.path.join(cluster_path, exp_name, sname, '%s_%s'(hyperparams['cluster_fea'], hyperparams['cluster_method']))
            testclf_cluster_save_path = os.path.join(testclf_cluster_path, exp_name, sname + '_' + method)
            online_save_path = os.path.join(online_path, exp_name, sname + '_' + method)
            testclf_img_save_path = os.path.join(testclf_img_path, exp_name, sname + '_' + method)

            if not os.path.exists(testclf_cluster_save_path):
                os.makedirs(testclf_cluster_save_path)
            
            if not os.path.exists(testclf_img_save_path):
                os.makedirs(testclf_img_save_path)

            if not os.path.exists(online_save_path):
                os.makedirs(online_save_path)
            
            classify_one_subexp(anno_file, os.path.join(data_path, exp_name, sname), os.path.join(result_path, exp_name, sname), flake_save_path, cluster_save_path, testclf_cluster_save_path, testclf_img_save_path, method, online_save_path)


if __name__ == '__main__':
    main()