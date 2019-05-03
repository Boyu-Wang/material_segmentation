"""
Test the classifier (flake/glue) on other set of experiments

By: Boyu Wang (boywang@cs.stonybrook.edu)
Created Data: 4 Apr 2019
Last Modified Date: 18 Apr 2019
"""


import numpy as np
from PIL import Image
import cv2
import argparse
import os
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from skimage.morphology import disk
import pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, AffinityPropagation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool
from joblib import Parallel, delayed
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier

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

hyperparams = { 'size_thre': 784, # after detect foreground regions, filter them based on its size. (784=28*28 corresponds to around 5 um regions)
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


# def readdb(dbname):
    # conn = sqlite3.connect('anno_userboyu.db')
    # c = conn.cursor()
    # c.execute('PRAGMA TABLE_INFO({})'.format('annotab'))
    # info = c.fetchall()
    # col_dict = dict()
    # for col in info:
    #     col_dict[col[1]] = 0

    # c.execute('SELECT * FROM annotab')
    # db = c.fetchall()

    # clustername_labels = []
    # for i in range(len(db)):
    #     # print(db[i][1])
    #     # print(labelmaps[db[i][1]])
    #     clustername_labels.append([db[i][0], labelmaps[db[i][1]]])

    # return clustername_labels


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

    # print(ori_img_name)
    # print(img_feas.shape)
    # print(mean_fea.shape)
    if img_feas.shape[0] == 0:
        return
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
def classify_one_subexp(subexp_dir, rslt_dir, flake_save_path, cluster_save_path, new_cluster_save_path, new_img_save_path, method, clf_save_path, mean_fea_save_path):
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

    clustername_labels = os.listdir(cluster_save_path)
    clustername_labels = [xx for xx in clustername_labels if '.png' in xx]
    
    n_total_clusters = len(clustername_labels)

    if os.path.exists(flake_save_path + 'imgflake_ids.p'):
        cluster_imgflake_ids = pickle.load(open(flake_save_path + 'imgflake_ids.p', 'rb'))
    else:
        cluster_imgflake_ids = Parallel(n_jobs=args.n_jobs)(delayed(load_one_cluster)(clustername_labels[i], cluster_save_path, img_names) for i in range(len(clustername_labels)))
        cluster_imgflake_ids = Parallel(n_jobs=args.n_jobs)(delayed(map_flake_ids)(img_flakes, cluster_imgflake_ids[i]) for i in range(n_total_clusters))
        pickle.dump(cluster_imgflake_ids, open(flake_save_path + 'imgflake_ids.p', 'wb'))
    print('id loading done')

    cluster_nflakes = [len(cluster_imgflake_ids[i]) for i in range(n_total_clusters)]
    # cluster_labels = [clustername_labels[i][1] for i in range(n_total_clusters)]
    
    if os.path.exists(flake_save_path + 'feats.p'):
        cluster_feas = pickle.load(open(flake_save_path + 'feats.p', 'rb'))
    else:
        cluster_feas = Parallel(n_jobs=args.n_jobs)(delayed(load_cluster_fea)(img_flakes, cluster_imgflake_ids[i]) for i in range(n_total_clusters))
        pickle.dump(cluster_feas, open(flake_save_path + 'feats.p', 'wb'))
    print('fea loading done')

    # cluster_train_ids = [i for i in range(n_total_clusters) if cluster_labels[i]==0 or cluster_labels[i]==1]
    # cluster_test_ids = [i for i in range(n_total_clusters) if cluster_labels[i]==2]
    cluster_test_ids = [i for i in range(n_total_clusters)]

    # normalize feature (subtract mean, divide by l2 norm)
    # print('/nfs/bigmind/add_disk0/boyu/Projects/material_segmentation/results/data_jan2019_script/cluster_sort_784_clf/YoungJaeShinSamples4meanfea.p')
    if os.path.exists(mean_fea_save_path):
        mean_fea = pickle.load(open(mean_fea_save_path, 'rb'))
    else:
        raise NotImplementedError

    # save classifier:
    if os.path.exists(clf_save_path):
        clf = pickle.load(open(clf_save_path, 'rb'))['clf']
    else:
        raise NotImplementedError

    for c in cluster_test_ids:
        test_one_cluster(cluster_imgflake_ids[c], cluster_feas[c], mean_fea, clf, img_flakes, new_cluster_save_path, clustername_labels[c])
    

    for i in range(len(img_names)):
        test_one_image(clf, img_flakes[i], mean_fea, subexp_dir, img_names[i], new_img_save_path)

def main():
    method = hyperparams['clf_method']

    data_path = '../data/data_jan2019'
    result_path = '../results/data_jan2019_script/mat'
    # fig_path = '../results/data_jan2019_script/fig'
    cluster_path = '../results/data_jan2019_script/cluster_sort_784'
    new_cluster_path = '../results/data_jan2019_script/cluster_sort_784_clf'
    new_img_path = '../results/data_jan2019_script/cluster_sort_784_clf_ori'
    clf_save_path = '../results/data_jan2019_script/cluster_sort_784_clf/YoungJaeShinSamples4classifier-%s.p'%(method)
    mean_fea_save_path = '../results/data_jan2019_script/cluster_sort_784_clf/YoungJaeShinSamples4meanfea.p'

    exp_names = os.listdir(data_path)
    exp_names.sort()
    exp_names = [ename for ename in exp_names if ename[0] != '.']

    # print(exp_names)
    # exp_names = exp_names[args.exp_sid: args.exp_eid]

    # method = 'linearsvm'
    # method = 'ridge'
    
    for d in range(args.exp_sid, args.exp_eid):
        exp_name = exp_names[d]
        subexp_names = os.listdir(os.path.join(data_path, exp_name))
        subexp_names = [sname for sname in subexp_names if os.path.isdir(os.path.join(data_path, exp_name, sname))]
        subexp_names.sort()
        # print(subexp_names)

        # process each subexp
        for s_d in range(args.subexp_sid, min(len(subexp_names), args.subexp_eid)):
            sname = subexp_names[s_d]
            
            flake_save_path = os.path.join(new_cluster_path, exp_name+sname)
            cluster_save_path = os.path.join(cluster_path, exp_name, sname, '%s_%s'%(hyperparams['cluster_fea'], hyperparams['cluster_method']))
            new_cluster_save_path = os.path.join(new_cluster_path, exp_name, sname + '_' + method)
            # online_save_path = os.path.join(online_path, exp_name, sname + '_' + method)
            new_img_save_path = os.path.join(new_img_path, exp_name, sname + '_' + method)

            if not os.path.exists(new_cluster_save_path):
                os.makedirs(new_cluster_save_path)
            
            if not os.path.exists(new_img_save_path):
                os.makedirs(new_img_save_path)

            classify_one_subexp(os.path.join(data_path, exp_name, sname), os.path.join(result_path, exp_name, sname), flake_save_path, cluster_save_path, new_cluster_save_path, new_img_save_path, method, clf_save_path, mean_fea_save_path)




if __name__ == '__main__':
    main()






