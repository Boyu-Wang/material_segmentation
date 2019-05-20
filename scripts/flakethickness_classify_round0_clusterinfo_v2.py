"""
Updates: remove repetitive labels
Train an classifier for each flake (thin/thick), according to cluster annotation

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

labelmaps = {'thin': 1, 'thick': 0, 'glue': 2, 'mixed cluster': 3, 'others': 4}
labelmapsback = {1: 'thin', 0: 'thick', 2: 'glue'}

hyperparams = { 'size_thre': 784, # after detect foreground regions, filter them based on its size. (784 corresponds to 5 um regions)
                'clf_method': 'linearsvm', # which classifier to use (linear): 'rigde', 'linearsvm'
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
    cnt = 0
    ori2newid = dict()
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
            ori2newid[i+1] = cnt
            # flakes[i]['new_id'] = cnt
            cnt += 1

    flakes = [flakes[j] for j in large_flake_idxs]
    return flakes, ori2newid


# create db
def createdb(all_dbname, img_names, ori2news, exclude_oriname_newflakeids, thickthin_oriname_newflakeids):
    conn = sqlite3.connect(all_dbname)
    c = conn.cursor()
    c.execute('ALTER TABLE annotab RENAME TO annotab2')
    c.execute('CREATE TABLE annotab(oriname_oriflakeid STRING PRIMARY KEY, labelfromcluster STRING, indlabel STRING, oriname_newflakeid INTEGER, clusterid INTEGER, withinclusterid INTEGER, flakecenterrow INTEGER, flakecentercolumn INTEGER, flakesize INTEGER)')
    c.execute('INSERT INTO annotab(oriname_oriflakeid, labelfromcluster, clusterid, withinclusterid, flakecenterrow, flakecentercolumn, flakesize) SELECT oriname_oriflakeid, thicklabel, clusterid, withinclusterid, flakecenterrow, flakecentercolumn, flakesize FROM annotab2;')
    c.execute('DROP TABLE annotab2')
    # c.execute('ALTER TABLE annotab ADD COLUMN indlabel STRING')
    conn.commit()

    # insert newflakeid
    # get the oriname_oriid and cluster label
    c.execute('SELECT oriname_oriflakeid, labelfromcluster FROM annotab')
    all_oriname_oriid_clusterlabel = c.fetchall()
    num_total = len(all_oriname_oriid_clusterlabel)
    print('total number of regions: %d' %(num_total))
    # if the cluster is 'others' or 'glue', 'thick', put the indlabel as cluster label
    for i in range(num_total):
        oriname, oriid = all_oriname_oriid_clusterlabel[i][0].split('-')
        oriid = int(oriid)
        name_id = img_names.index(oriname+'.tif')
        newid = ori2news[name_id][oriid]
        oriname_newid = oriname + '.tif-' + str(newid)
        c.execute('''UPDATE annotab SET oriname_newflakeid='%s' WHERE oriname_oriflakeid='%s' '''%(oriname_newid, all_oriname_oriid_clusterlabel[i][0]))
        conn.commit()
        if all_oriname_oriid_clusterlabel[i][1] in ['others', 'glue', 'thick'] :
        # if all_oriname_oriid_clusterlabel[i][1] in ['thick']:
            c.execute('''UPDATE annotab SET indlabel='%s' WHERE oriname_oriflakeid='%s' '''%(all_oriname_oriid_clusterlabel[i][1], all_oriname_oriid_clusterlabel[i][0]))
            conn.commit()

    # read label from glue/flake annotation, only save glue, other label
    num_glue = len(exclude_oriname_newflakeids)
    for i in range(num_glue):
        # print(exclude_oriname_newflakeids[i])
        c.execute('''UPDATE annotab SET indlabel='%s' WHERE oriname_newflakeid='%s' ''' % (exclude_oriname_newflakeids[i][2], exclude_oriname_newflakeids[i][0] + '-' + str(exclude_oriname_newflakeids[i][1])))
        conn.commit()
    print('label from glue (glue/flake): %d'%(num_glue))

    # add the first round labeled thin/thick flake
    num_thickthin = len(thickthin_oriname_newflakeids)
    for i in range(num_thickthin):
        c.execute('''UPDATE annotab SET indlabel='%s' WHERE oriname_newflakeid='%s' ''' % (thickthin_oriname_newflakeids[i][2], thickthin_oriname_newflakeids[i][0] + '-' + str(thickthin_oriname_newflakeids[i][1])))
        conn.commit()
    print('label from thin/thick: %d' % (num_thickthin))


# read from the annotation, find what needs to be labeled.
def readalldb(all_dbname):
    conn = sqlite3.connect(all_dbname)
    c = conn.cursor()
    c.execute('SELECT oriname_newflakeid, indlabel, labelfromcluster FROM annotab')
    db = c.fetchall()

    num_tolabel = 0
    num_glue = 0
    num_thin = 0
    num_thick = 0
    oriname_newflakeids = []
    for i in range(len(db)):
        oriname_newflakeid = db[i][0]
        ori_name, newflakeid = oriname_newflakeid.split('-')
        newflakeid = int(newflakeid)
        label = db[i][1]
        labelfromcluster = db[i][2]
        oriname_newflakeids.append([ori_name, newflakeid, label, labelfromcluster])
        if label is None:
            num_tolabel += 1
        elif label == 'glue':
            num_glue += 1
        elif label == 'thin':
            num_thin += 1
        elif label == 'thick':
            num_thick += 1
        # else:
        #     print(oriname_newflakeid, label)
        #     raise NotImplementedError

    print('tolabel: %d, glue: %d, thin: %d, thick: %d'%(num_tolabel, num_glue, num_thin, num_thick))

    return oriname_newflakeids


# only load the flake labels
# read from glue/flake annotation
def readflakegluedb(flakeglue_dbname):
    conn = sqlite3.connect(flakeglue_dbname)
    c = conn.cursor()
    c.execute('PRAGMA TABLE_INFO({})'.format('annotab'))
    info = c.fetchall()
    col_dict = dict()
    for col in info:
        col_dict[col[1]] = 0

    c.execute('SELECT * FROM annotab')
    db = c.fetchall()

    flake_oriname_newflakeids = []
    # these regions no need for labeling
    exclude_oriname_newflakeids = []
    for i in range(len(db)):
        # print(db[i][1])
        # print(labelmaps[db[i][1]])
        oriname_newflakeid = db[i][0]
        ori_name, newflakeid = oriname_newflakeid.split('-')
        newflakeid = int(newflakeid)
        label = db[i][1]
        if label == 'flake':
            flake_oriname_newflakeids.append([ori_name, newflakeid, label])
        if label == 'glue' or label == 'others':
            exclude_oriname_newflakeids.append([ori_name, newflakeid, label])

    return flake_oriname_newflakeids, exclude_oriname_newflakeids


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
        if label == 'thin' or label == 'thick':
            oriname_newflakeids.append([ori_name, newflakeid, label])

    return oriname_newflakeids


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


def draw_img(flake, online_save_path, img_name, k):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    contours = flake['flake_contour_loc']
    contours[:,0] = contours[:,0] - flake['flake_large_bbox'][0]
    contours[:,1] = contours[:,1] - flake['flake_large_bbox'][2]
    contours = np.expand_dims(np.flip(contours), 1).astype(np.int32)
    ori_img = copy.copy(flake['flake_img'])
    contour_img = cv2.drawContours(ori_img, contours, -1, (255,0,0), 2)
    ori_contour = np.concatenate([flake['flake_img'], 255*np.ones([5, ori_img.shape[1],3], dtype=np.uint8), contour_img], axis=0)
    ax.imshow(ori_contour)
    ax.axis('off')
    fig.savefig(os.path.join(online_save_path, 'img-%s-%d.png'%(img_name,  k)))
    plt.close()


# process one sub exp, read all the data, and do clustering
def classify_one_subexp(clusterdetail_anno_file, flakeglue_anno_file, thickthin_anno_file, all_anno_file, subexp_dir, rslt_dir, flake_save_path, method, online_save_path):
    img_names = os.listdir(subexp_dir)
    img_names.sort()
    print('process ' + subexp_dir)
    print('n images: %d'%(len(img_names)))

    if os.path.exists(flake_save_path + 'flakes.p'):
        img_flakes_ori2new = pickle.load(open(flake_save_path + 'flakes.p', 'rb'))
    else:
        img_flakes_ori2new = Parallel(n_jobs=args.n_jobs)(delayed(load_one_image)(os.path.join(subexp_dir, img_names[i]), os.path.join(rslt_dir, img_names[i][:-4]+'.p'))
                                     for i in range(len(img_names)))
        pickle.dump(img_flakes_ori2new, open(flake_save_path + 'flakes.p', 'wb'))
    print('loading done')
    img_flakes = [img_flakes_ori2new[_][0] for _ in range(len(img_names))]
    ori2news = [img_flakes_ori2new[_][1] for _ in range(len(img_names))]

    if not os.path.exists(all_anno_file):
        os.system('cp %s %s' % (clusterdetail_anno_file, all_anno_file))

        # clustername_labels = readclusterdb(cluster_anno_file)
        indi_flake_oriname_newflakeids, exlude_indi_flake_oriname_newflakeids = readflakegluedb(flakeglue_anno_file)
        thickthin_oriname_newflakeids = readthickthindb(thickthin_anno_file)
        # n_total_clusters = len(clustername_labels)
        n_indi_flakes = len(indi_flake_oriname_newflakeids)
        n_exlude_indi_flakes = len(exlude_indi_flake_oriname_newflakeids)
        n_thickthin_flakes = len(thickthin_oriname_newflakeids)
        print(n_indi_flakes, n_exlude_indi_flakes, n_thickthin_flakes)

        createdb(all_anno_file, img_names, ori2news, exlude_indi_flake_oriname_newflakeids, thickthin_oriname_newflakeids)

    # read from the annotation, find what's already labeled and what need to be labeled.
    oriname_newflakeids = readalldb(all_anno_file)
    num_regions = len(oriname_newflakeids)

    # find train/ test ids
    train_ids = [i for i in range(num_regions) if oriname_newflakeids[i][2] in ['glue', 'thin', 'thick']]
    test_ids = [i for i in range(num_regions) if oriname_newflakeids[i][2] is None]
    num_train = len(train_ids)
    num_test = len(test_ids)
    print('train: %d, test: %d'%(num_train, num_test))

    if os.path.exists(flake_save_path + 'thickness_feats.p'):
        all_feas = pickle.load(open(flake_save_path + 'thickness_feats.p', 'rb'))
    else:
        all_feas = []
        for i in range(num_regions):
            img_id = img_names.index(oriname_newflakeids[i][0])
            all_feas.append(np.concatenate([img_flakes[img_id][oriname_newflakeids[i][1]]['flake_shape_fea'],
                                            img_flakes[img_id][oriname_newflakeids[i][1]]['flake_color_fea']]))
            # all_feas.append(img_flakes[img_id][oriname_newflakeids[i][1]])
        # print(all_feas[0].shape)
        all_feas = np.stack(all_feas)
        pickle.dump(all_feas, open(flake_save_path + 'thickness_feats.p', 'wb'))
    print('fea loading done')
    print(all_feas.shape)

    train_feas = all_feas[train_ids, :]
    # train_feas = np.array(train_feas)
    test_feas = all_feas[test_ids, :]
    # test_feas = [all_feas[i, :] for i in test_ids]
    # test_feas = np.array(test_feas)
    # print(train_feas.shape, test_feas.shape)
    # train labels is 0, 1, 2
    train_labels = [labelmaps[oriname_newflakeids[x][2]] for x in train_ids]

    # normalize feature (subtract mean, divide by l2 norm)
    if os.path.exists(flake_save_path + 'thickness_normfea.p'):
        norm_fea = pickle.load(open(flake_save_path + 'thickness_normfea.p', 'rb'))
        mean_fea = norm_fea['mean']
        std_fea = norm_fea['std']
    else:
        mean_fea = np.mean(train_feas, axis=0, keepdims=True)
        std_fea = np.std(train_feas, axis=0, keepdims=True)
        norm_fea = {}
        norm_fea['mean'] = mean_fea
        norm_fea['std'] = std_fea
        pickle.dump(norm_fea, open(flake_save_path + 'thickness_normfea.p', 'wb'))

    train_feas -= mean_fea
    train_feas /= std_fea
    test_feas -= mean_fea
    test_feas /= std_fea

    # save classifier:
    if os.path.exists(flake_save_path + 'thickness_classifier-%s.p' % (method)):
        clf = pickle.load(open(flake_save_path + 'thickness_classifier-%s.p' % (method), 'rb'))
    else:
        if method == 'linearsvm':
            clf = LinearSVC(random_state=0, tol=1e-5, fit_intercept=True, C=1, max_iter=5e4)
            clf.fit(train_feas, train_labels)
        # elif method == 'ridge':
        #     clf = RidgeClassifier(random_state=0, alpha=1, fit_intercept=False)
        #     clf.fit(train_feas, train_labels)
        else:
            raise NotImplementedError

        pickle.dump(clf, open(flake_save_path + 'thickness_classifier-%s.p' % (method), 'wb'))

    # test, compare with cluster labels
    train_pred_cls = clf.predict(train_feas)
    train_pred_scores = clf.decision_function(train_feas)
    test_pred_cls = clf.predict(test_feas)
    test_pred_name = [labelmapsback[x] for x in test_pred_cls]
    test_pred_scores = clf.decision_function(test_feas)
    from sklearn.metrics import accuracy_score, confusion_matrix
    train_acc = accuracy_score(train_labels, train_pred_cls)
    train_conf = confusion_matrix(train_labels, train_pred_cls)
    # val_acc = accuracy_score(, test_pred_cls)
    print('train acc: %.4f' % (train_acc))
    print(train_conf)


    if not os.path.exists(os.path.join(online_save_path, 'all_test_names.txt')):
        all_test_imgflake = []
        glue_test_imgflake = []
        thin_test_imgflake = []
        thick_test_imgflake = []

        all_test_name = []
        for i in range(num_test):
            entry = oriname_newflakeids[test_ids[i]]
            all_test_imgflake.append('%s-%d,%s,%s' % (entry[0], entry[1], test_pred_name[i], entry[3]))
            if test_pred_name[i] == 'glue':
                glue_test_imgflake.append('%s-%d,%s,%s' % (entry[0], entry[1], test_pred_name[i], entry[3]))
            elif test_pred_name[i] == 'thin':
                thin_test_imgflake.append('%s-%d,%s,%s' % (entry[0], entry[1], test_pred_name[i], entry[3]))
            elif test_pred_name[i] == 'thick':
                thick_test_imgflake.append('%s-%d,%s,%s' % (entry[0], entry[1], test_pred_name[i], entry[3]))
            all_test_name.append('%s-%d'%(entry[0], entry[1]))
            ori_idx = img_names.index(entry[0])
            draw_img(img_flakes[ori_idx][entry[1]], online_save_path, entry[0], entry[1])

        print('test num: %d, unique num: %d'%(len(all_test_name), len(list(set(all_test_name)))))

        with open(os.path.join(online_save_path, 'all_test_names.txt'), 'w') as f:
            f.write('\n'.join(all_test_imgflake))
        with open(os.path.join(online_save_path, 'glue_test_names.txt'), 'w') as f:
            f.write('\n'.join(glue_test_imgflake))
        with open(os.path.join(online_save_path, 'thin_test_names.txt'), 'w') as f:
            f.write('\n'.join(thin_test_imgflake))
        with open(os.path.join(online_save_path, 'thick_test_names.txt'), 'w') as f:
            f.write('\n'.join(thick_test_imgflake))


def main():
    data_path = '../data/data_jan2019'
    result_path = '../results/data_jan2019_script/mat'
    # fig_path = '../results/data_jan2019_script/fig'
    cluster_anno_file = '../data/data_jan2019_anno/anno_cluster_YoungJaeShinSamples_4_useryoungjae.db'
    clusterdetail_anno_file = '../data/data_jan2019_anno/anno_cluster_detailed_YoungJaeShinSamples_4_useryoungjae.db'
    flakeglue_anno_file = '../data/data_jan2019_anno/anno_flakeglue_YoungJaeShinSamples_4_useryoungjae.db'
    thickthin_anno_file = '../data/data_jan2019_anno/anno_thickthin_incomplete_YoungJaeShinSamples_4_useryoungjae.db'
    all_anno_file = '../data/data_jan2019_anno/anno_all_incomplete_YoungJaeShinSamples_4_useryoungjae.db'


    cluster_path = '../results/data_jan2019_script/cluster_sort_%d'%(hyperparams['size_thre'])
    online_path = '../results/data_jan2019_script/cluster_sort_%d_thickness_online_v2'%(hyperparams['size_thre'])
    towrite_path = '../results/data_jan2019_script/cluster_sort_%d_thickness_v2' % (hyperparams['size_thre'])

    # testclf_img_path = '../results/data_jan2019_script/cluster_sort_%d_testclf0_oriimg'%(hyperparams['size_thre'])

    exp_names = os.listdir(data_path)
    exp_names.sort()
    exp_names = [ename for ename in exp_names if ename[0] != '.']

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
            
            flake_save_path = os.path.join(towrite_path, exp_name+sname)
            # cluster_save_path = os.path.join(cluster_path, exp_name, sname, '%s_%s'%(hyperparams['cluster_fea'], hyperparams['cluster_method']))
            # towrite_save_path = os.path.join(towrite_path, exp_name, sname + '_' + method)
            online_save_path = os.path.join(online_path, exp_name, sname + '_' + method)
            # testclf_img_save_path = os.path.join(testclf_img_path, exp_name, sname + '_' + method)

            # if not os.path.exists(testclf_cluster_save_path):
            #     os.makedirs(testclf_cluster_save_path)
            
            if not os.path.exists(towrite_path):
                os.makedirs(towrite_path)

            if not os.path.exists(online_save_path):
                os.makedirs(online_save_path)
            
            classify_one_subexp(clusterdetail_anno_file, flakeglue_anno_file, thickthin_anno_file, all_anno_file, os.path.join(data_path, exp_name, sname), os.path.join(result_path, exp_name, sname), flake_save_path, method, online_save_path)


if __name__ == '__main__':
    main()