#do clustering for each sub folder

import scipy.io as sio
import scipy.misc as misc
from skimage import color
from skimage import io
import numpy as np
import os
import itertools
import pickle
import sklearn
from sklearn.cluster import KMeans
import math
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from multiprocessing import Pool
from joblib import Parallel, delayed


class flake(object):
    def __init__(self, img_name=None, flake_id=None, flake_size=0, flake_bbox=None, flake_center=None, flake_fea=None, flake_mask=None, flake_img=None):
        self.img_name = img_name
        self.flake_id = flake_id
        self.flake_size = flake_size
        self.flake_bbox = flake_bbox
        self.flake_center = flake_center
        self.flake_fea = flake_fea
        self.flake_img = flake_img
        # self.flake_mask = flake_mask


# load the detected flake and get features for the flake
def load_one_image(img_name, mask_name):
# def load_one_image(names):
#     img_name, mask_name = names
    flake_labelmap = sio.loadmat(mask_name)
    # img = misc.imread(img_name)
    img = io.imread(img_name)
    img_gray = color.rgb2gray(img)
    img_hsv = color.rgb2hsv(img)
    # build a list of flakes
    num_flakes = flake_labelmap['num_regions'][0][0]
    flakes = []
    for i in range(num_flakes):
        flake_id = i+1
        flake_size = sum(flake_labelmap['new_region_labelmap']==i+1)
        f_mask_r, f_mask_c= np.nonzero(flake_labelmap['new_region_labelmap']==i+1)
        height, width = flake_labelmap['new_region_labelmap'].shape
        f_mask_r_min = min(f_mask_r)
        f_mask_r_max = max(f_mask_r)
        f_mask_height = f_mask_r_max - f_mask_r_min
        f_mask_c_min = min(f_mask_c)
        f_mask_c_max = max(f_mask_c)
        f_mask_width = f_mask_c_max - f_mask_c_min

        flake_bbox = [f_mask_r_min, f_mask_r_max, f_mask_c_min, f_mask_c_max]
        flake_center = [np.mean(f_mask_r), np.mean(f_mask_c)]
        flake_img = img[max(0, f_mask_r_min-int(0.1*f_mask_height)): min(height, f_mask_r_max+int(0.1*f_mask_height)),
                            max(0, f_mask_c_min - int(0.1 * f_mask_width)): min(width, f_mask_c_max + int(0.1 * f_mask_width)), :]
        # compute features for the flake
        # the features are: contrast_gray, contrast_v, mean RGB, mean gray, mean HSV, std RGB, std gray, std HSV
        f_fea = []
        f_fea.extend(list(flake_labelmap['region_feas'][i]))
        f_rgb = img[f_mask_r, f_mask_c, :]
        f_gray = img_gray[f_mask_r, f_mask_c]
        f_hsv = img_hsv[f_mask_r, f_mask_c, :]
        f_fea.extend(list(np.mean(f_rgb, (0))))
        f_fea.append(np.mean(f_gray))
        f_fea.extend(list(np.mean(f_hsv, (0))))
        f_fea.extend(list(np.std(f_rgb, (0))))
        f_fea.append(np.std(f_gray))
        f_fea.extend(list(np.std(f_hsv, (0))))
        flake_fea = np.array(f_fea)
        flake_i = flake(img_name, flake_id=flake_id, flake_size=flake_size, flake_bbox=flake_bbox, flake_center=flake_center, flake_fea=flake_fea, flake_img=flake_img)

        flakes.append(flake_i)

    return flakes


# process one sub exp, read all the data, and do clustering
def cluster_one_subexp(subexp_dir, rslt_dir, flake_save_path, cluster_save_path, fea='contrast'):
    img_names = os.listdir(subexp_dir)
    print('process ' + subexp_dir)
    print('n images: %d'%(len(img_names)) )
    if os.path.exists(flake_save_path + 'flakes.p'):
        img_flakes = pickle.load(open(flake_save_path + 'flakes.p', 'rb'))
    else:
        # img_flakes = Parallel(n_jobs=20)(delayed(load_one_image)(os.path.join(subexp_dir, img_names[i]), os.path.join(rslt_dir, img_names[i][:-4]+'.mat') )
        #                                  for i in range(len(img_names)))
        # a = [os.path.join(subexp_dir, img_names[i]) for i in range(len(img_names))]
        # b = [os.path.join(rslt_dir, img_names[i][:-4]+'.mat') for i in range(len(img_names))]
        # pool = Pool(20)
        # img_flakes = pool.map(load_one_image, zip(a, b))
        img_flakes = []
        for i in range(len(img_names)):
            img_flakes.append(load_one_image(os.path.join(subexp_dir, img_names[i]), os.path.join(rslt_dir, img_names[i][:-4]+'.mat')))
        img_flakes = list(itertools.chain.from_iterable(img_flakes))
        pickle.dump(img_flakes, open(flake_save_path + 'flakes.p', 'wb'))

    num_flakes = len(img_flakes)

    # clustering
    all_feas = [img_flakes[i].flake_fea for i in range(num_flakes)]
    all_feas = np.array(all_feas)
    if fea == 'contrast':
        all_feas = all_feas[:, :2]

    num_clusters = 50
    all_feas = sklearn.preprocessing.normalize(all_feas, norm='l2', axis=0)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_jobs=-1).fit(all_feas)
    pickle.dump(kmeans, open(flake_save_path + 'clusters.p', 'wb'))
    assignment = kmeans.labels_

    # visualize each cluster
    Parallel(n_jobs=20)(delayed(visualize_cluster)(i, [img_flakes[mm] for mm in np.nonzero(assignment==i)[0]], cluster_save_path) for i in range(num_clusters))
    # for i in range(num_clusters):
    #     visualize_cluster(i, img_flakes[assignment==i], cluster_save_path)


# visualize each cluster
def visualize_cluster(cluster_id, flakes, cluster_save_path):
    num_flakes = len(flakes)
    num_row = 5
    # num_vis = int(math.ceil(num_flakes * 1.0 / num_row / num_row))
    # fig, ax = plt.subplots(num_row, num_row)
    fig = plt.figure()
    cnt = 0
    for i in range(num_flakes):
        if i % (num_row*num_row) == 0:
            fig.clear()
        ax = fig.add_subplot(num_row, num_row, i %(num_row*num_row) + 1)
        # ax.plot() = fig.add_subplot(num_row, num_row, i %(num_row*num_row) + 1)
        ax.imshow(flakes[i].flake_img)
        ax.axis('off')
        if (i+1) % (num_row*num_row) == 0 or i == num_flakes-1:
            plt.tight_layout()
            fig.subplots_adjust(wspace=0.05, hspace=0.05)
            fig.savefig(os.path.join(cluster_save_path, 'cluster-%d-%d.png'%(cluster_id, cnt)))
            cnt += 1

    plt.close()



def main():
    data_path = '/nfs/bigmind/add_disk0/boyu/Projects/material_segmentation/data/data_jan2019'
    # data_exp = os.listdir(data_path)
    rslt_path = '/nfs/bigmind/add_disk0/boyu/Projects/material_segmentation/results/data_jan2019'
    fea = 'contrast'
    # fea = 'all'

    cluster_path = '/nfs/bigmind/add_disk0/boyu/Projects/material_segmentation/results/data_jan2019_cluster%s'%(fea)
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)
    exp = os.listdir(rslt_path)

    num_exps = len(exp)
    for i in range(num_exps):
        subexp = os.listdir(os.path.join(rslt_path, exp[i]))
        num_subexp = len(subexp)
        for j in range(num_subexp):
            subexp_dir = os.path.join(data_path, exp[i], subexp[j])
            rslt_dir = os.path.join(rslt_path, exp[i], subexp[j])
            flake_save_path = os.path.join(cluster_path, exp[i]+subexp[j])
            cluster_save_path = os.path.join(cluster_path, exp[i], subexp[j])
            if not os.path.exists(cluster_save_path):
                os.makedirs(cluster_save_path)
            cluster_one_subexp(subexp_dir, rslt_dir, flake_save_path, cluster_save_path, fea)


if __name__ == '__main__':
    main()