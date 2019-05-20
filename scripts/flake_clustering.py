"""
Clustering flakes based on different features: shape, color stats

By: Boyu Wang (boywang@cs.stonybrook.edu)
Created Data: 21 Feb 2019
Last Modified Date: 15 Mar 2019
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
from multiprocessing import Pool
from joblib import Parallel, delayed
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
import math
# from multiprocessing import Pool

parser = argparse.ArgumentParser(description='flake segmentation')
parser.add_argument('--exp_sid', default=5, type=int, help='exp start id')
parser.add_argument('--exp_eid', default=6, type=int, help='exp end id')
parser.add_argument('--subexp_sid', default=0, type=int, help='subexp start id')
parser.add_argument('--subexp_eid', default=1, type=int, help='subexp end id')
parser.add_argument('--n_jobs', default=8, type=int, help='multiprocessing cores')
# parser.add_argument('--c_sid', default=0, type=int, help='subexp start id')
# parser.add_argument('--c_eid', default=400, type=int, help='subexp end id')

args = parser.parse_args()


hyperparams = { 'size_thre': 784, # after detect foreground regions, filter them based on its size. (784=28*28 corresponds to 5 around um regions)
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

# process one sub exp, read all the data, and do clustering
def cluster_one_subexp(subexp_dir, rslt_dir, flake_save_path, cluster_save_path):
    img_names = os.listdir(subexp_dir)
    img_names = [n_i for n_i in img_names if n_i[0] != '.']
    img_names.sort()
    print('process ' + subexp_dir)
    print('n images: %d'%(len(img_names)) )
    if os.path.exists(flake_save_path + 'flakes.p'):
        img_flakes = pickle.load(open(flake_save_path + 'flakes.p', 'rb'))
    else:
        img_flakes = Parallel(n_jobs=args.n_jobs)(delayed(load_one_image)(os.path.join(subexp_dir, img_names[i]), os.path.join(rslt_dir, img_names[i][:-4]+'.p'))
                                         for i in range(len(img_names)))
        # a = [os.path.join(subexp_dir, img_names[i]) for i in range(len(img_names))]
        # b = [os.path.join(rslt_dir, img_names[i][:-4]+'.mat') for i in range(len(img_names))]
        # pool = Pool(20)
        # img_flakes = pool.map(load_one_image, zip(a, b))
        # img_flakes = []
        # for i in range(len(img_names)):
        #     img_flakes.append(load_one_image(os.path.join(subexp_dir, img_names[i]), os.path.join(rslt_dir, img_names[i][:-4]+'.mat')))
        img_flakes = list(itertools.chain.from_iterable(img_flakes))
        pickle.dump(img_flakes, open(flake_save_path + 'flakes.p', 'wb'))

    num_flakes = len(img_flakes)

    # clustering
    # feas = ['shape', 'color', 'all']
    # feas = ['all']
    feas = [hyperparams['cluster_fea']]
    for fea in feas:
        if fea == 'shape':
            all_feas = [img_flakes[i]['flake_shape_fea'] for i in range(num_flakes)]
        elif fea == 'color':
            all_feas = [img_flakes[i]['flake_color_fea'] for i in range(num_flakes)]
        elif fea == 'all':
            all_feas = [np.concatenate([img_flakes[i]['flake_color_fea'], img_flakes[i]['flake_shape_fea']]) for i in range(num_flakes)]
        all_feas = np.array(all_feas)
        contrast_fea = np.array([img_flakes[i]['flake_color_fea'][0:2] for i in range(num_flakes)])
        print('number of flakes: %d' %(num_flakes))
        # if fea == 'contrast':
        #     all_feas = all_feas[:, :2]

        # cluster_methods = ['kmeans', 'meanshift', 'affinity']
        # cluster_methods = ['kmeans', 'affinity']
        # cluster_methods = ['meanshift']
        # cluster_methods = ['affinity']
        cluster_methods = [hyperparams['cluster_method']]
        # cluster_methods = ['kmeans']

        for c_ms in cluster_methods:
            perform_clustering(c_ms, all_feas, contrast_fea, fea, flake_save_path, cluster_save_path, img_flakes)
        # num_clusters = 50
        # if os.path.exists(flake_save_path + '_%s_clusters.p'%fea):
        #     kmeans = pickle.load(open(flake_save_path + '_%s_clusters.p'%fea, 'rb'))
        # else:
        #     all_feas = sklearn.preprocessing.normalize(all_feas, norm='l2', axis=0)
        #     kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_jobs=-1).fit(all_feas)
        #     pickle.dump(kmeans, open(flake_save_path + '_%s_clusters.p'%fea, 'wb'))
        # assignment = kmeans.labels_

        # cluster_save_path_fea = os.path.join(cluster_save_path, fea)
        # if not os.path.exists(cluster_save_path_fea):
        #     os.makedirs(cluster_save_path_fea)
        # # visualize each cluster
        # Parallel(n_jobs=30)(delayed(visualize_cluster)(i, [img_flakes[mm] for mm in np.nonzero(assignment==i)[0]], cluster_save_path_fea) for i in range(num_clusters))

        # i = 1
        # visualize_cluster(i, [img_flakes[mm] for mm in np.nonzero(assignment==i)[0]], cluster_save_path_fea)
    # for i in range(num_clusters):
    #     visualize_cluster(i, img_flakes[assignment==i], cluster_save_path)


def perform_clustering(method, all_feas, contrast_fea, fea_type, flake_save_path, cluster_save_path, img_flakes):
    if os.path.exists(flake_save_path + '_%s_clusters-%s.p'%(fea_type, method)):
        print('loading cluster')
        cluster_rslt = pickle.load(open(flake_save_path + '_%s_clusters-%s.p'%(fea_type, method), 'rb'))
        print('loading done')
        cluster_rslt = cluster_rslt['cluster_info']

        # assignment is for unsorted index
        assignment = cluster_rslt.labels_
        num_clusters = len(np.unique(assignment))
        # sort center based on contrast value
        center_contrast_feas = np.zeros([num_clusters, contrast_fea.shape[1]])
        for i in range(num_clusters):
             mm = np.nonzero(assignment==i)[0]
             center_contrast_feas[i,:] = np.mean(contrast_fea[mm, :])
        center_srt_idxs = np.argsort(np.abs(center_contrast_feas).sum(1))
        unsort2sort = np.unique(center_srt_idxs, return_index=True)[1]

    else:
        # all_feas = sklearn.preprocessing.normalize(all_feas, norm='l2', axis=0)
        all_feas = StandardScaler().fit_transform(all_feas)
        if method == 'kmeans':
            num_clusters = 50
            cluster_rslt = KMeans(n_clusters=num_clusters, random_state=0, n_jobs=-1).fit(all_feas)
        elif method == 'meanshift':
            # The following bandwidth can be automatically detected using
            bandwidth = estimate_bandwidth(all_feas, quantile=0.1)#, n_samples=int(all_feas.shape[0]/10))
            cluster_rslt = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(all_feas)
        elif method == 'affinity':
            cluster_rslt = AffinityPropagation().fit(all_feas)
        else:
            raise NotImplementedError

        # assignment is for unsorted index
        assignment = cluster_rslt.labels_
        num_clusters = len(np.unique(assignment))
        # sort center based on contrast value
        center_contrast_feas = np.zeros([num_clusters, contrast_fea.shape[1]])
        for i in range(num_clusters):
             mm = np.nonzero(assignment==i)[0]
             center_contrast_feas[i,:] = np.mean(contrast_fea[mm, :])
        center_srt_idxs = np.argsort(np.abs(center_contrast_feas).sum(1))
        unsort2sort = np.unique(center_srt_idxs, return_index=True)[1]

        # print('%s, %s, %s: %d'%(flake_save_path, fea_type, method, num_clusters))
        cluster_centers = cluster_rslt.cluster_centers_[center_srt_idxs]
        cluster_center_dis = cdist(cluster_centers, cluster_centers)
        to_save = dict()
        # unsorted cluster
        to_save['cluster_info'] = cluster_rslt
        to_save['center_srt_idxs'] = center_srt_idxs
        to_save['unsort2sort'] = unsort2sort
        # sorted clusters
        to_save['cluster_centers'] = cluster_centers
        to_save['cluster_center_dis'] = cluster_center_dis
        pickle.dump(to_save, open(flake_save_path + '_%s_clusters-%s.p'%(fea_type, method), 'wb'), protocol=4)

        print('PCA 3D projection...')
        pca = PCA(n_components=3)
        x1 = pca.fit_transform(all_feas)

        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(x1[:,0], x1[:,1], x1[:,2], c=assignment, edgecolors=None)
        ax = fig.add_subplot(122)
        # ax.imshow(cluster_center_dis, extent=[0, 1, 0, 1])
        ax.imshow(cluster_center_dis, cmap='jet')
        # ax.set_xticks(np.arange(num_clusters))
        # ax.set_yticks(np.arange(num_clusters))
        #plt.show()
        plt.savefig(flake_save_path + '_%s_clusters-%s_vis.png'%(fea_type, method), dpi=300)
        plt.close()

    print('%s, %s, %s: %d'%(flake_save_path, fea_type, method, num_clusters))
    cluster_save_path_fea = os.path.join(cluster_save_path, '%s_%s'%(fea_type, method))
    if not os.path.exists(cluster_save_path_fea):
        os.makedirs(cluster_save_path_fea)
    # visualize each cluster
    # Parallel(n_jobs=30)(delayed(visualize_cluster)(unsort2sort[i], [img_flakes[mm] for mm in np.nonzero(assignment==i)[0]], cluster_save_path_fea) for i in range(num_clusters))
    # for i in range(num_clusters):
    #     visualize_cluster2(unsort2sort[i], [img_flakes[mm] for mm in np.nonzero(assignment==i)[0]], cluster_save_path_fea)
    # Parallel(n_jobs=30)(delayed(visualize_cluster2)(unsort2sort[i], [img_flakes[mm] for mm in np.nonzero(assignment==i)[0]], cluster_save_path_fea) for i in range(num_clusters))

    # cluster_ids_todo = []
    # for i in range(num_clusters):
    #     ci = unsort2sort[i]
    #     # if not os.path.exists(os.path.join(cluster_save_path_fea, str(ci))) or not os.path.exists(os.path.join(cluster_save_path_fea, str(ci), 'names.txt')):
    #     #     print(os.path.join(cluster_save_path_fea, str(ci), 'names.txt'))

    #     num_img = np.sum(assignment == i)
    #     num_cls_imgs = int(math.ceil(num_img / 25))
    #     if not os.path.exists(os.path.join(cluster_save_path_fea, 'cluster-%d-%d.png'%(ci, num_cls_imgs-1))):
    #         cluster_ids_todo.append(i)

    # print(len(cluster_ids_todo))

    with Pool(processes=args.n_jobs) as pool:
        pool.starmap(visualize_cluster2, [(unsort2sort[i], [img_flakes[mm] for mm in np.nonzero(assignment==i)[0]], cluster_save_path_fea) for i in range(num_clusters)])
        # pool.starmap(visualize_cluster2, [(unsort2sort[i], [img_flakes[mm] for mm in np.nonzero(assignment==i)[0]], cluster_save_path_fea) for i in cluster_ids_todo])
        # pool.starmap(visualize_cluster2, [(unsort2sort[i], [img_flakes[mm] for mm in np.nonzero(assignment==i)[0]], cluster_save_path_fea) for i in range(args.c_sid, args.c_eid)])
        pool.starmap(visualize_cluster, [(unsort2sort[i], [img_flakes[mm] for mm in np.nonzero(assignment==i)[0]], cluster_save_path_fea) for i in range(num_clusters)])
        # pool.starmap(visualize_cluster, [(unsort2sort[i], [img_flakes[mm] for mm in np.nonzero(assignment==i)[0]], cluster_save_path_fea) for i in cluster_ids_todo])


# visualize each cluster, plot multiple images together
def visualize_cluster(cluster_id, flakes, cluster_save_path):
    num_flakes = len(flakes)
    print(cluster_id, num_flakes)
    num_row = 5
    # num_vis = int(math.ceil(num_flakes * 1.0 / num_row / num_row))
    # fig, ax = plt.subplots(num_row, num_row)
    fig = plt.figure()
    cnt = 0
    for i in range(num_flakes):
        if i % (num_row*num_row) == 0:
            fig.clear()
            # print(i)
        ax = fig.add_subplot(num_row, num_row, i %(num_row*num_row) + 1)
        # # also plot contour
        # contours = flakes[i]['flake_contour_loc']
        # contours[:,0] = contours[:,0] - flakes[i]['flake_large_bbox'][0]
        # contours[:,1] = contours[:,1] - flakes[i]['flake_large_bbox'][2]
        # contours = np.expand_dims(np.flip(contours), 1).astype(np.int32)
        # contour_img = cv2.drawContours(flakes[i]['flake_img'], contours, -1, (255,0,0), 2)
        # ax.imshow(contour_img)
        ax.imshow(flakes[i]['flake_img'])

        # print(flakes[i]['flake_img'].shape)
        # print(flakes[i]['flake_bbox'])
        ax.axis('off')
        if (i+1) % (num_row*num_row) == 0 or i == num_flakes-1:
            # plt.tight_layout()
            # fig.subplots_adjust(wspace=0.05, hspace=0.05)
            fig.savefig(os.path.join(cluster_save_path, 'cluster-%d-%d.png'%(cluster_id, cnt)))
            cnt += 1

    plt.close()


# save each flake image separately
def visualize_cluster2(cluster_id, flakes, cluster_save_path):
    num_flakes = len(flakes)
    print(cluster_id, num_flakes)
    num_row = 1
    # num_vis = int(math.ceil(num_flakes * 1.0 / num_row / num_row))
    # fig, ax = plt.subplots(num_row, num_row)
    # fig = plt.figure()
    # cnt = 0
    if not os.path.exists(os.path.join(cluster_save_path, str(cluster_id))):
        os.makedirs(os.path.join(cluster_save_path, str(cluster_id)))
    file_name_list = []

    for i in range(num_flakes):
        # if i % (num_row*num_row) == 0:
        fig = plt.figure()
        fig.clear()
        ax = fig.add_subplot(num_row, num_row, i %(num_row*num_row) + 1)
        ax.imshow(flakes[i]['flake_img'])
        ax.axis('off')

        ori_name = flakes[i]['img_name'].split('/')[-1][:-4]
        ori_flakeid = flakes[i]['flake_id']
        fig.savefig(os.path.join(cluster_save_path, str(cluster_id), 'flake%04d-%s-%d.png'%(i, ori_name, ori_flakeid)))
        file_name_list.append('flake%04d-%s-%d.png'%(i, ori_name, ori_flakeid))

        # # also plot contour
        fig.clear()
        ax = fig.add_subplot(num_row, num_row, i %(num_row*num_row) + 1)
        contours = flakes[i]['flake_contour_loc']
        contours[:,0] = contours[:,0] - flakes[i]['flake_large_bbox'][0]
        contours[:,1] = contours[:,1] - flakes[i]['flake_large_bbox'][2]
        contours = np.expand_dims(np.flip(contours), 1).astype(np.int32)
        contour_img = cv2.drawContours(flakes[i]['flake_img'], contours, -1, (255,0,0), 2)
        ax.imshow(contour_img)
        ax.axis('off')
        fig.savefig(os.path.join(cluster_save_path, str(cluster_id), 'flake%04d-%s-%d_bw.png'%(i, ori_name, ori_flakeid)))

        # also show the location of the flake in ori image
        # fig.clear()
        # ax = fig.add_subplot(num_row, num_row, i %(num_row*num_row) + 1)
        flake_large_bbox = flakes[i]['flake_large_bbox']
        ori_img = Image.open(flakes[i]['img_name'])
        ori_img = np.array(ori_img).astype(np.uint8)
        # imH, imW, _ = ori_img.shape
        # print(ori_img.shape, flake_large_bbox[2]-1, flake_large_bbox[0], flake_large_bbox[3]-1, flake_large_bbox[1])
        bbox_img = cv2.rectangle(ori_img, (flake_large_bbox[2]-1, flake_large_bbox[0]), (flake_large_bbox[3]-1, flake_large_bbox[1]),(255,0,0),2)
        cv2.imwrite(os.path.join(cluster_save_path, str(cluster_id), 'flake%04d-%s-%d_bbox.png'%(i, ori_name, ori_flakeid)), cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR))
        # ax.imshow(bbox_img)
        # ax.axis('off')
        # fig.savefig(os.path.join(cluster_save_path, str(cluster_id), 'flake%04d-%s-%d_bbox.png'%(i, ori_name, ori_flakeid)))

        plt.close()
        gc.collect()

    with open(os.path.join(cluster_save_path, str(cluster_id), 'names.txt'), 'w') as f:
        f.write("\n".join(file_name_list))

    # plt.close()

    gc.collect()


def main():
    data_path = '../data/data_jan2019'
    result_path = '../results/data_jan2019_script/mat'
    # cluster_path = '../results/data_jan2019_script/cluster'
    cluster_path = '../results/data_jan2019_script/cluster_sort_%d'%(hyperparams['size_thre'])

    exp_names = os.listdir(data_path)
    exp_names.sort()
    # exp_names = exp_names[args.exp_sid: args.exp_eid]

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
            flake_save_path = os.path.join(cluster_path, exp_name+sname)
            cluster_save_path = os.path.join(cluster_path, exp_name, sname)
            
            if not os.path.exists(cluster_save_path):
                os.makedirs(cluster_save_path)
            
            cluster_one_subexp(os.path.join(data_path, exp_name, sname), os.path.join(result_path, exp_name, sname), flake_save_path, cluster_save_path)



if __name__ == '__main__':
    main()






