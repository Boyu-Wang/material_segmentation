"""
Binary Graphene classification. 
For each image, estimate the bg color
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
from mpl_toolkits.mplot3d import Axes3D
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

def load_data(data_path, graphene_path, thick_path, glue_path):
    size_thre = 100

    graphene_flakes = []
    graphene_feats = []
    graphene_names = []

    other_flakes = []
    other_feats = []
    other_names = []

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
            
            # graphene_flakes.extend(tmp_flake)
            for i in range(len(tmp_flake)):
                if tmp_flake[i]['flake_size'] > size_thre:
                    graphene_names.append(fname+'-'+str(tmp_flake[i]['flake_id']))
                    f_mask_r_min, f_mask_r_max, f_mask_c_min, f_mask_c_max = tmp_flake[i]['flake_exact_bbox']
                    f_mask_height = f_mask_r_max - f_mask_r_min
                    f_mask_width = f_mask_c_max - f_mask_c_min
                    flake_large_bbox = [max(0, f_mask_r_min - int(0.5 * f_mask_height)),
                                        min(imH, f_mask_r_max + int(0.5 * f_mask_height)),
                                        max(0, f_mask_c_min - int(0.5 * f_mask_width)),
                                        min(imW, f_mask_c_max + int(0.5 * f_mask_width))]
                    tmp_flake[i]['flake_large_bbox'] = flake_large_bbox
                    tmp_flake[i]['flake_img'] = im_rgb[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3], :].astype(np.uint8)

                    graphene_flakes.append(tmp_flake[i])


    num_graphene = len(graphene_flakes)
    print("number of graphene: {}".format(num_graphene))
    # load features
    for g_flake in graphene_flakes:
        # graphene_feats.append(np.concatenate([g_flake['flake_shape_fea'],g_flake['flake_color_fea']]))
        graphene_feats.append(g_flake['flake_color_fea'])
    graphene_feats = np.stack(graphene_feats)
    
    # load thick
    fnames = os.listdir(thick_path)
    for fname in fnames:
        tmp_flake = pickle.load(open(os.path.join(thick_path, fname), 'rb'))
        tmp_flake = tmp_flake['flakes']
        if len(tmp_flake) > 0:
            # print(os.path.join(data_path, fname[:-2] + 'tiff'))
            image = Image.open(os.path.join(data_path, fname[:-2] + 'tiff'))
            im_rgb = np.array(image).astype('float')
            imH, imW, _ = im_rgb.shape
            
            # thick_flakes.extend(tmp_flake)
            for i in range(len(tmp_flake)):
                tmp_flake_name = fname+'-'+str(tmp_flake[i]['flake_id'])
                if tmp_flake[i]['flake_size'] > size_thre and tmp_flake_name not in graphene_names:
                    other_names.append(fname+'-'+str(tmp_flake[i]['flake_id']))
                    
                    f_mask_r_min, f_mask_r_max, f_mask_c_min, f_mask_c_max = tmp_flake[i]['flake_exact_bbox']
                    f_mask_height = f_mask_r_max - f_mask_r_min
                    f_mask_width = f_mask_c_max - f_mask_c_min
                    flake_large_bbox = [max(0, f_mask_r_min - int(0.5 * f_mask_height)),
                                        min(imH, f_mask_r_max + int(0.5 * f_mask_height)),
                                        max(0, f_mask_c_min - int(0.5 * f_mask_width)),
                                        min(imW, f_mask_c_max + int(0.5 * f_mask_width))]
                    tmp_flake[i]['flake_large_bbox'] = flake_large_bbox
                    tmp_flake[i]['flake_img'] = im_rgb[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3], :].astype(np.uint8)

                    other_flakes.append(tmp_flake[i])

    # load glue
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
            
            # glue_flakes.extend(tmp_flake)
            for i in range(len(tmp_flake)):
                tmp_flake_name = fname+'-'+str(tmp_flake[i]['flake_id'])
                if tmp_flake[i]['flake_size'] > size_thre and tmp_flake_name not in graphene_names:
                    other_names.append(fname+'-'+str(tmp_flake[i]['flake_id']))
                    # glue_names.append(fname+'-glue-'+str(i))
                    f_mask_r_min, f_mask_r_max, f_mask_c_min, f_mask_c_max = tmp_flake[i]['flake_exact_bbox']
                    f_mask_height = f_mask_r_max - f_mask_r_min
                    f_mask_width = f_mask_c_max - f_mask_c_min
                    flake_large_bbox = [max(0, f_mask_r_min - int(0.5 * f_mask_height)),
                                        min(imH, f_mask_r_max + int(0.5 * f_mask_height)),
                                        max(0, f_mask_c_min - int(0.5 * f_mask_width)),
                                        min(imW, f_mask_c_max + int(0.5 * f_mask_width))]
                    tmp_flake[i]['flake_large_bbox'] = flake_large_bbox
                    tmp_flake[i]['flake_img'] = im_rgb[flake_large_bbox[0]: flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3], :].astype(np.uint8)

                    other_flakes.append(tmp_flake[i])

    num_other = len(other_flakes)
    print("number of others: {}".format(num_other))
    # load features
    for g_flake in other_flakes:
        # other_feats.append(np.concatenate([g_flake['flake_shape_fea'],g_flake['flake_color_fea']]))
        other_feats.append(g_flake['flake_color_fea'])
    other_feats = np.stack(other_feats)

    return graphene_feats, other_feats

def visualize(graphene_feats, other_feats, vis_save_path):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, 0], other_feats[:, 1], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, 0], graphene_feats[:, 1], alpha=0.8, c="red", label='graphene')
    plt.title('Gray and V')
    plt.xlabel('Gray')
    plt.ylabel('V')
    # plt.show()
    plt.savefig(os.path.join(vis_save_path, 'vis_gray_v.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, 0], other_feats[:, 4], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, 0], graphene_feats[:, 4], alpha=0.8, c="red", label='graphene')
    plt.title('Gray and H')
    plt.xlabel('Gray')
    plt.ylabel('H')
    plt.savefig(os.path.join(vis_save_path, 'vis_gray_h.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, 0], other_feats[:, 5], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, 0], graphene_feats[:, 5], alpha=0.8, c="red", label='graphene')
    plt.title('Gray and S')
    plt.xlabel('Gray')
    plt.ylabel('S')
    plt.savefig(os.path.join(vis_save_path, 'vis_gray_s.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, 4], other_feats[:, 5], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, 4], graphene_feats[:, 5], alpha=0.8, c="red", label='graphene')
    plt.title('H and S')
    plt.xlabel('H')
    plt.ylabel('S')
    plt.savefig(os.path.join(vis_save_path, 'vis_h_s.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, 4], other_feats[:, 6], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, 4], graphene_feats[:, 6], alpha=0.8, c="red", label='graphene')
    plt.title('H and V')
    plt.xlabel('H')
    plt.ylabel('V')
    plt.savefig(os.path.join(vis_save_path, 'vis_h_v.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, 5], other_feats[:, 6], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, 5], graphene_feats[:, 6], alpha=0.8, c="red", label='graphene')
    plt.title('S and V')
    plt.xlabel('S')
    plt.ylabel('V')
    plt.savefig(os.path.join(vis_save_path, 'vis_s_v.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, 4], other_feats[:, 3], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, 4], graphene_feats[:, 3], alpha=0.8, c="red", label='graphene')
    plt.title('H and Gray std')
    plt.xlabel('H')
    plt.ylabel('Gray std')
    plt.savefig(os.path.join(vis_save_path, 'vis_h_graystd.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, 4], other_feats[:, 16], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, 4], graphene_feats[:, 16], alpha=0.8, c="red", label='graphene')
    plt.title('H and Gray entropy')
    plt.xlabel('H')
    plt.ylabel('Gray entropy')
    plt.savefig(os.path.join(vis_save_path, 'vis_h_grayentropy.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(other_feats[:, 4], other_feats[:, 5], other_feats[:, 6], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, 4], graphene_feats[:, 5], graphene_feats[:, 6], alpha=0.8, c="red", label='graphene')
    plt.title('H S V')
    ax.set_xlabel('H')
    ax.set_ylabel('S')
    ax.set_zlabel('V')
    plt.savefig(os.path.join(vis_save_path, 'vis_h_s_v.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(other_feats[:, 4], other_feats[:, 3], other_feats[:, 5], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, 4], graphene_feats[:, 3], graphene_feats[:, 5], alpha=0.8, c="red", label='graphene')
    plt.title('H, Gray std, S')
    ax.set_xlabel('H')
    ax.set_ylabel('Gray std')
    ax.set_zlabel('S')
    plt.savefig(os.path.join(vis_save_path, 'vis_h_graystd_s.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(other_feats[:, 4], other_feats[:, 3], other_feats[:, 16], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, 4], graphene_feats[:, 3], graphene_feats[:, 16], alpha=0.8, c="red", label='graphene')
    plt.title('H, Gray std, Gray entropy')
    ax.set_xlabel('H')
    ax.set_ylabel('Gray std')
    ax.set_zlabel('Gray entropy')
    plt.savefig(os.path.join(vis_save_path, 'vis_h_graystd_grayentropy.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(other_feats[:, 10], other_feats[:, 11], other_feats[:, 12], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, 10], graphene_feats[:, 11], graphene_feats[:, 12], alpha=0.8, c="red", label='graphene')
    plt.title('R G B')
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    plt.savefig(os.path.join(vis_save_path, 'vis_r_g_b.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(other_feats[:, 13], other_feats[:, 14], other_feats[:, 15], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, 13], graphene_feats[:, 14], graphene_feats[:, 15], alpha=0.8, c="red", label='graphene')
    plt.title('Rstd Gstd Bstd')
    ax.set_xlabel('Rstd')
    ax.set_ylabel('Gstd')
    ax.set_zlabel('Bstd')
    plt.savefig(os.path.join(vis_save_path, 'vis_rstd_gstd_bstd.png'), dpi=300)
    plt.close()


def main():
    data_path = '../data/data_sep2019/EXP1/09192019 Graphene'
    graphene_path = '../results/data_sep2019_script/labelmat_graphene/EXP1/09192019 Graphene'
    thick_path = '../results/data_sep2019_script/labelmat_thick/EXP1/09192019 Graphene'
    glue_path = '../results/data_sep2019_script/labelmat_glue/EXP1/09192019 Graphene'

    vis_save_path = '../results/data_sep2019_script/visualization'
    if not os.path.exists(vis_save_path):
        os.makedirs(vis_save_path)

    graphene_feats, other_feats = load_data(data_path, graphene_path, thick_path, glue_path)
    visualize(graphene_feats, other_feats, vis_save_path)

if __name__ == '__main__':
    main()