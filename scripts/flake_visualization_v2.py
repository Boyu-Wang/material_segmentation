"""
Visualization of multiple features for mix bg graphene and non-graphene.
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

def visualize(graphene_feats, other_feats, vis_save_path):
    contrast_start_idx = 0
    bg_start_idx = 16
    shape_start_idx = bg_start_idx + 14 # 30
    
    # innercontrast_start_idx = 16
    # bg_start_idx = innercontrast_start_idx + 16 # 32
    # shape_start_idx = bg_start_idx + 14 # 46

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+0], other_feats[:, contrast_start_idx+1], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, contrast_start_idx+0], graphene_feats[:, contrast_start_idx+1], alpha=0.8, c="red", label='graphene')
    plt.title('Contrast Gray and Contrast V')
    plt.xlabel('Contrast Gray')
    plt.ylabel('Contrast V')
    # plt.show()
    plt.savefig(os.path.join(vis_save_path, 'vis_contrastgray_contrastv.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+0], other_feats[:, contrast_start_idx+3], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, contrast_start_idx+0], graphene_feats[:, contrast_start_idx+3], alpha=0.8, c="red", label='graphene')
    plt.title('Contrast Gray and Contrast H')
    plt.xlabel('Contrast Gray')
    plt.ylabel('Contrast H')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrastgray_contrasth.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+0], other_feats[:, contrast_start_idx+4], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, contrast_start_idx+0], graphene_feats[:, contrast_start_idx+4], alpha=0.8, c="red", label='graphene')
    plt.title('Contrast Gray and Contrast S')
    plt.xlabel('Contrast Gray')
    plt.ylabel('Contrast S')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrastgray_contrasts.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+3], other_feats[:, contrast_start_idx+4], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, contrast_start_idx+3], graphene_feats[:, contrast_start_idx+4], alpha=0.8, c="red", label='graphene')
    plt.title('Contrast H and Contrast S')
    plt.xlabel('Contrast H')
    plt.ylabel('Contrast S')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrasth_contrasts.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+3], other_feats[:, contrast_start_idx+5], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, contrast_start_idx+3], graphene_feats[:, contrast_start_idx+5], alpha=0.8, c="red", label='graphene')
    plt.title('Contrast H and Contrast V')
    plt.xlabel('Contrast H')
    plt.ylabel('Contrast V')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrasth_contrastv.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+4], other_feats[:, contrast_start_idx+5], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, contrast_start_idx+4], graphene_feats[:, contrast_start_idx+5], alpha=0.8, c="red", label='graphene')
    plt.title('Contrast S and Contrast V')
    plt.xlabel('Contrast S')
    plt.ylabel('Contrast V')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrasts_contrastv.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+3], other_feats[:, contrast_start_idx+2], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, contrast_start_idx+3], graphene_feats[:, contrast_start_idx+2], alpha=0.8, c="red", label='graphene')
    plt.title('Contrast H and Contrast Gray std')
    plt.xlabel('Contrast H')
    plt.ylabel('Contrast Gray std')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrasth_contrastgraystd.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+3], other_feats[:, contrast_start_idx+15], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, contrast_start_idx+3], graphene_feats[:, contrast_start_idx+15], alpha=0.8, c="red", label='graphene')
    plt.title('Contrast H and Contrast Gray entropy')
    plt.xlabel('Contrast H')
    plt.ylabel('Contrast Gray entropy')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrasth_contrastgrayentropy.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(other_feats[:, contrast_start_idx+3], other_feats[:, contrast_start_idx+4], other_feats[:, contrast_start_idx+5], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, contrast_start_idx+3], graphene_feats[:, contrast_start_idx+4], graphene_feats[:, contrast_start_idx+5], alpha=0.8, c="red", label='graphene')
    plt.title('Contrast H, Contrast S, and Contrast V')
    ax.set_xlabel('Contrast H')
    ax.set_ylabel('Contrast S')
    ax.set_zlabel('Contrast V')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrasth_contrasts_contrastv.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(other_feats[:, contrast_start_idx+3], other_feats[:, contrast_start_idx+2], other_feats[:, contrast_start_idx+4], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, contrast_start_idx+3], graphene_feats[:, contrast_start_idx+2], graphene_feats[:, contrast_start_idx+4], alpha=0.8, c="red", label='graphene')
    plt.title('Contrast H, Contrast Gray std, Contrast S')
    ax.set_xlabel('Contrast H')
    ax.set_ylabel('Contrast Gray std')
    ax.set_zlabel('Contrast S')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrasth_contrastgraystd_contrasts.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(other_feats[:, contrast_start_idx+3], other_feats[:, contrast_start_idx+2], other_feats[:, contrast_start_idx+15], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, contrast_start_idx+3], graphene_feats[:, contrast_start_idx+2], graphene_feats[:, contrast_start_idx+15], alpha=0.8, c="red", label='graphene')
    plt.title('Contrast H, Contrast Gray std, Contrast Gray entropy')
    ax.set_xlabel('Contrast H')
    ax.set_ylabel('Contrast Gray std')
    ax.set_zlabel('Contrast Gray entropy')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrasth_contrastgraystd_contrastgrayentropy.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(other_feats[:, contrast_start_idx+9], other_feats[:, contrast_start_idx+10], other_feats[:, contrast_start_idx+11], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, contrast_start_idx+9], graphene_feats[:, contrast_start_idx+10], graphene_feats[:, contrast_start_idx+11], alpha=0.8, c="red", label='graphene')
    plt.title('Contrast R, Contrast G, Contrast B')
    ax.set_xlabel('Contrast R')
    ax.set_ylabel('Contrast G')
    ax.set_zlabel('Contrast B')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrastr_contrastg_contrastb.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(other_feats[:, contrast_start_idx+12], other_feats[:, contrast_start_idx+13], other_feats[:, contrast_start_idx+14], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, contrast_start_idx+12], graphene_feats[:, contrast_start_idx+13], graphene_feats[:, contrast_start_idx+14], alpha=0.8, c="red", label='graphene')
    plt.title('Contrast Rstd, Contrast Gstd, Contrast Bstd')
    ax.set_xlabel('Contrast Rstd')
    ax.set_ylabel('Contrast Gstd')
    ax.set_zlabel('Contrast Bstd')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrastrstd_contrastgstd_contrastbstd.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+0], other_feats[:, bg_start_idx+0], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, contrast_start_idx+0], graphene_feats[:, bg_start_idx+0], alpha=0.8, c="red", label='graphene')
    plt.title('Contrast Gray and Background Gray')
    plt.xlabel('Contrast Gray')
    plt.ylabel('Background Gray')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrastgray_bggray.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+0], other_feats[:, bg_start_idx+2], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, contrast_start_idx+0], graphene_feats[:, bg_start_idx+2], alpha=0.8, c="red", label='graphene')
    plt.title('Contrast Gray and Background H')
    plt.xlabel('Contrast Gray')
    plt.ylabel('Background H')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrastgray_bgh.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+3], other_feats[:, bg_start_idx+0], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, contrast_start_idx+3], graphene_feats[:, bg_start_idx+0], alpha=0.8, c="red", label='graphene')
    plt.title('Contrast H and Background Gray')
    plt.xlabel('Contrast H')
    plt.ylabel('Background Gray')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrasth_bggray.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+3], other_feats[:, bg_start_idx+2], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, contrast_start_idx+3], graphene_feats[:, bg_start_idx+2], alpha=0.8, c="red", label='graphene')
    plt.title('Contrast H and Background H')
    plt.xlabel('Contrast H')
    plt.ylabel('Background H')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrasth_bgh.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+0], other_feats[:, shape_start_idx+0], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, contrast_start_idx+0], graphene_feats[:, shape_start_idx+0], alpha=0.8, c="red", label='graphene')
    plt.title('Contrast Gray and Len area ratio')
    plt.xlabel('Contrast Gray')
    plt.ylabel('Len area ratio')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrastgray_lenarearatio.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+0], other_feats[:, shape_start_idx+1], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, contrast_start_idx+0], graphene_feats[:, shape_start_idx+1], alpha=0.8, c="red", label='graphene')
    plt.title('Contrast Gray and Fracdim')
    plt.xlabel('Contrast Gray')
    plt.ylabel('Fracdim')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrastgray_fracdim.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+3], other_feats[:, shape_start_idx+0], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, contrast_start_idx+3], graphene_feats[:, shape_start_idx+0], alpha=0.8, c="red", label='graphene')
    plt.title('Contrast H and Len area ratio')
    plt.xlabel('Contrast H')
    plt.ylabel('Len area ratio')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrasth_lenarearatio.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+3], other_feats[:, shape_start_idx+1], alpha=0.6, c="blue", label='others')
    ax.scatter(graphene_feats[:, contrast_start_idx+3], graphene_feats[:, shape_start_idx+1], alpha=0.8, c="red", label='graphene')
    plt.title('Contrast H and Fracdim')
    plt.xlabel('Contrast H')
    plt.ylabel('Fracdim')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrasth_fracdim.png'), dpi=300)
    plt.close()


    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(other_feats[:, innercontrast_start_idx+0], other_feats[:, innercontrast_start_idx+3], alpha=0.6, c="blue", label='others')
    # ax.scatter(graphene_feats[:, innercontrast_start_idx+0], graphene_feats[:, innercontrast_start_idx+3], alpha=0.8, c="red", label='graphene')
    # plt.title('InnerContrast Gray and InnerContrast H')
    # plt.xlabel('InnerContrast Gray')
    # plt.ylabel('InnerContrast H')
    # plt.savefig(os.path.join(vis_save_path, 'vis_innercontrastgray_innercontrasth.png'), dpi=300)
    # plt.close()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(other_feats[:, innercontrast_start_idx+0], other_feats[:, innercontrast_start_idx+4], alpha=0.6, c="blue", label='others')
    # ax.scatter(graphene_feats[:, innercontrast_start_idx+0], graphene_feats[:, innercontrast_start_idx+4], alpha=0.8, c="red", label='graphene')
    # plt.title('InnerContrast Gray and InnerContrast S')
    # plt.xlabel('InnerContrast Gray')
    # plt.ylabel('InnerContrast S')
    # plt.savefig(os.path.join(vis_save_path, 'vis_innercontrastgray_innercontrasts.png'), dpi=300)
    # plt.close()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(other_feats[:, innercontrast_start_idx+0], other_feats[:, innercontrast_start_idx+1], alpha=0.6, c="blue", label='others')
    # ax.scatter(graphene_feats[:, innercontrast_start_idx+0], graphene_feats[:, innercontrast_start_idx+1], alpha=0.8, c="red", label='graphene')
    # plt.title('InnerContrast Gray and InnerContrast V')
    # plt.xlabel('InnerContrast Gray')
    # plt.ylabel('InnerContrast V')
    # plt.savefig(os.path.join(vis_save_path, 'vis_innercontrastgray_innercontrastv.png'), dpi=300)
    # plt.close()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(other_feats[:, innercontrast_start_idx+4], other_feats[:, innercontrast_start_idx+1], alpha=0.6, c="blue", label='others')
    # ax.scatter(graphene_feats[:, innercontrast_start_idx+4], graphene_feats[:, innercontrast_start_idx+1], alpha=0.8, c="red", label='graphene')
    # plt.title('InnerContrast S and InnerContrast V')
    # plt.xlabel('InnerContrast S')
    # plt.ylabel('InnerContrast V')
    # plt.savefig(os.path.join(vis_save_path, 'vis_innercontrasts_innercontrastv.png'), dpi=300)
    # plt.close()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(other_feats[:, innercontrast_start_idx+3], other_feats[:, innercontrast_start_idx+2], alpha=0.6, c="blue", label='others')
    # ax.scatter(graphene_feats[:, innercontrast_start_idx+3], graphene_feats[:, innercontrast_start_idx+2], alpha=0.8, c="red", label='graphene')
    # plt.title('InnerContrast H and InnerContrast Gray std')
    # plt.xlabel('InnerContrast H')
    # plt.ylabel('InnerContrast Gray std')
    # plt.savefig(os.path.join(vis_save_path, 'vis_innercontrasth_innercontrastgraystd.png'), dpi=300)
    # plt.close()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(other_feats[:, innercontrast_start_idx+3], other_feats[:, innercontrast_start_idx+15], alpha=0.6, c="blue", label='others')
    # ax.scatter(graphene_feats[:, innercontrast_start_idx+3], graphene_feats[:, innercontrast_start_idx+15], alpha=0.8, c="red", label='graphene')
    # plt.title('InnerContrast H and InnerContrast Gray entropy')
    # plt.xlabel('InnerContrast H')
    # plt.ylabel('InnerContrast Gray entropy')
    # plt.savefig(os.path.join(vis_save_path, 'vis_innercontrasth_innercontrastgrayentropy.png'), dpi=300)
    # plt.close()
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(other_feats[:, contrast_start_idx+0], other_feats[:, innercontrast_start_idx+0], alpha=0.6, c="blue", label='others')
    # ax.scatter(graphene_feats[:, contrast_start_idx+0], graphene_feats[:, innercontrast_start_idx+0], alpha=0.8, c="red", label='graphene')
    # plt.title('Contrast Gray and InnerContrast Gray')
    # plt.xlabel('Contrast Gray')
    # plt.ylabel('InnerContrast Gray')
    # plt.savefig(os.path.join(vis_save_path, 'vis_contrastgray_innercontrastgray.png'), dpi=300)
    # plt.close()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(other_feats[:, contrast_start_idx+3], other_feats[:, innercontrast_start_idx+3], alpha=0.6, c="blue", label='others')
    # ax.scatter(graphene_feats[:, contrast_start_idx+3], graphene_feats[:, innercontrast_start_idx+3], alpha=0.8, c="red", label='graphene')
    # plt.title('Contrast H and InnerContrast H')
    # plt.xlabel('Contrast H')
    # plt.ylabel('InnerContrast H')
    # plt.savefig(os.path.join(vis_save_path, 'vis_contrasth_innercontrasth.png'), dpi=300)
    # plt.close()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(other_feats[:, contrast_start_idx+4], other_feats[:, innercontrast_start_idx+4], alpha=0.6, c="blue", label='others')
    # ax.scatter(graphene_feats[:, contrast_start_idx+4], graphene_feats[:, innercontrast_start_idx+4], alpha=0.8, c="red", label='graphene')
    # plt.title('Contrast S and InnerContrast S')
    # plt.xlabel('Contrast S')
    # plt.ylabel('InnerContrast S')
    # plt.savefig(os.path.join(vis_save_path, 'vis_contrasts_innercontrasts.png'), dpi=300)
    # plt.close()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(other_feats[:, contrast_start_idx+1], other_feats[:, innercontrast_start_idx+1], alpha=0.6, c="blue", label='others')
    # ax.scatter(graphene_feats[:, contrast_start_idx+1], graphene_feats[:, innercontrast_start_idx+1], alpha=0.8, c="red", label='graphene')
    # plt.title('Contrast V and InnerContrast V')
    # plt.xlabel('Contrast V')
    # plt.ylabel('InnerContrast V')
    # plt.savefig(os.path.join(vis_save_path, 'vis_contrastv_innercontrastv.png'), dpi=300)
    # plt.close()

    

def visualize_pca(graphene_feats, other_feats, vis_save_path):
    graphene_to_remove = [35, 36, 28, 24, 20]
    graphene_feats = np.array([graphene_feats[id] for id in range(graphene_feats.shape[0]) if id not in graphene_to_remove])
    # PCA decomposition of features
    num_graphene = graphene_feats.shape[0]
    num_others = other_feats.shape[0]

    graphene_color = "lightskyblue"
    others_color = "orange"
    all_feats = np.concatenate([graphene_feats, other_feats], axis=0)
    all_feats = StandardScaler().fit_transform(all_feats)

    pca_2 = PCA(n_components=2)
    all_feats_2 = pca_2.fit_transform(all_feats)
    for id in range(num_graphene):
        if all_feats_2[id, 1] > 4:
            print('y>4', id) 
        if all_feats_2[id, 0] > 5:
            print('x>5', id)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(all_feats_2[num_graphene:, 0], all_feats_2[num_graphene:, 1], alpha=1, c=others_color, label='other flakes')
    ax.scatter(all_feats_2[:num_graphene, 0], all_feats_2[:num_graphene, 1], alpha=1, c=graphene_color, label='graphene')
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect((x1-x0)/(y1-y0))
    plt.title('Feature projection')
    ax.legend()
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.tick_params(axis='both', which='both', length=0)
    plt.savefig(os.path.join(vis_save_path, 'vis_pca_2.png'), dpi=300)
    plt.close()

    pca_3 = PCA(n_components=3)
    all_feats_3 = pca_3.fit_transform(all_feats)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(all_feats_3[num_graphene:, 0], all_feats_3[num_graphene:, 1], all_feats_3[num_graphene:, 2], alpha=1, c=others_color, label='other flakes')
    ax.scatter(all_feats_3[:num_graphene, 0], all_feats_3[:num_graphene, 2], all_feats_3[:num_graphene, 2], alpha=1, c=graphene_color, label='graphene')
    plt.title('Feature projection')
    ax.legend()
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.setp(ax.get_zticklabels(), visible=False)
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    # ax.set_aspect('equal', 'box')
    ax.tick_params(axis='both', which='both', length=0)
    plt.savefig(os.path.join(vis_save_path, 'vis_pca_3_012.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(all_feats_3[num_graphene:, 0], all_feats_3[num_graphene:, 2], all_feats_3[num_graphene:, 1], alpha=1, c=others_color, label='other flakes')
    ax.scatter(all_feats_3[:num_graphene, 0], all_feats_3[:num_graphene, 2], all_feats_3[:num_graphene, 1], alpha=1, c=graphene_color, label='graphene')
    plt.title('Feature projection')
    ax.legend()
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.setp(ax.get_zticklabels(), visible=False)
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    # ax.set_aspect('equal', 'box')
    ax.tick_params(axis='both', which='both', length=0)
    plt.savefig(os.path.join(vis_save_path, 'vis_pca_3_021.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(all_feats_3[num_graphene:, 1], all_feats_3[num_graphene:, 0], all_feats_3[num_graphene:, 2], alpha=1, c=others_color, label='other flakes')
    ax.scatter(all_feats_3[:num_graphene, 1], all_feats_3[:num_graphene, 0], all_feats_3[:num_graphene, 2], alpha=1, c=graphene_color, label='graphene')
    plt.title('Feature projection')
    ax.legend()
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.setp(ax.get_zticklabels(), visible=False)
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    # ax.set_aspect('equal', 'box')
    ax.tick_params(axis='both', which='both', length=0)
    plt.savefig(os.path.join(vis_save_path, 'vis_pca_3_102.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(all_feats_3[num_graphene:, 1], all_feats_3[num_graphene:, 2], all_feats_3[num_graphene:, 0], alpha=1, c=others_color, label='other flakes')
    ax.scatter(all_feats_3[:num_graphene, 1], all_feats_3[:num_graphene, 2], all_feats_3[:num_graphene, 0], alpha=1, c=graphene_color, label='graphene')
    plt.title('Feature projection')
    ax.legend()
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.setp(ax.get_zticklabels(), visible=False)
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    # ax.set_aspect('equal', 'box')
    ax.tick_params(axis='both', which='both', length=0)
    plt.savefig(os.path.join(vis_save_path, 'vis_pca_3_120.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(all_feats_3[num_graphene:, 2], all_feats_3[num_graphene:, 0], all_feats_3[num_graphene:, 1], alpha=1, c=others_color, label='other flakes')
    ax.scatter(all_feats_3[:num_graphene, 2], all_feats_3[:num_graphene, 0], all_feats_3[:num_graphene, 1], alpha=1, c=graphene_color, label='graphene')
    plt.title('Feature projection')
    ax.legend()
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.setp(ax.get_zticklabels(), visible=False)
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    # ax.set_aspect('equal', 'box')
    ax.tick_params(axis='both', which='both', length=0)
    plt.savefig(os.path.join(vis_save_path, 'vis_pca_3_201.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(all_feats_3[num_graphene:, 2], all_feats_3[num_graphene:, 1], all_feats_3[num_graphene:, 0], alpha=1, c=others_color, label='other flakes')
    ax.scatter(all_feats_3[:num_graphene, 2], all_feats_3[:num_graphene, 1], all_feats_3[:num_graphene, 0], alpha=1, c=graphene_color, label='graphene')
    plt.title('Feature projection')
    ax.legend()
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.setp(ax.get_zticklabels(), visible=False)
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    # ax.set_aspect('equal', 'box')
    ax.tick_params(axis='both', which='both', length=0)
    plt.savefig(os.path.join(vis_save_path, 'vis_pca_3_210.png'), dpi=300)
    plt.close()

    
def main():
    # # load sep features
    # classifier_save_path_v1 = '../results/data_sep2019_script/graphene_classifier_binary_fea-%s'%'contrast-bg-shape'
    # feat_save_path_v1 = os.path.join(classifier_save_path_v1, 'feature.p')
    # if os.path.exists(feat_save_path_v1):
    #     feats = pickle.load(open(feat_save_path_v1, 'rb'))
    # else:
    #     raise NotImplementedError
    # graphene_feats_v1 = feats['graphene_feats']
    # other_feats_v1 = feats['other_feats']

    # # load oct features
    # classifier_save_path_v2 = '../results/10222019G wtih Suji_script/center_patch_500_500/graphene_classifier_binary_fea-%s'%'contrast-bg-shape'
    # feat_save_path_v2 = os.path.join(classifier_save_path_v2, 'feature.p')
    # if os.path.exists(feat_save_path_v2):
    #     feats = pickle.load(open(feat_save_path_v2, 'rb'))
    # else:
    #     raise NotImplementedError
    # graphene_feats_v2 = feats['graphene_feats']
    # other_feats_v2 = feats['other_feats']

    # other_feats = np.concatenate([other_feats_v1, other_feats_v2], axis=0)
    # graphene_feats = np.concatenate([graphene_feats_v1, graphene_feats_v2], axis=0)

    # load clean annoatation data
    # fea_type = 'contrast-bg-shape'
    # fea_type = 'innercontrast-bg-shape'
    # fea_type = 'subsegment-bg-shape'
    fea_type = 'subsegment-contrast-bg-shape'
    # classifier_save_path = '../results/data_111x_individual_script/graphene_classifier_with_clean_anno_colorfea-%s'%'innercontrast-bg-shape'
    classifier_save_path = '../results/data_111x_individual_script/graphene_classifier_with_clean_anno_colorfea-%s'%fea_type
    feat_save_path = os.path.join(classifier_save_path, 'features_withname.p')
    if os.path.exists(feat_save_path):
        feats = pickle.load(open(feat_save_path, 'rb'))
    else:
        raise NotImplementedError
    all_labeled_feats = feats['labeled_feats']
    all_labels = feats['labeled_labels']
    graphene_ids = [x for x in range(len(all_labels)) if all_labels[x] == 1]
    others_ids = [x for x in range(len(all_labels)) if all_labels[x] == 0]
    graphene_feats = all_labeled_feats[graphene_ids,:]
    other_feats = all_labeled_feats[others_ids, :]

    print("number of graphene: {}".format(graphene_feats.shape[0]))
    print("number of others: {}".format(other_feats.shape[0]))

    print("number of feature dimensions: {}".format(graphene_feats.shape[1]))

    # vis_save_path = '../results/data_sep-oct_script/visualization'
    vis_save_path = '../results/data_111x_individual_script/clean_anno_visualization_%s'%fea_type
    if not os.path.exists(vis_save_path):
        os.makedirs(vis_save_path)

    # visualize(graphene_feats, other_feats, vis_save_path)
    visualize_pca(graphene_feats, other_feats, vis_save_path)

if __name__ == '__main__':
    main()