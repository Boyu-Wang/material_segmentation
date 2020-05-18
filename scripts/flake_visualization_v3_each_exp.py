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
from skimage import io, color
# from skimage.filters.rank import entropy
from scipy.stats import entropy
from skimage.morphology import disk
# from multiprocessing import Pool
# from joblib import Parallel, delayed
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

parser = argparse.ArgumentParser(description='graphene classification')
parser.add_argument('--color_fea', default='threesub-contrast-bg-shape', type=str, help='which color feature to use: contrast, ori, both, contrast-bg, ori-bg, both-bg, contrast-bg-shape, innercontrast-bg-shape, subsegment-contrast-bg-shape, twosub-contrast-bg-shape')
parser.add_argument('--n_jobs', default=30, type=int, help='multiprocessing cores')
args = parser.parse_args()


def visualize(graphene_feats, other_feats, vis_save_path):
    contrast_start_idx = 0
    bg_start_idx = 96+16
    shape_start_idx = bg_start_idx + 14 # 30
    
    graphene_color = "lightskyblue"
    others_color = "orange"
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+0], other_feats[:, contrast_start_idx+1], alpha=1, c=others_color, label='other flakes')
    ax.scatter(graphene_feats[:, contrast_start_idx+0], graphene_feats[:, contrast_start_idx+1], alpha=1, c=graphene_color, label='graphene')
    for i, feat_i in enumerate(graphene_feats[:, [contrast_start_idx+0, contrast_start_idx+1]]):
        txt = 'G:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    for i, feat_i in enumerate(other_feats[:, [contrast_start_idx+0, contrast_start_idx+1]]):
        txt = 'O:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))

    plt.title('Contrast Gray and Contrast V')
    plt.xlabel('Contrast Gray')
    plt.ylabel('Contrast V')
    # plt.show()
    plt.savefig(os.path.join(vis_save_path, 'vis_contrastgray_contrastv.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+0], other_feats[:, contrast_start_idx+3], alpha=1, c=others_color, label='other flakes')
    ax.scatter(graphene_feats[:, contrast_start_idx+0], graphene_feats[:, contrast_start_idx+3], alpha=1, c=graphene_color, label='graphene')
    for i, feat_i in enumerate(graphene_feats[:, [contrast_start_idx+0, contrast_start_idx+3]]):
        txt = 'G:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    for i, feat_i in enumerate(other_feats[:, [contrast_start_idx+0, contrast_start_idx+3]]):
        txt = 'O:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))

    plt.title('Contrast Gray and Contrast H')
    plt.xlabel('Contrast Gray')
    plt.ylabel('Contrast H')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrastgray_contrasth.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+0], other_feats[:, contrast_start_idx+4], alpha=1, c=others_color, label='other flakes')
    ax.scatter(graphene_feats[:, contrast_start_idx+0], graphene_feats[:, contrast_start_idx+4], alpha=1, c=graphene_color, label='graphene')
    for i, feat_i in enumerate(graphene_feats[:, [contrast_start_idx+0, contrast_start_idx+4]]):
        txt = 'G:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    for i, feat_i in enumerate(other_feats[:, [contrast_start_idx+0, contrast_start_idx+4]]):
        txt = 'O:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))

    plt.title('Contrast Gray and Contrast S')
    plt.xlabel('Contrast Gray')
    plt.ylabel('Contrast S')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrastgray_contrasts.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+3], other_feats[:, contrast_start_idx+4], alpha=1, c=others_color, label='other flakes')
    ax.scatter(graphene_feats[:, contrast_start_idx+3], graphene_feats[:, contrast_start_idx+4], alpha=1, c=graphene_color, label='graphene')
    for i, feat_i in enumerate(graphene_feats[:, [contrast_start_idx+3, contrast_start_idx+4]]):
        txt = 'G:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    for i, feat_i in enumerate(other_feats[:, [contrast_start_idx+3, contrast_start_idx+4]]):
        txt = 'O:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    plt.title('Contrast H and Contrast S')
    plt.xlabel('Contrast H')
    plt.ylabel('Contrast S')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrasth_contrasts.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+3], other_feats[:, contrast_start_idx+5], alpha=1, c=others_color, label='other flakes')
    ax.scatter(graphene_feats[:, contrast_start_idx+3], graphene_feats[:, contrast_start_idx+5], alpha=1, c=graphene_color, label='graphene')
    for i, feat_i in enumerate(graphene_feats[:, [contrast_start_idx+3, contrast_start_idx+5]]):
        txt = 'G:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    for i, feat_i in enumerate(other_feats[:, [contrast_start_idx+3, contrast_start_idx+5]]):
        txt = 'O:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    plt.title('Contrast H and Contrast V')
    plt.xlabel('Contrast H')
    plt.ylabel('Contrast V')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrasth_contrastv.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+4], other_feats[:, contrast_start_idx+5], alpha=1, c=others_color, label='other flakes')
    ax.scatter(graphene_feats[:, contrast_start_idx+4], graphene_feats[:, contrast_start_idx+5], alpha=1, c=graphene_color, label='graphene')
    for i, feat_i in enumerate(graphene_feats[:, [contrast_start_idx+4, contrast_start_idx+5]]):
        txt = 'G:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    for i, feat_i in enumerate(other_feats[:, [contrast_start_idx+4, contrast_start_idx+5]]):
        txt = 'O:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    plt.title('Contrast S and Contrast V')
    plt.xlabel('Contrast S')
    plt.ylabel('Contrast V')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrasts_contrastv.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+3], other_feats[:, contrast_start_idx+2], alpha=1, c=others_color, label='other flakes')
    ax.scatter(graphene_feats[:, contrast_start_idx+3], graphene_feats[:, contrast_start_idx+2], alpha=1, c=graphene_color, label='graphene')
    for i, feat_i in enumerate(graphene_feats[:, [contrast_start_idx+3, contrast_start_idx+2]]):
        txt = 'G:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    for i, feat_i in enumerate(other_feats[:, [contrast_start_idx+3, contrast_start_idx+2]]):
        txt = 'O:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    plt.title('Contrast H and Contrast Gray std')
    plt.xlabel('Contrast H')
    plt.ylabel('Contrast Gray std')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrasth_contrastgraystd.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+3], other_feats[:, contrast_start_idx+15], alpha=1, c=others_color, label='other flakes')
    ax.scatter(graphene_feats[:, contrast_start_idx+3], graphene_feats[:, contrast_start_idx+15], alpha=1, c=graphene_color, label='graphene')
    for i, feat_i in enumerate(graphene_feats[:, [contrast_start_idx+3, contrast_start_idx+15]]):
        txt = 'G:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    for i, feat_i in enumerate(other_feats[:, [contrast_start_idx+3, contrast_start_idx+15]]):
        txt = 'O:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    plt.title('Contrast H and Contrast Gray entropy')
    plt.xlabel('Contrast H')
    plt.ylabel('Contrast Gray entropy')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrasth_contrastgrayentropy.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(other_feats[:, contrast_start_idx+3], other_feats[:, contrast_start_idx+4], other_feats[:, contrast_start_idx+5], alpha=1, c=others_color, label='other flakes')
    ax.scatter(graphene_feats[:, contrast_start_idx+3], graphene_feats[:, contrast_start_idx+4], graphene_feats[:, contrast_start_idx+5], alpha=1, c=graphene_color, label='graphene')
    for i, feat_i in enumerate(graphene_feats[:, [contrast_start_idx+3, contrast_start_idx+4, contrast_start_idx+5]]):
        txt = 'G:%d'%i
        ax.text(feat_i[0], feat_i[1], feat_i[2], txt)
        # ax.annotate(txt, (feat_i[0], feat_i[1]))
    for i, feat_i in enumerate(other_feats[:, [contrast_start_idx+3, contrast_start_idx+4, contrast_start_idx+5]]):
        txt = 'O:%d'%i
        ax.text(feat_i[0], feat_i[1], feat_i[2], txt)
        # ax.annotate(txt, (feat_i[0], feat_i[1]))
    plt.title('Contrast H, Contrast S, and Contrast V')
    ax.set_xlabel('Contrast H')
    ax.set_ylabel('Contrast S')
    ax.set_zlabel('Contrast V')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrasth_contrasts_contrastv.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(other_feats[:, contrast_start_idx+3], other_feats[:, contrast_start_idx+2], other_feats[:, contrast_start_idx+4], alpha=1, c=others_color, label='other flakes')
    ax.scatter(graphene_feats[:, contrast_start_idx+3], graphene_feats[:, contrast_start_idx+2], graphene_feats[:, contrast_start_idx+4], alpha=1, c=graphene_color, label='graphene')
    for i, feat_i in enumerate(graphene_feats[:, [contrast_start_idx+3, contrast_start_idx+2, contrast_start_idx+4]]):
        txt = 'G:%d'%i
        ax.text(feat_i[0], feat_i[1], feat_i[2], txt)
        # ax.annotate(txt, (feat_i[0], feat_i[1]))
    for i, feat_i in enumerate(other_feats[:, [contrast_start_idx+3, contrast_start_idx+2, contrast_start_idx+4]]):
        txt = 'O:%d'%i
        ax.text(feat_i[0], feat_i[1], feat_i[2], txt)
        # ax.annotate(txt, (feat_i[0], feat_i[1]))
    plt.title('Contrast H, Contrast Gray std, Contrast S')
    ax.set_xlabel('Contrast H')
    ax.set_ylabel('Contrast Gray std')
    ax.set_zlabel('Contrast S')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrasth_contrastgraystd_contrasts.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(other_feats[:, contrast_start_idx+3], other_feats[:, contrast_start_idx+2], other_feats[:, contrast_start_idx+15], alpha=1, c=others_color, label='other flakes')
    ax.scatter(graphene_feats[:, contrast_start_idx+3], graphene_feats[:, contrast_start_idx+2], graphene_feats[:, contrast_start_idx+15], alpha=1, c=graphene_color, label='graphene')
    for i, feat_i in enumerate(graphene_feats[:, [contrast_start_idx+3, contrast_start_idx+2, contrast_start_idx+15]]):
        txt = 'G:%d'%i
        ax.text(feat_i[0], feat_i[1], feat_i[2], txt)
        # ax.annotate(txt, (feat_i[0], feat_i[1]))
    for i, feat_i in enumerate(other_feats[:, [contrast_start_idx+3, contrast_start_idx+2, contrast_start_idx+15]]):
        txt = 'O:%d'%i
        ax.text(feat_i[0], feat_i[1], feat_i[2], txt)
        # ax.annotate(txt, (feat_i[0], feat_i[1]))
    plt.title('Contrast H, Contrast Gray std, Contrast Gray entropy')
    ax.set_xlabel('Contrast H')
    ax.set_ylabel('Contrast Gray std')
    ax.set_zlabel('Contrast Gray entropy')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrasth_contrastgraystd_contrastgrayentropy.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(other_feats[:, contrast_start_idx+9], other_feats[:, contrast_start_idx+10], other_feats[:, contrast_start_idx+11], alpha=1, c=others_color, label='other flakes')
    ax.scatter(graphene_feats[:, contrast_start_idx+9], graphene_feats[:, contrast_start_idx+10], graphene_feats[:, contrast_start_idx+11], alpha=1, c=graphene_color, label='graphene')
    for i, feat_i in enumerate(graphene_feats[:, [contrast_start_idx+9, contrast_start_idx+10, contrast_start_idx+11]]):
        txt = 'G:%d'%i
        ax.text(feat_i[0], feat_i[1], feat_i[2], txt)
        # ax.annotate(txt, (feat_i[0], feat_i[1]))
    for i, feat_i in enumerate(other_feats[:, [contrast_start_idx+9, contrast_start_idx+10, contrast_start_idx+11]]):
        txt = 'O:%d'%i
        ax.text(feat_i[0], feat_i[1], feat_i[2], txt)
        # ax.annotate(txt, (feat_i[0], feat_i[1]))
    plt.title('Contrast R, Contrast G, Contrast B')
    ax.set_xlabel('Contrast R')
    ax.set_ylabel('Contrast G')
    ax.set_zlabel('Contrast B')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrastr_contrastg_contrastb.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(other_feats[:, contrast_start_idx+12], other_feats[:, contrast_start_idx+13], other_feats[:, contrast_start_idx+14], alpha=1, c=others_color, label='other flakes')
    ax.scatter(graphene_feats[:, contrast_start_idx+12], graphene_feats[:, contrast_start_idx+13], graphene_feats[:, contrast_start_idx+14], alpha=1, c=graphene_color, label='graphene')
    for i, feat_i in enumerate(graphene_feats[:, [contrast_start_idx+12, contrast_start_idx+13, contrast_start_idx+14]]):
        txt = 'G:%d'%i
        ax.text(feat_i[0], feat_i[1], feat_i[2], txt)
        # ax.annotate(txt, (feat_i[0], feat_i[1]))
    for i, feat_i in enumerate(other_feats[:, [contrast_start_idx+12, contrast_start_idx+13, contrast_start_idx+14]]):
        txt = 'O:%d'%i
        ax.text(feat_i[0], feat_i[1], feat_i[2], txt)
        # ax.annotate(txt, (feat_i[0], feat_i[1]))
    plt.title('Contrast Rstd, Contrast Gstd, Contrast Bstd')
    ax.set_xlabel('Contrast Rstd')
    ax.set_ylabel('Contrast Gstd')
    ax.set_zlabel('Contrast Bstd')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrastrstd_contrastgstd_contrastbstd.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+0], other_feats[:, bg_start_idx+0], alpha=1, c=others_color, label='other flakes')
    ax.scatter(graphene_feats[:, contrast_start_idx+0], graphene_feats[:, bg_start_idx+0], alpha=1, c=graphene_color, label='graphene')
    for i, feat_i in enumerate(graphene_feats[:, [contrast_start_idx+0, bg_start_idx+0]]):
        txt = 'G:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    for i, feat_i in enumerate(other_feats[:, [contrast_start_idx+0, bg_start_idx+0]]):
        txt = 'O:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    plt.title('Contrast Gray and Background Gray')
    plt.xlabel('Contrast Gray')
    plt.ylabel('Background Gray')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrastgray_bggray.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+0], other_feats[:, bg_start_idx+2], alpha=1, c=others_color, label='other flakes')
    ax.scatter(graphene_feats[:, contrast_start_idx+0], graphene_feats[:, bg_start_idx+2], alpha=1, c=graphene_color, label='graphene')
    for i, feat_i in enumerate(graphene_feats[:, [contrast_start_idx+0, bg_start_idx+2]]):
        txt = 'G:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    for i, feat_i in enumerate(other_feats[:, [contrast_start_idx+0, bg_start_idx+2]]):
        txt = 'O:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    plt.title('Contrast Gray and Background H')
    plt.xlabel('Contrast Gray')
    plt.ylabel('Background H')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrastgray_bgh.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+3], other_feats[:, bg_start_idx+0], alpha=1, c=others_color, label='other flakes')
    ax.scatter(graphene_feats[:, contrast_start_idx+3], graphene_feats[:, bg_start_idx+0], alpha=1, c=graphene_color, label='graphene')
    for i, feat_i in enumerate(graphene_feats[:, [contrast_start_idx+3, bg_start_idx+0]]):
        txt = 'G:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    for i, feat_i in enumerate(other_feats[:, [contrast_start_idx+3, bg_start_idx+0]]):
        txt = 'O:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    plt.title('Contrast H and Background Gray')
    plt.xlabel('Contrast H')
    plt.ylabel('Background Gray')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrasth_bggray.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+3], other_feats[:, bg_start_idx+2], alpha=1, c=others_color, label='other flakes')
    ax.scatter(graphene_feats[:, contrast_start_idx+3], graphene_feats[:, bg_start_idx+2], alpha=1, c=graphene_color, label='graphene')
    for i, feat_i in enumerate(graphene_feats[:, [contrast_start_idx+3, bg_start_idx+2]]):
        txt = 'G:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    for i, feat_i in enumerate(other_feats[:, [contrast_start_idx+3, bg_start_idx+2]]):
        txt = 'O:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    plt.title('Contrast H and Background H')
    plt.xlabel('Contrast H')
    plt.ylabel('Background H')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrasth_bgh.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+0], other_feats[:, shape_start_idx+0], alpha=1, c=others_color, label='other flakes')
    ax.scatter(graphene_feats[:, contrast_start_idx+0], graphene_feats[:, shape_start_idx+0], alpha=1, c=graphene_color, label='graphene')
    for i, feat_i in enumerate(graphene_feats[:, [contrast_start_idx+0, shape_start_idx+0]]):
        txt = 'G:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    for i, feat_i in enumerate(other_feats[:, [contrast_start_idx+0, shape_start_idx+0]]):
        txt = 'O:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    plt.title('Contrast Gray and Len area ratio')
    plt.xlabel('Contrast Gray')
    plt.ylabel('Len area ratio')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrastgray_lenarearatio.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+0], other_feats[:, shape_start_idx+1], alpha=1, c=others_color, label='other flakes')
    ax.scatter(graphene_feats[:, contrast_start_idx+0], graphene_feats[:, shape_start_idx+1], alpha=1, c=graphene_color, label='graphene')
    for i, feat_i in enumerate(graphene_feats[:, [contrast_start_idx+0, shape_start_idx+1]]):
        txt = 'G:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    for i, feat_i in enumerate(other_feats[:, [contrast_start_idx+0, shape_start_idx+1]]):
        txt = 'O:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    plt.title('Contrast Gray and Fracdim')
    plt.xlabel('Contrast Gray')
    plt.ylabel('Fracdim')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrastgray_fracdim.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+3], other_feats[:, shape_start_idx+0], alpha=1, c=others_color, label='other flakes')
    ax.scatter(graphene_feats[:, contrast_start_idx+3], graphene_feats[:, shape_start_idx+0], alpha=1, c=graphene_color, label='graphene')
    for i, feat_i in enumerate(graphene_feats[:, [contrast_start_idx+3, shape_start_idx+0]]):
        txt = 'G:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    for i, feat_i in enumerate(other_feats[:, [contrast_start_idx+3, shape_start_idx+0]]):
        txt = 'O:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    plt.title('Contrast H and Len area ratio')
    plt.xlabel('Contrast H')
    plt.ylabel('Len area ratio')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrasth_lenarearatio.png'), dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(other_feats[:, contrast_start_idx+3], other_feats[:, shape_start_idx+1], alpha=1, c=others_color, label='other flakes')
    ax.scatter(graphene_feats[:, contrast_start_idx+3], graphene_feats[:, shape_start_idx+1], alpha=1, c=graphene_color, label='graphene')
    for i, feat_i in enumerate(graphene_feats[:, [contrast_start_idx+3, shape_start_idx+1]]):
        txt = 'G:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    for i, feat_i in enumerate(other_feats[:, [contrast_start_idx+3, shape_start_idx+1]]):
        txt = 'O:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))
    plt.title('Contrast H and Fracdim')
    plt.xlabel('Contrast H')
    plt.ylabel('Fracdim')
    plt.savefig(os.path.join(vis_save_path, 'vis_contrasth_fracdim.png'), dpi=300)
    plt.close()


def visualize_pca(graphene_feats, other_feats, vis_save_path):
    # graphene_to_remove = [35, 36, 28, 24, 20]
    # graphene_feats = np.array([graphene_feats[id] for id in range(graphene_feats.shape[0]) if id not in graphene_to_remove])
    # PCA decomposition of features
    num_graphene = graphene_feats.shape[0]
    num_others = other_feats.shape[0]

    graphene_color = "lightskyblue"
    others_color = "orange"
    all_feats = np.concatenate([graphene_feats, other_feats], axis=0)
    all_feats = StandardScaler().fit_transform(all_feats)

    pca_2 = PCA(n_components=2)
    all_feats_2 = pca_2.fit_transform(all_feats)
    # for id in range(num_graphene):
    #     if all_feats_2[id, 1] > 4:
    #         print('y>4', id) 
    #     if all_feats_2[id, 0] > 5:
    #         print('x>5', id)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(all_feats_2[num_graphene:, 0], all_feats_2[num_graphene:, 1], alpha=1, c=others_color, label='other flakes')
    ax.scatter(all_feats_2[:num_graphene, 0], all_feats_2[:num_graphene, 1], alpha=1, c=graphene_color, label='graphene')
    for i, feat_i in enumerate(all_feats_2):
        if i < num_graphene:
            txt = 'G:%d'%i
        else:
            txt = 'O:%d'%i
        ax.annotate(txt, (feat_i[0], feat_i[1]))

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
    for i, feat_i in enumerate(all_feats_3):
        if i < num_graphene:
            txt = 'G:%d'%i
        else:
            txt = 'O:%d'%i
        # ax.annotate(txt, (feat_i[0], feat_i[1], feat_i[2]))
        ax.text(feat_i[0], feat_i[1], feat_i[2], txt)

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
    for i, feat_i in enumerate(all_feats_3):
        if i < num_graphene:
            txt = 'G:%d'%i
        else:
            txt = 'O:%d'%i
        # ax.annotate(txt, (feat_i[0], feat_i[2], feat_i[1]))
        ax.text(feat_i[0], feat_i[2], feat_i[1], txt)

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
    for i, feat_i in enumerate(all_feats_3):
        if i < num_graphene:
            txt = 'G:%d'%i
        else:
            txt = 'O:%d'%i
        # ax.annotate(txt, (feat_i[1], feat_i[0], feat_i[2]))
        ax.text(feat_i[1], feat_i[0], feat_i[1], txt)

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
    for i, feat_i in enumerate(all_feats_3):
        if i < num_graphene:
            txt = 'G:%d'%i
        else:
            txt = 'O:%d'%i
        # ax.annotate(txt, (feat_i[1], feat_i[2], feat_i[0]))
        ax.text(feat_i[1], feat_i[2], feat_i[0], txt)

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
    for i, feat_i in enumerate(all_feats_3):
        if i < num_graphene:
            txt = 'G:%d'%i
        else:
            txt = 'O:%d'%i
        # ax.annotate(txt, (feat_i[2], feat_i[0], feat_i[1]))
        ax.text(feat_i[2], feat_i[0], feat_i[1], txt)

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
    for i, feat_i in enumerate(all_feats_3):
        if i < num_graphene:
            txt = 'G:%d'%i
        else:
            txt = 'O:%d'%i
        # ax.annotate(txt, (feat_i[2], feat_i[1], feat_i[0]))
        ax.text(feat_i[2], feat_i[1], feat_i[0], txt)

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

# read from the annotation
def readdb(all_dbname):
    conn = sqlite3.connect(all_dbname)
    c = conn.cursor()
    c.execute('SELECT imflakeid, thicklabel, qualitylabel FROM annotab')
    db = c.fetchall()

    num_graphene = 0
    num_junk = 0
    num_thin = 0
    num_thick = 0
    num_multi = 0
    num_others = 0
    oriname_flakeids = []
    for i in range(len(db)):
        imflakeid = db[i][0]
        flake_id = int(imflakeid.split('_')[3].split('-')[1])
        flake_oriname = imflakeid.split('_', 5)[5]
        label = db[i][1]
        qualitylabel = db[i][2]
        if label == 'junk':
            num_junk += 1
            num_others += 1
            label_id = 0
        elif label == 'thick':
            num_thick += 1
            num_others += 1
            # label_id = 1
            label_id = 0
        elif label == 'thin':
            num_thin += 1
            num_others += 1
            # label_id = 2
            label_id = 0
        elif label == 'multi':
            num_multi += 1
            num_others += 1
            # label_id = 3
            label_id = 0
        elif label == 'graphene':
            num_graphene += 1
            # label_id = 4
            label_id = 1
        # else:
        #     raise NotImplementedError

        oriname_flakeids.append([flake_oriname+'-'+str(flake_id), flake_oriname, flake_id, label_id])
        
    print('junk: %d, thick: %d, thin: %d, multi: %d, graphene: %d'%(num_junk, num_thick, num_thin, num_multi, num_graphene))
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
                    img_fea = list(tmp_flake[i]['flake_contrast_color_fea'])
                    if isinstance(img_fea[-1], np.ndarray):
                        img_fea = img_fea[:-1] + list(img_fea[-1])
                elif 'both' in color_fea:
                    img_fea = np.concatenate([tmp_flake[i]['flake_color_fea'], tmp_flake[i]['flake_contrast_color_fea']])
                else:
                    img_fea = np.empty([0])
                    # raise NotImplementedError

                if 'subsegment' in color_fea:
                    img_fea = np.concatenate([img_fea, tmp_flake[i]['subsegment_features']])
                elif 'threesub' in color_fea:
                    img_fea = np.concatenate([img_fea, tmp_flake[i]['subsegment_features_3']])
                elif 'locsub3' in color_fea:
                    img_fea = np.concatenate([img_fea, tmp_flake[i]['subsegment_features_3_loc_1']])
                elif 'twosub' in color_fea:
                    img_fea = np.concatenate([img_fea, tmp_flake[i]['subsegment_features_2']])
                elif 'foursub' in color_fea:
                    img_fea = np.concatenate([img_fea, tmp_flake[i]['subsegment_features_4']])


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

  
def process_anno_data(data_path, result_path, vis_path, annotation_path):
    # # read all annotations, find all exp names
    # all_anno_data = readdb(annotation_path)

    exp_names = os.listdir(data_path)
    exp_names = [ename for ename in exp_names if ename[0] not in ['.', '_']]
    exp_names.sort()
    for exp_name in exp_names:
        subexp_names = os.listdir(os.path.join(data_path, exp_name))
        subexp_names = [sname for sname in subexp_names if sname[0] not in ['.', '_']]
        subexp_names = [sname for sname in subexp_names if os.path.isdir(os.path.join(data_path, exp_name, sname))]
        subexp_names.sort()
        for sname in subexp_names:
            # get features
            path_to_save = os.path.join(vis_path, exp_name, sname)
            feat_save_path = os.path.join(path_to_save, 'features_withname.p')
            if os.path.exists(feat_save_path):
                continue
            sub_anno_path = os.path.join(annotation_path, exp_name, sname, 'anno_user-youngjae.db')
            if not os.path.exists(sub_anno_path):
                print('not exist annotation! ', sub_anno_path)
                continue
            sub_anno_data = readdb(sub_anno_path)
            # sub_anno_data = [all_anno_i[2:] for all_anno_i in all_anno_data if all_anno_i[0] == exp_name and all_anno_i[1] == sname]
            print(exp_name, sname, len(sub_anno_data))
            if len(sub_anno_data) > 0:
                labeled_feats, unlabeled_feats, labeled_flake_name_ids, labels = load_one_dir(sub_anno_data, os.path.join(data_path, exp_name, sname), os.path.join(result_path, exp_name, sname))
                
                feats ={}
                feats['labeled_feats'] = labeled_feats
                feats['unlabeled_feats'] = unlabeled_feats
                feats['labeled_labels'] = labels
                feats['labeled_flake_name_ids'] = labeled_flake_name_ids

                if not os.path.exists(path_to_save):
                    os.makedirs(path_to_save)
                pickle.dump(feats, open(feat_save_path, 'wb'))


def vis_flakes(graphene_name_ids, other_name_ids, data_path, flake_path, vis_save_path):
    Parallel(n_jobs=args.n_jobs)(delayed(vis_flakes_helper)(data_path, flake_path, graphene_name_ids[i], os.path.join(vis_save_path, 'graphene-%d.png'%i)) for i in range(len(graphene_name_ids)))
    Parallel(n_jobs=args.n_jobs)(delayed(vis_flakes_helper)(data_path, flake_path, other_name_ids[i], os.path.join(vis_save_path, 'others-%d.png'%i)) for i in range(len(other_name_ids)))



def vis_flakes_helper(data_path, flake_path, flake_name_id, flake_save_name):
    if os.path.exists(flake_save_name):
        return
    
    fname, flake_id = flake_name_id.rsplit('-', 1)
    flake_id = int(flake_id)
    image = Image.open(os.path.join(data_path, fname[:-2] + 'tiff'))

    im_rgb = np.array(image).astype('float')
    im_tosave = im_rgb.astype(np.uint8)
    im_tosave_withcontour = im_rgb.astype(np.uint8)        
    imH, imW, _ = im_tosave.shape
    
    tmp_flake = pickle.load(open(os.path.join(flake_path, fname), 'rb'))
    # image_labelmap = tmp_flake['image_labelmap']
    flakes = tmp_flake['flakes']

    color = (255, 255, 255)
    contours = flakes[flake_id]['flake_contour_loc']
    contours = np.expand_dims(np.flip(contours), 1).astype(np.int32)
    im_tosave_withcontour = cv2.drawContours(im_tosave_withcontour, contours, -1, color, 2)

    # get patch of the image
    flake_large_bbox = flakes[flake_id]['flake_large_bbox']
    flake_r = flake_large_bbox[1] - flake_large_bbox[0]
    flake_c = flake_large_bbox[3] - flake_large_bbox[2]
    flake_r = int(max(1.2 * max(flake_r, flake_c), 100))
    flake_large_bbox[0] = max(0, flake_large_bbox[0]-flake_r)
    flake_large_bbox[1] = min(imH, flake_large_bbox[1]+flake_r)
    flake_large_bbox[2] = max(0, flake_large_bbox[2]-flake_r)
    flake_large_bbox[3] = min(imW, flake_large_bbox[3]+flake_r)
    im_tosave_withcontour = im_tosave_withcontour[flake_large_bbox[0]:flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3], :]
    im_tosave = im_tosave[flake_large_bbox[0]:flake_large_bbox[1], flake_large_bbox[2]:flake_large_bbox[3], :]

    # stick withcontour and without contour together
    imH, imW, _ = im_tosave.shape
    black_strip = np.zeros([imH, max(2, int(imW*0.03)), 3], dtype=np.int)
    im_tosave = np.concatenate([im_tosave_withcontour, black_strip, im_tosave], 1)

    cv2.imwrite(flake_save_name, np.flip(im_tosave, 2))
    
def main():
    data_path = '../data/data_111x_individual/'
    result_path = '../results/data_111x_individual_script/mat_2.0_100'
    vis_path = '../results/data_111x_individual_script/visualization_moreanno_v2_colorfea-%s'%args.color_fea
    annotation_path = '../data/anno_graphene_v3_youngjae'

    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    process_anno_data(data_path, result_path, vis_path, annotation_path)
    
    exp_names = os.listdir(result_path)
    exp_names = [ename for ename in exp_names if ename[0] not in ['.', '_']]
    exp_names.sort()
    for exp_name in exp_names:
        subexp_names = os.listdir(os.path.join(result_path, exp_name))
        subexp_names = [sname for sname in subexp_names if sname[0] not in ['.', '_']]
        subexp_names = [sname for sname in subexp_names if os.path.isdir(os.path.join(result_path, exp_name, sname))]
        subexp_names.sort()
        for sname in subexp_names:
            path_to_save = os.path.join(vis_path, exp_name, sname)
            feat_save_path = os.path.join(path_to_save, 'features_withname.p')
            if os.path.exists(feat_save_path):
                feats = pickle.load(open(feat_save_path, 'rb'))            
                all_labeled_feats = feats['labeled_feats']
                all_labeled_feats[np.isnan(all_labeled_feats)]= 0
                all_labels = feats['labeled_labels']
                labeled_flake_name_ids = feats['labeled_flake_name_ids']
                graphene_ids = [x for x in range(len(all_labels)) if all_labels[x] == 1]
                others_ids = [x for x in range(len(all_labels)) if all_labels[x] == 0]
                graphene_feats = all_labeled_feats[graphene_ids,:]
                other_feats = all_labeled_feats[others_ids, :]
                graphene_name_ids = [labeled_flake_name_ids[ids] for ids in graphene_ids]
                other_name_ids = [labeled_flake_name_ids[ids] for ids in others_ids]

                # print(graphene_name_ids)
                print(exp_name, sname)
                print("number of graphene: {}".format(graphene_feats.shape[0]))
                print("number of others: {}".format(other_feats.shape[0]))

                print("number of feature dimensions: {}".format(graphene_feats.shape[1]))

                vis_save_path = os.path.join(vis_path, exp_name, sname)
                
                visualize(graphene_feats, other_feats, vis_save_path)
                visualize_pca(graphene_feats, other_feats, vis_save_path)
                vis_flakes(graphene_name_ids, other_name_ids, os.path.join(data_path, exp_name, sname), os.path.join(result_path, exp_name, sname), vis_save_path)

if __name__ == '__main__':
    main()