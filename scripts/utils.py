"""
Utility functions for robust fit

By: Boyu Wang (boywang@cs.stonybrook.edu)
Created Data: 21 Feb 2019
Last Modified Date: 21 Feb 2019
"""

import numpy as np
from PIL import Image
import robustfit
import cv2


# def test():
#     # im = skimage.io.imread('../data/data_jan2019/EXPT2TransferPressure/HighPressure/tile_x004_y012.tif')
# im = Image.open('../data/data_jan2019/EXPT2TransferPressure/HighPressure/tile_x004_y012.tif')
# # save conversion as matlab
# im_gray = np.array(im.convert('L', (0.2989, 0.5870, 0.1140, 0))).astype('float')
# im_hsv = np.array(im.convert('HSV')).astype('float')
# # im = skimage.io.imread('tile_x004_y012.tif')
# # im_gray = skimage.color.rgb2gray(im) * 255
# # im_hsv = skimage.color.rgb2hsv(im) * 255
# imH, imW = im_gray.shape
# [C, R] = np.meshgrid(np.arange(0, imW), np.arange(0, imH))
# Y = np.reshape(R, [-1]) / (imH - 1) - 0.5
# X = np.reshape(C, [-1]) / (imW - 1) - 0.5
# A = np.stack([np.ones([imH*imW]), X*X, Y*Y, X*Y, X, Y, ], axis=1)
# brob_rlm_model = robustfit.RLM()
# brob_rlm_model.fit(A, np.reshape(im_gray, [-1]))


# def test2():
#     import numpy as np
#     import skimage
#     import skimage.io
#     import statsmodels.api as sm


#     # brob_rlm_model = sm.RLM(y, np.stack([x, np.ones([10])], axis=1), M=sm.robust.norms.TukeyBiweight())
#     # brob_rlm_results = brob_rlm_model.fit()
#     import robustfit
#     import numpy as np
#     y = [8.5377, 7.8339, 1.7412, 2.8622, 0.3188, -3.3077, -4.4336, -5.6574, -4.4216, 0]
#     y = np.array(y)
#     x = np.arange(1,11)
#     brob_rlm_model = robustfit.RLM()
#     brob_rlm_model.fit(np.stack([x, np.ones([10])], axis=1), y)


# perform robust on 3 channels: gray, h, v, and add their results together
def perform_robustfit_multichannel(im_hsv, im_gray, im_thre=10, size_thre=10):
    # im_hsv = im_hsv.astype('float')
    # im_gray = im_gray.astype('float')
    im_ghs = np.concatenate([np.expand_dims(im_gray,2), im_hsv[:,:,:2]], axis=2)
    imH, imW, _ = im_ghs.shape
    [C, R] = np.meshgrid(np.arange(0, imW), np.arange(0, imH))
    Y = np.reshape(R, [-1]) / (imH - 1) - 0.5
    X = np.reshape(C, [-1]) / (imW - 1) - 0.5
    A = np.stack([np.ones([imH*imW]), X*X, Y*Y, X*Y, X, Y, ], axis=1)
    res_map = np.zeros([imH, imW])
    for c in range(3):
        im_c = im_ghs[:,:,c]
        brob_rlm_model = robustfit.RLM()
        brob_rlm_model.fit(A, np.reshape(im_c, [-1]))
        if c == 0:
            w = 1
            ref_s = brob_rlm_model.sigma
        else:
            w = ref_s / brob_rlm_model.sigma
        res_map = res_map + w * np.abs(np.reshape(brob_rlm_model.resid, [imH, imW]))
    outlier_map = res_map>im_thre
    outlier_map = outlier_map.astype(np.uint8)

    # connected component detection
    nCC, image_labelmap, _, flake_centroids = cv2.connectedComponentsWithStats(outlier_map)
    # flake_centroid: [m, 2] array, indicates row, column of the centroid
    flake_centroids = np.flip(flake_centroids, 1)
    flake_centroids = flake_centroids[1:].astype('int')

    _, flake_sizes = np.unique(image_labelmap, return_counts=True)
    # remove the background size
    flake_sizes = flake_sizes[1:]
    if size_thre > 0:
        # remove small connect component
        large_flakes = flake_sizes > size_thre
        large_flake_idxs = np.nonzero(large_flakes)[0]
        new_image_labels = np.zeros([imH, imW])
        cnt = 0
        for idx in large_flake_idxs:
            cnt += 1
            new_image_labels[image_labelmap==idx+1] = cnt
        image_labelmap = new_image_labels
        num_flakes = large_flakes.sum()
        flake_centroids = flake_centroids[large_flake_idxs]
        flake_sizes = flake_sizes[large_flake_idxs]
    else:
        num_flakes = nCC - 1

    return res_map, image_labelmap, flake_centroids, flake_sizes, num_flakes


# perform robust on gray
def perform_robustfit(im_gray, im_thre=3, size_thre=0):
    # im_gray = im_gray.astype('float')
    imH, imW = im_gray.shape
    [C, R] = np.meshgrid(np.arange(0, imW), np.arange(0, imH))
    Y = np.reshape(R, [-1]) / (imH - 1) - 0.5
    X = np.reshape(C, [-1]) / (imW - 1) - 0.5
    A = np.stack([np.ones([imH*imW]), X*X, Y*Y, X*Y, X, Y, ], axis=1)
    # res_map = np.zeros([imH, imW])

    brob_rlm_model = robustfit.RLM()
    brob_rlm_model.fit(A, np.reshape(im_gray, [-1]))
    res_map = np.abs(np.reshape(brob_rlm_model.resid, [imH, imW]))
    outlier_map = res_map > im_thre * brob_rlm_model.sigma
    outlier_map = outlier_map.astype(np.uint8)

    # connected component detection
    nCC, image_labelmap, _, flake_centroids = cv2.connectedComponentsWithStats(outlier_map)
    # flake_centroid: [m, 2] array, indicates row, column of the centroid
    flake_centroids = np.flip(flake_centroids, 1)
    flake_centroids = flake_centroids[1:].astype('int')

    _, flake_sizes = np.unique(image_labelmap, return_counts=True)
    # remove the background size
    flake_sizes = flake_sizes[1:]
    if size_thre > 0:
        # remove small connect component
        large_flakes = flake_sizes > size_thre
        large_flake_idxs = np.nonzero(large_flakes)[0]
        new_image_labels = np.zeros([imH, imW])
        cnt = 0
        for idx in large_flake_idxs:
            cnt += 1
            new_image_labels[image_labelmap==idx+1] = cnt
        image_labelmap = new_image_labels
        num_flakes = large_flakes.sum()
        flake_centroids = flake_centroids[large_flake_idxs+1]
        flake_sizes = flake_sizes[large_flake_idxs+1]
    else:
        num_flakes = nCC - 1

    return res_map, image_labelmap, flake_centroids, flake_sizes, num_flakes


# use stats model
# slow, do not have same results as matlab
def perform_robustfit_v2(im_gray, im_thre=3, size_thre=0):
    import statsmodels.api as sm
    # im_gray = im_gray.astype('float')
    imH, imW = im_gray.shape
    [C, R] = np.meshgrid(np.arange(0, imW), np.arange(0, imH))
    Y = np.reshape(R, [-1]) / (imH - 1) - 0.5
    X = np.reshape(C, [-1]) / (imW - 1) - 0.5
    A = np.stack([np.ones([imH*imW]), X*X, Y*Y, X*Y, X, Y, ], axis=1)
    # res_map = np.zeros([imH, imW])

    brob_rlm_model = sm.RLM(np.reshape(im_gray, [-1]), A, M=sm.robust.norms.TukeyBiweight())
    brob_rlm_results = brob_rlm_model.fit(cov='H2')

    # get the estimate of sigma
    # https://www.mathworks.com/help/stats/robustfit.html
    sigma = np.median(np.abs(brob_rlm_results.resid)) / 0.6745

    # brob_rlm_model = robustfit.RLM()
    # brob_rlm_model.fit(A, np.reshape(im_gray, [-1]))
    res_map = np.abs(np.reshape(brob_rlm_results.resid, [imH, imW]))
    # outlier_map = res_map > im_thre * brob_rlm_model.sigma
    outlier_map = res_map > im_thre * sigma
    outlier_map = outlier_map.astype(np.uint8)

    # connected component detection
    nCC, image_labelmap, _, flake_centroids = cv2.connectedComponentsWithStats(outlier_map)
    # flake_centroid: [m, 2] array, indicates row, column of the centroid
    flake_centroids = np.flip(flake_centroids, 1)
    flake_centroids = flake_centroids[1:].astype('int')

    _, flake_sizes = np.unique(image_labelmap, return_counts=True)
    # remove the background size
    flake_sizes = flake_sizes[1:]
    if size_thre > 0:
        # remove small connect component
        large_flakes = flake_sizes > size_thre
        large_flake_idxs = np.nonzero(large_flakes)[0]
        new_image_labels = np.zeros([imH, imW])
        cnt = 0
        for idx in large_flake_idxs:
            cnt += 1
            new_image_labels[image_labelmap==idx+1] = cnt
        image_labelmap = new_image_labels
        num_flakes = large_flakes.sum()
        flake_centroids = flake_centroids[large_flake_idxs+1]
        flake_sizes = flake_sizes[large_flake_idxs+1]
    else:
        num_flakes = nCC - 1

    return res_map, image_labelmap, flake_centroids, flake_sizes, num_flakes


# Z is a binary image
def fractal_dimension(Z):
    # Only for 2d image
    assert(len(Z.shape) == 2)

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])

    # # Transform Z into a binary array
    # Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]
