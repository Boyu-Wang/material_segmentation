"""
Generate synthetic dataset.
"""
import math
import numpy as np
import cv2
from PIL import Image
from skimage import io, color
import robustfit
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool
from joblib import Parallel, delayed
import copy

def generate_hv(thickness, max_thickness=100):
    """
    Given thickness, get the corresponding h and v values.
    Both H and V ossilate with thickness. 
    """
    H_max = 1
    H_min = 0
    V_min = 0.2
    V_max = 1
    H = math.sin(1.5 * thickness) * thickness * 0.1
    V = math.sin(2.5 * thickness) * thickness * 0.1

    H = H / (2 * max_thickness * 0.1) * (H_max - H_min) + H_min
    V = V / (2 * max_thickness * 0.1) * (V_max - V_min) + V_min
    return H, V

def generate_polygon(row_min, row_max, column_min, column_max, im_size=(256, 256)):
    flag = True
    while flag:
        n = np.random.randint(3, 7)
        x = np.random.randint(column_min, column_max, n)
        y = np.random.randint(row_min, row_max, n)

        ##computing the (or a) 'center point' of the polygon
        center_point = [np.sum(x)/n, np.sum(y)/n]

        angles = np.arctan2(x-center_point[0],y-center_point[1])

        ##sorting the points:
        sort_tups = sorted([(i,j,k) for i,j,k in zip(x,y,angles)], key = lambda t: t[2])

        ##making sure that there are no duplicates:
        if len(sort_tups) == len(set(sort_tups)):
            flag = False
            # raise Exception('two equal coordinates -- exiting')

    x,y,angles = zip(*sort_tups)
    x = list(x)
    y = list(y)
    pts = np.array([x, y]).transpose()

    mask = np.zeros(im_size)
    mask = cv2.fillPoly(mask, [pts], 1)
    mask = mask.astype(bool)
    return mask


def generate_bg(img_name):
    # img_name = '../data/data_sep2019/EXP1/09192019 Graphene/6 graphene-1.tiff'

    image = Image.open(img_name)
    im_rgb = np.array(image).astype('float')
    im_gray = np.array(image.convert('L', (0.2989, 0.5870, 0.1140, 0))).astype('float')
    imH, imW = im_gray.shape
    # im_hsv = color.rgb2hsv(im_rgb)
    # im_hsv[:,:,2] = im_hsv[:,:,2]/255.0
    bg_rgb = []
    [C, R] = np.meshgrid(np.arange(0, imW), np.arange(0, imH))
    Y = np.reshape(R, [-1]) / (imH - 1) - 0.5
    X = np.reshape(C, [-1]) / (imW - 1) - 0.5
    A = np.stack([np.ones([imH*imW]), X*X, Y*Y, X*Y, X, Y, ], axis=1)
    for c in range(3):
        brob_rlm_model = robustfit.RLM()
        brob_rlm_model.fit(A, np.reshape(im_rgb[:,:,c], [-1]))
        pred_map = np.reshape(brob_rlm_model.predict(A), [imH, imW])
        bg_rgb.append(pred_map)
    bg_rgb = np.stack(bg_rgb, axis=2).astype(np.uint8)
    bg_hsv = color.rgb2hsv(bg_rgb)
    # bg_hsv[:,:,2] = bg_hsv[:,:,2]/255.0

    # bg_rgb: [0, 255]. bg_hs [0, 1.0], v [0, 255]
    return bg_rgb, bg_hsv



def generate_flake(im_size):
    flake_width = np.random.randint(30, 100)
    flake_height = np.random.randint(30, 100)
    row_min = np.random.randint(0, im_size[0]-flake_height)
    column_min = np.random.randint(0, im_size[1]-flake_width)
    flake_mask = generate_polygon(row_min, row_min+flake_height, column_min, column_min+flake_width, im_size)
    thickness = np.random.randint(1, 30)
    # print(thickness)
    # H, V = generate_hv(thickness)

    # H_noise = np.random.rand(im_size[0], im_size[1]) / 100 * flake_mask.astype(float)
    # V_noise = np.random.rand(im_size[0], im_size[1]) / 100 * flake_mask.astype(float)

    # H = H_noise + H
    # V = V_noise + V

    # im_hsv = copy.copy(bg_hsv)
    # # print(flake_mask)
    # # print(bg_hsv[flake_mask, 0])
    # im_hsv[flake_mask, 0] = H[flake_mask]
    # im_hsv[flake_mask, 2] = V[flake_mask]
    # return im_hsv

    thickness_mask = flake_mask.astype(float) * thickness

    return thickness_mask





def generate_one_img(bg_hsv, im_size, save_dir, img_idx):
    num_flakes = np.random.randint(1, 10)
    bgH, bgW, _ = bg_hsv.shape
    im_h = np.random.randint(0, bgH-im_size[0])
    im_w = np.random.randint(0, bgW-im_size[1])
    im_hsv = copy.copy(bg_hsv[im_h:im_h+im_size[0], im_w:im_w+im_size[1], :])
    # for i in range(num_flakes):
    #     im_hsv = generate_flake(im_hsv, im_size)

    # generate thickness
    thickness_mask = np.zeros([im_size[0], im_size[1]])
    for i in range(num_flakes):
        thickness_mask += generate_flake(im_size)

    # map thickness to hsv
    u_thick = np.unique(thickness_mask)
    for thickness in u_thick:
        if thickness != 0:
            mask = thickness_mask == thickness
            H, V = generate_hv(thickness)
            im_hsv[mask, 0] = H
            im_hsv[mask, 2] = V

    im_rgb = color.hsv2rgb(im_hsv) * 255
    im_rgb = im_rgb.astype(np.uint8)

    # im_rgb = cv2.GaussianBlur(im_rgb,(3,3),0)

    cv2.imwrite(os.path.join(save_dir, '{:08x}.png'.format(img_idx)), np.flip(im_rgb, 2))




def main():
    save_dir = '../data/synthetic_v2/'
    img_save_dir = os.path.join(save_dir, 'images')
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    bg_name = os.path.join(save_dir, 'bg_v1.tiff')
    if not os.path.exists(bg_name):
        img_name = '../data/data_sep2019/EXP1/09192019 Graphene/6 graphene-1.tiff'
        bg_rgb, bg_hsv = generate_bg(img_name)    
        cv2.imwrite(bg_name, np.flip(bg_rgb, 2))
    else:
        bg_rgb = cv2.imread(bg_name)
        bg_rgb = np.flip(bg_rgb, 2)
        bg_hsv = color.rgb2hsv(bg_rgb)
        # bg_hsv[:,:,2] = bg_hsv[:,:,2]/255.0

    im_size = (256, 256)

    num_img = 50

    # generate_one_img(bg_hsv, im_size, img_save_dir, 1)
    Parallel(n_jobs=4)(delayed(generate_one_img)(bg_hsv, im_size, img_save_dir, i) for i in range(num_img))





    
if __name__=='__main__':
    # main()

    bg_name = '../data/synthetic_v2/bg_2.tiff'
    img_name = '../data/data_jan2019/YoungJaeShinSamples/4/tile_x001_y016.tif'
    bg_rgb, bg_hsv = generate_bg(img_name)    
    cv2.imwrite(bg_name, np.flip(bg_rgb, 2))



