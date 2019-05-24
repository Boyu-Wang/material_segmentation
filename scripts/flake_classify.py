"""
Test the classifier (thick/thin/glue) on other set of experiments
For classification, only consider regions larger than 28*28=784.
Red boundary means thick, gree boundary means thin, black boundary means glue, white bounday means regions smaller than 784
"""


import numpy as np
from PIL import Image
import cv2
import argparse
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='flake segmentation')
parser.add_argument('--exp_sid', default=0, type=int, help='exp start id')
parser.add_argument('--exp_eid', default=1, type=int, help='exp end id')
parser.add_argument('--subexp_sid', default=0, type=int, help='subexp start id')
parser.add_argument('--subexp_eid', default=3, type=int, help='subexp end id')
# parser.add_argument('--img_sid', default=0, type=int)
# parser.add_argument('--img_eid', default=294, type=int)
parser.add_argument('--n_jobs', default=30, type=int, help='multiprocessing cores')
# parser.add_argument('--c_sid', default=0, type=int, help='subexp start id')
# parser.add_argument('--c_eid', default=400, type=int, help='subexp end id')

args = parser.parse_args()

labelmaps = {'thin': 1, 'thick': 0, 'glue': 2, 'mixed cluster': 3, 'others': 4}

hyperparams = { 'size_thre': 784, # after detect foreground regions, filter them based on its size. (784=28*28 corresponds to around 5 um regions)
                'clf_method': 'linearsvm', # which classifier to use (linear): 'rigde', 'linearsvm'
                }

def classify_one_image(img_name, info_name, classifier, norm_fea, new_img_save_path):
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
    im_tosave = im_rgb.astype(np.uint8)
    for i in range(num_flakes):
        flake_size = flakes[i]['flake_size']
        color = (255, 255, 255)
        contours = flakes[i]['flake_contour_loc']
        contours = np.expand_dims(np.flip(contours), 1).astype(np.int32)
        if flake_size > hyperparams['size_thre']:
            large_flake_idxs.append(i)
            img_fea = np.concatenate([flakes[i]['flake_shape_fea'], flakes[i]['flake_color_fea']])
            img_fea = np.expand_dims(img_fea, 0)
            img_fea -= norm_fea['mean']
            img_fea /= norm_fea['std']
            pred_cls = classifier.predict(img_fea)
            if pred_cls == 0:
                # thick, red
                color = (255, 0, 0)
            elif pred_cls == 1:
                # thin, green
                color = (0, 255, 0)
            elif pred_cls == 2:
                # glue, black
                color = (0, 0, 0)

        im_tosave = cv2.drawContours(im_tosave, contours, -1, color, 2)

    cv2.imwrite(new_img_save_path, np.flip(im_tosave, 2))

    plt.close()


# process one sub exp, read all the data, and do clustering
def classify_one_subexp(subexp_dir, rslt_dir, result_classify_save_path, norm_fea, classifier):
    img_names = os.listdir(subexp_dir)
    img_names = [n_i for n_i in img_names if n_i[0]  not in ['.', '_']]
    img_names.sort()
    # print('process ' + subexp_dir)
    print('n images: %d'%(len(img_names)))

    Parallel(n_jobs=args.n_jobs)(delayed(classify_one_image)(os.path.join(subexp_dir, img_names[i]), os.path.join(rslt_dir, img_names[i][:-4] + '.p'),
                       classifier, norm_fea, os.path.join(result_classify_save_path, img_names[i])) for i in range(len(img_names)))
    # for i in range(len(img_names)):
    #     classify_one_image(os.path.join(subexp_dir, img_names[i]), os.path.join(rslt_dir, img_names[i][:-4]+'.p'), classifier, norm_fea, os.path.join(result_classify_save_path, img_names[i]))


def main():
    data_path = '../data/data_jan2019'
    result_path = '../results/data_jan2019_script/mat'

    result_classify_path = '../results/data_jan2019_script/classify'

    # norm_fea = pickle.load(open('../results/data_jan2019_script/flakeglue_clf_incomplete/YoungJaeShinSamples/4/normfea.p', 'rb'))
    # classifier = pickle.load(open('../results/data_jan2019_script/flakeglue_clf_incomplete/YoungJaeShinSamples/4/feanorm_classifier-linearsvm-0.100000.p','rb'))
    norm_fea = pickle.load(open('../results/data_jan2019_script/thickthinglue_clf_complete/YoungJaeShinSamples/4/normfea.p', 'rb'))
    classifier = pickle.load(open('../results/data_jan2019_script/thickthinglue_clf_complete/YoungJaeShinSamples/4/feanorm_classifier-linearsvm-0.500000.p','rb'))

    exp_names = os.listdir(data_path)
    exp_names = [ename for ename in exp_names if ename[0] not in ['.', '_']]
    exp_names.sort()

    # print(exp_names)
    # exp_names = exp_names[args.exp_sid: args.exp_eid]

    for d in range(args.exp_sid, args.exp_eid):
        exp_name = exp_names[d]
        subexp_names = os.listdir(os.path.join(data_path, exp_name))
        subexp_names = [sname for sname in subexp_names if sname[0] not in ['.', '_']]
        subexp_names = [sname for sname in subexp_names if os.path.isdir(os.path.join(data_path, exp_name, sname))]
        subexp_names.sort()
        # print(subexp_names)

        # process each subexp
        for s_d in range(args.subexp_sid, min(len(subexp_names), args.subexp_eid)):
            sname = subexp_names[s_d]
            print('classify for images under directory: %s' %(os.path.join(data_path, exp_name, sname)))
            result_classify_save_path = os.path.join(result_classify_path, exp_name, sname)
            if not os.path.exists(result_classify_save_path):
                os.makedirs(result_classify_save_path)

            classify_one_subexp(os.path.join(data_path, exp_name, sname), os.path.join(result_path, exp_name, sname), result_classify_save_path, norm_fea, classifier)




if __name__ == '__main__':
    main()






