import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
from torch.autograd import Function
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data as DD

import numpy as np
from PIL import Image
import pickle
import os

# 0: others. 1: graphene.
class completeLabelDataLoader(DD.Dataset):
    def __init__(self, data_type, img_dir_prefixs, mask_dir_prefixs, contrast_dir_prefixs, bg_dir_prefixs, input_type, input_with_mask, subset='train', transform=None, img_size=256, color_fea='contrast-bg-shape'):
        super(completeLabelDataLoader, self).__init__()
        self.all_img_names = []
        self.all_mask_names = []
        self.all_contrast_names = []
        self.all_bg_names = []
        self.all_graphene_feats = {}
        self.all_other_feats = {}
        self.all_data_types = []
        data_type = data_type.split('-')

        for (d_type, img_dir_prefix, mask_dir_prefix, contrast_dir_prefix, bg_dir_prefix) in zip(data_type, img_dir_prefixs, mask_dir_prefixs, contrast_dir_prefixs, bg_dir_prefixs):

            data_path = os.path.join(img_dir_prefix, subset)
            mask_path = os.path.join(mask_dir_prefix, subset)
            contrast_path = os.path.join(contrast_dir_prefix, subset)
            bg_path = os.path.join(bg_dir_prefix, subset)

            img_names = os.listdir(data_path)
            img_names = [iname for iname in img_names if iname[0] not in ['.', '_']]
            es_img_names = [os.path.join(data_path, n) for n in img_names]
            es_mask_names = [os.path.join(mask_path, n) for n in img_names]
            es_contrast_names = [os.path.join(contrast_path, n) for n in img_names]
            es_bg_names = [os.path.join(bg_path, n) for n in img_names]
            self.all_img_names.extend(es_img_names)
            self.all_mask_names.extend(es_mask_names)
            self.all_contrast_names.extend(es_contrast_names)
            self.all_bg_names.extend(es_bg_names)

            # load label and handcrafted features
            if d_type == 'sep':
                feature_path = '/nfs/bigpupil/boyu/Projects/material_segmentation/results/data_sep2019_script/graphene_classifier_binary_fea-%s/feature.p'%color_fea
            elif d_type == 'oct':
                feature_path = '/nfs/bigpupil/boyu/Projects/material_segmentation/results/10222019G wtih Suji_script/center_patch_500_500/graphene_classifier_binary_fea-%s/feature.p'%color_fea
            elif d_type == 'nov':
                # feature_path = '/nfs/bigpupil/boyu/Projects/material_segmentation/results/data_111x_individual_script/graphene_classifier_with_clean_anno_colorfea-%s/feature_separate.p'%color_fea
                feature_path = '/nfs/bigpupil/boyu/Projects/material_segmentation/results/data_111x_individual_script/graphene_classifier_with_clean_anno_doublecheck_colorfea-%s/feature_separate.p'%color_fea
            else:
                raise NotImplementedError
            feats = pickle.load(open(feature_path, 'rb'))
            graphene_feats = feats['graphene_feats']
            # remove noise in graphene annotations
            if d_type == 'nov':
                # graphene_to_remove = [35, 36, 28, 24, 20]
                graphene_to_remove = []
                # graphene_feats = np.array([graphene_feats[id] for id in range(graphene_feats.shape[0]) if id not in graphene_to_remove])
                # also remove the name list
                self.all_img_names = [nm for nm in self.all_img_names if not ('graphene' in nm.rsplit('/', 1)[1] and int(nm.rsplit('/', 1)[1].split('.')[0].split('_')[1]) in graphene_to_remove)]
                self.all_mask_names = [nm for nm in self.all_mask_names if not ('graphene' in nm.rsplit('/', 1)[1] and int(nm.rsplit('/', 1)[1].split('.')[0].split('_')[1]) in graphene_to_remove)]
                self.all_contrast_names = [nm for nm in self.all_contrast_names if not ('graphene' in nm.rsplit('/', 1)[1] and int(nm.rsplit('/', 1)[1].split('.')[0].split('_')[1]) in graphene_to_remove)]
                self.all_bg_names = [nm for nm in self.all_bg_names if not ('graphene' in nm.rsplit('/', 1)[1] and int(nm.rsplit('/', 1)[1].split('.')[0].split('_')[1]) in graphene_to_remove)]
            other_feats = feats['other_feats']
            self.all_graphene_feats[d_type] = graphene_feats
            self.all_other_feats[d_type] = other_feats
            self.all_data_types.extend([d_type]*len(img_names))

        self.transforms = transform
        self.img_size = img_size
        self.input_type = input_type
        self.input_with_mask = input_with_mask
        self.num_imgs = len(self.all_img_names)
        print('num of images: %d'%(self.num_imgs))

    def __getitem__(self, index):
        img_name = self.all_img_names[index]
        mask_name = self.all_mask_names[index]
        contrast_name = self.all_contrast_names[index]
        bg_name = self.all_bg_names[index]

        img = Image.open(img_name)
        mask_img = Image.open(mask_name)
        contrast_img = Image.open(contrast_name)
        bg_img = Image.open(bg_name)

        img_all = []
        if 'rgb' in self.input_type:
            im_rgb = np.array(img)
            img_all.append(im_rgb)
        if 'gray' in self.input_type:
            im_gray = np.array(img.convert('L', (0.2989, 0.5870, 0.1140, 0)))
            img_all.append(np.expand_dims(im_gray, 2))
        if 'hsv' in self.input_type:
            im_hsv = np.array(img.convert('HSV'))
            img_all.append(im_hsv)
        if 'contrast' in self.input_type:
            im_contrast = np.array(contrast_img)
            img_all.append(im_contrast)
        if 'bg' in self.input_type:
            im_bg = np.array(bg_img)
            img_all.append(im_bg)
        if self.input_with_mask:
            img_all.append(np.expand_dims(mask_img, 2))
        img_all = np.concatenate(img_all, 2)

        if self.transforms:
            img_all = self.transforms(img_all)
            mask_img = self.transforms(mask_img)

        iname = img_name.rsplit('/', 1)[1]
        # print(iname)
        idx = int(iname.split('.')[0].split('_')[1])
        d_type = self.all_data_types[index]
        # label is 0,1
        if 'graphene' in iname:
            label = 1
            feat = np.array(self.all_graphene_feats[d_type][idx])
        elif 'others' in iname:
            label = 0
            feat = np.array(self.all_other_feats[d_type][idx])
        else:
            raise NotImplementedError
        return img_all, mask_img, label, feat, iname

    def __len__(self):
        return self.num_imgs


# load all data
# input_type: 'rgb', 'hsv', 'gray', 'rgb-hsv-gray'
# input_with_mask: True, False
class AllDataLoader(DD.Dataset):
    def __init__(self, data_type, img_dir_prefixs, mask_dir_prefixs, contrast_dir_prefixs, bg_dir_prefixs, input_type, input_with_mask, transform=None, img_size=256):
        super(AllDataLoader, self).__init__()

        # get all image names
        self.all_img_names = []
        self.all_mask_names = []
        self.all_contrast_names = []
        self.all_bg_names = []

        for (img_dir_prefix, mask_dir_prefix, contrast_dir_prefix, bg_dir_prefix) in zip(img_dir_prefixs, mask_dir_prefixs, contrast_dir_prefixs, bg_dir_prefixs):
            exp_names = os.listdir(img_dir_prefix)
            exp_names = [ename for ename in exp_names if ename[0] not in ['.', '_']]
            for ename in exp_names:
                sub_names = os.listdir(os.path.join(img_dir_prefix, ename))
                sub_names = [sname for sname in sub_names if sname[0] not in ['.', '_']]
                for sname in sub_names:
                    pth = os.path.join(img_dir_prefix, ename, sname)
                    mask_pth = os.path.join(mask_dir_prefix, ename, sname)
                    contrast_pth = os.path.join(contrast_dir_prefix, ename, sname)
                    bg_path = os.path.join(bg_dir_prefix, ename, sname)
                    img_names = os.listdir(pth)
                    img_names = [iname for iname in img_names if iname[0] not in ['.', '_']]
                    es_img_names = [os.path.join(pth, n) for n in img_names]
                    es_mask_names = [os.path.join(mask_pth, n) for n in img_names]
                    es_contrast_names = [os.path.join(contrast_pth, n) for n in img_names]
                    es_bg_names = [os.path.join(bg_path, n) for n in img_names]
                    self.all_img_names.extend(es_img_names)
                    self.all_mask_names.extend(es_mask_names)
                    self.all_contrast_names.extend(es_contrast_names)
                    self.all_bg_names.extend(es_bg_names)
        
        self.transforms = transform
        self.img_size = img_size
        self.num_imgs = len(self.all_img_names)
        self.input_type = input_type
        self.input_with_mask = input_with_mask
        print('num of images: %d'%(self.num_imgs))

    def __getitem__(self, index):
        img_name = self.all_img_names[index]
        mask_name = self.all_mask_names[index]
        contrast_name = self.all_contrast_names[index]
        bg_name = self.all_bg_names[index]

        img = Image.open(img_name)
        mask_img = Image.open(mask_name)
        contrast_img = Image.open(contrast_name)
        bg_img = Image.open(bg_name)

        img_all = []
        if 'rgb' in self.input_type:
            im_rgb = np.array(img)
            img_all.append(im_rgb)
        if 'gray' in self.input_type:
            im_gray = np.array(img.convert('L', (0.2989, 0.5870, 0.1140, 0)))
            img_all.append(np.expand_dims(im_gray, 2))
        if 'hsv' in self.input_type:
            im_hsv = np.array(img.convert('HSV'))
            img_all.append(im_hsv)
        if 'contrast' in self.input_type:
            im_contrast = np.array(contrast_img)
            img_all.append(im_contrast)
        if 'bg' in self.input_type:
            im_bg = np.array(bg_img)
            img_all.append(im_bg)
        if self.input_with_mask:
            img_all.append(np.expand_dims(mask_img,2))
        img_all = np.concatenate(img_all, 2)

        if self.transforms:
            img_all = self.transforms(img_all)
            mask_img = self.transforms(mask_img)

        # ../../results/data_jan2019_script/ae_mask/YoungJaeShinSamples/5/tile_x006_y003-1.png
        # if mask_img.sum() == 0:
            # print(mask_name)
        return img_all, mask_img

    def __len__(self):
        return self.num_imgs
