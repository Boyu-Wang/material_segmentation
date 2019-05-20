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

# load the data with annotation
class LabelDataLoader(DD.Dataset):
    def __init__(self, img_dir_prefix, mask_dir_prefix, contrast_dir_prefix, input_type, input_with_mask, subset='train', transform=None, img_size=256):
        super(LabelDataLoader, self).__init__()
        label_file = '/nfs/bigmind/add_disk0/boyu/Projects/material_segmentation/results/data_jan2019_script/flakeglue_clf/YoungJaeShinSamples/4/train_val_split.p'
        to_load = pickle.load(open(label_file, 'rb'))
        self.labels = to_load['%s_labels'%(subset)]
        self.data_path = os.path.join(img_dir_prefix, subset)
        self.mask_path = os.path.join(mask_dir_prefix, subset)
        self.contrast_path = os.path.join(contrast_dir_prefix, subset)
        self.transforms = transform
        self.img_size = img_size
        self.input_type = input_type
        self.input_with_mask = input_with_mask
        self.num_imgs = len(self.labels)

        fea_name = '/nfs/bigmind/add_disk0/boyu/Projects/material_segmentation/results/data_jan2019_script/flakeglue_clf/YoungJaeShinSamples/4/train_val_data.p'
        fea_to_load = pickle.load(open(fea_name, 'rb'))
        self.all_feas = fea_to_load['%s_feats'%subset]

    def __getitem__(self, index):
        img_name = os.path.join(self.data_path, '%d.png'%index)
        mask_name = os.path.join(self.mask_path, '%d.png'%index)
        contrast_name = os.path.join(self.contrast_path, '%d.png' % index)
        img = Image.open(img_name)
        mask_img = Image.open(mask_name)
        contrast_img = Image.open(contrast_name)

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
            im_contrast = np.array(contrast_img)[:,:,:2]
            img_all.append(im_contrast)
        if self.input_with_mask:
            img_all.append(np.expand_dims(mask_img, 2))
        img_all = np.concatenate(img_all, 2)

        if self.transforms:
            img_all = self.transforms(img_all)
            mask_img = self.transforms(mask_img)

        # label is 1, -1
        return img_all, mask_img, self.labels[index], self.all_feas[index]

    def __len__(self):
        return self.num_imgs

# 0: thick. 1: thin, 2: glue
class completeLabelDataLoader(DD.Dataset):
    def __init__(self, img_dir_prefix, mask_dir_prefix, contrast_dir_prefix, input_type, input_with_mask, subset='train', transform=None, img_size=256):
        super(completeLabelDataLoader, self).__init__()
        label_file = '/nfs/bigmind/add_disk0/boyu/Projects/material_segmentation/results/data_jan2019_script/thickthinglue_clf_complete/YoungJaeShinSamples/4/train_val_split.p'
        to_load = pickle.load(open(label_file, 'rb'))
        self.labels = to_load['%s_labels'%(subset)]
        self.data_path = os.path.join(img_dir_prefix, subset)
        self.mask_path = os.path.join(mask_dir_prefix, subset)
        self.contrast_path = os.path.join(contrast_dir_prefix, subset)
        self.transforms = transform
        self.img_size = img_size
        self.input_type = input_type
        self.input_with_mask = input_with_mask
        self.num_imgs = len(self.labels)

        fea_name = '/nfs/bigmind/add_disk0/boyu/Projects/material_segmentation/results/data_jan2019_script/thickthinglue_clf_complete/YoungJaeShinSamples/4/train_val_data.p'
        fea_to_load = pickle.load(open(fea_name, 'rb'))
        self.all_feas = fea_to_load['%s_feats'%subset]

    def __getitem__(self, index):
        img_name = os.path.join(self.data_path, '%d.png'%index)
        mask_name = os.path.join(self.mask_path, '%d.png'%index)
        contrast_name = os.path.join(self.contrast_path, '%d.png' % index)
        img = Image.open(img_name)
        mask_img = Image.open(mask_name)
        contrast_img = Image.open(contrast_name)

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
            im_contrast = np.array(contrast_img)[:,:,:2]
            img_all.append(im_contrast)
        if self.input_with_mask:
            img_all.append(np.expand_dims(mask_img, 2))
        img_all = np.concatenate(img_all, 2)

        if self.transforms:
            img_all = self.transforms(img_all)
            mask_img = self.transforms(mask_img)

        # label is 1, -1
        return img_all, mask_img, self.labels[index], self.all_feas[index]

    def __len__(self):
        return self.num_imgs


# load all data
# input_type: 'rgb', 'hsv', 'gray', 'rgb-hsv-gray'
# input_with_mask: True, False
class AllDataLoader(DD.Dataset):
    def __init__(self, data_type, img_dir_prefix, mask_dir_prefix, contrast_dir_prefix, input_type, input_with_mask, transform=None, img_size=256):
        super(AllDataLoader, self).__init__()

        # get all image names
        self.all_img_names = []
        self.all_mask_names = []
        self.all_contrast_names = []
        if data_type == 'allexp':
            exp_names = os.listdir(img_dir_prefix)
            for ename in exp_names:
                sub_names = os.listdir(os.path.join(img_dir_prefix, ename))
                for sname in sub_names:
                    pth = os.path.join(img_dir_prefix, ename, sname)
                    mask_pth = os.path.join(mask_dir_prefix, ename, sname)
                    contrast_pth = os.path.join(contrast_dir_prefix, ename, sname)
                    img_names = os.listdir(pth)
                    es_img_names = [os.path.join(pth, n) for n in img_names]
                    es_mask_names = [os.path.join(mask_pth, n) for n in img_names]
                    es_contrast_names = [os.path.join(contrast_pth, n) for n in img_names]
                    self.all_img_names.extend(es_img_names)
                    self.all_mask_names.extend(es_mask_names)
                    self.all_contrast_names.extend(es_contrast_names)
        elif data_type == 'allsubexp':
            ename = 'YoungJaeShinSamples'
            sub_names = os.listdir(os.path.join(img_dir_prefix, ename))
            for sname in sub_names:
                pth = os.path.join(img_dir_prefix, ename, sname)
                mask_pth = os.path.join(mask_dir_prefix, ename, sname)
                contrast_pth = os.path.join(contrast_dir_prefix, ename, sname)
                img_names = os.listdir(pth)
                es_img_names = [os.path.join(pth, n) for n in img_names]
                es_mask_names = [os.path.join(mask_pth, n) for n in img_names]
                es_contrast_names = [os.path.join(contrast_pth, n) for n in img_names]
                self.all_img_names.extend(es_img_names)
                self.all_mask_names.extend(es_mask_names)
                self.all_contrast_names.extend(es_contrast_names)
        elif data_type == 'singlesubexp':
            ename = 'YoungJaeShinSamples'
            sname = '4'
            pth = os.path.join(img_dir_prefix, ename, sname)
            mask_pth = os.path.join(mask_dir_prefix, ename, sname)
            contrast_pth = os.path.join(contrast_dir_prefix, ename, sname)
            img_names = os.listdir(pth)
            es_img_names = [os.path.join(pth, n) for n in img_names]
            es_mask_names = [os.path.join(mask_pth, n) for n in img_names]
            es_contrast_names = [os.path.join(contrast_pth, n) for n in img_names]
            self.all_img_names.extend(es_img_names)
            self.all_mask_names.extend(es_mask_names)
            self.all_contrast_names.extend(es_contrast_names)
        else:
            raise NotImplementedError
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

        img = Image.open(img_name)
        mask_img = Image.open(mask_name)
        contrast_img = Image.open(contrast_name)

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
            im_contrast = np.array(contrast_img)[:,:,:2]
            img_all.append(im_contrast)
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
