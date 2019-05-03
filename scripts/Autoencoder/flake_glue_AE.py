"""
Train an auto encoder for flake/glue, use bottleneck feature for classification

By: Boyu Wang (boywang@cs.stonybrook.edu)
Created Data: 24 Apr 2019
Last Modified Date: 2 May 2019
"""


import numpy as np
from PIL import Image
import cv2
import argparse
import os
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from skimage.morphology import disk
import pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import itertools
from mpl_toolkits.mplot3d import Axes3D
import gc
import copy
import sqlite3
from multiprocessing import Pool
from joblib import Parallel, delayed
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as DD
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.autograd import gradcheck
from torch.autograd import Function
import misc

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
parser.add_argument('--ngf', type=int, default=32)
parser.add_argument('--ndf', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--gpuid', type=int, default=0, help='ids of GPUs to use')
parser.add_argument('--modelPathEpoch', type=int, default=199, help="model epoch to load")
parser.add_argument('--modelPath', type=str, default='', help="model path to load")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--epoch_iter', type=int,default=200, help='number of epochs on entire dataset')
parser.add_argument('--evaluate', type=int, default=0, help ='1 for evaluate')
parser.add_argument('--handfea', type=int, default=0, help ='1 use handcrafted features')
parser.add_argument('--normfea', type=int, default=1, help ='1 to normalize feature')
parser.add_argument('--C', type=float, default=10, help ='C for SVM')
parser.add_argument('--data_dir_prefix', type=str, default='/nfs/bigmind/add_disk0/boyu/Projects/material_segmentation/results/data_jan2019_script/masked_img/', help='data path')
opt = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpuid)
# misc
opt.output_dir_prefix = '../../results/AE_trainval/'
# load encoder model which is pretrained on reconsturction
# opt.modelPath = opt.output_dir_prefix + 'checkpoints/model_epoch_99'
opt.dirCheckpoints = opt.output_dir_prefix + 'checkpoints'
opt.dirImageoutput = opt.output_dir_prefix + 'images'
opt.dirLogoutput = opt.output_dir_prefix + 'logs'
opt.dirTestingoutput = opt.output_dir_prefix + 'testing'


opt.imgSize = 256
opt.cuda = True
opt.use_dropout = 0
opt.zdim = 512
opt.use_gpu = True
opt.nc = 3
print(opt)

try:
    os.makedirs(opt.dirCheckpoints)
except OSError:
    pass
try:
    os.makedirs(opt.dirImageoutput)
except OSError:
    pass
try:
    os.makedirs(opt.dirLogoutput)
except OSError:
    pass
try:
    os.makedirs(opt.dirTestingoutput)
except OSError:
    pass


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


# ---- The model ---- #
import models
encoders = models.Encoders(opt)
decoders = models.Decoders(opt)
# mlp_classify = models.mlp_classify(opt)


if opt.cuda:
    encoders.cuda()
    decoders.cuda()
    # mlp_classify.cuda()


if not opt.modelPath=='':
    print('Reload previous model at: '+ opt.modelPath)
    encoders.load_state_dict(torch.load(opt.modelPath+'_encoders.pth'))
    decoders.load_state_dict(torch.load(opt.modelPath+'_decoders.pth'))
else:
    print('No previous model found, initializing model weight.')
    encoders.apply(misc.weights_init)
    decoders.apply(misc.weights_init)


updator_encoders = optim.Adam(encoders.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
updator_decoders = optim.Adam(decoders.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# updator_classify = optim.Adam(mlp_classify.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

criterionRecon = models.WeightABSLoss(opt)
# criterionClassify = nn.BCEWithLogitsLoss()
# criterionClassify = criterionClassify.cuda()

def process_masked_imgs():
    flake_save_name = os.path.join('/nfs/bigmind/add_disk0/boyu/Projects/material_segmentation/results/data_jan2019_script/flakeglue_clf/YoungJaeShinSamples/4', 'train_val_data.p')
    to_load = pickle.load(open(flake_save_name, 'rb'))
    train_flakes = to_load['train_flakes']
    val_flakes = to_load['val_flakes']
    output_path = '/nfs/bigmind/add_disk0/boyu/Projects/material_segmentation/results/data_jan2019_script/masked_img'
    def helper(flakes, output_path):
        img_size = 256
        try:
            os.makedirs(output_path)
        except OSError:
            pass
        for i in range(len(flakes)):
            print(i)
            flake = flakes[i]
            contours = flake['flake_contour_loc']
            contours[:,0] = contours[:,0] - flake['flake_large_bbox'][0]
            contours[:,1] = contours[:,1] - flake['flake_large_bbox'][2]
            contours = np.expand_dims(np.flip(contours), 1).astype(np.int32)
            flake_centroids = flake['flake_center'].astype('int')
            flake_centroids[0] = flake_centroids[0] - flake['flake_large_bbox'][0]
            flake_centroids[1] = flake_centroids[1] - flake['flake_large_bbox'][2]
            flake_img_size = flake['flake_img'].shape
            mask = np.zeros(flake_img_size).astype(np.uint8)
            mask = cv2.drawContours(mask, [contours], -1, (1,1,1), -1)
            masked_img = flake['flake_img'] * mask
            hoffset = 0
            woffset = 0
            if masked_img.shape[0] > img_size:
                hoffset = int(masked_img.shape[0]/2-img_size/2)
                masked_img = masked_img[int(masked_img.shape[0]/2-img_size/2): int(masked_img.shape[0]/2+img_size/2), :, :]
            if masked_img.shape[1] > img_size:
                woffset = int(masked_img.shape[1]/2-img_size/2)
                masked_img = masked_img[:, int(masked_img.shape[1]/2-img_size/2): int(masked_img.shape[1]/2+img_size/2), :]
            new_img = np.zeros([img_size, img_size, 3]).astype(np.uint8)
            flake_img_size = masked_img.shape
            hs = max(0, img_size//2 - flake_centroids[0] + hoffset)
            he = min(hs +flake_img_size[0], img_size)
            hs = he - flake_img_size[0]
            ws = max(0, img_size//2 - flake_centroids[1] + woffset)
            we = min(ws + flake_img_size[1], img_size)
            ws = we - flake_img_size[1]
            # print(masked_img.shape, hs, he, ws, we, new_img.shape, new_img[hs:he, ws:we,:].shape)
            new_img[hs:he, ws:we,:] = masked_img# max(0, ) : min(img_size//2 - flake_centroids[1] +flake_img_size[1], img_size), :] = masked_img
            cv2.imwrite(os.path.join(output_path, '%d.png'%(i)), np.flip(new_img, 2))
    helper(train_flakes, os.path.join(output_path, 'train'))
    helper(val_flakes, os.path.join(output_path, 'val'))


class DataLoder(DD.Dataset):
    def __init__(self, data_dir_prefix, subset='train', transform=None, img_size=256):
        super(DataLoder, self).__init__()
        label_file = '/nfs/bigmind/add_disk0/boyu/Projects/material_segmentation/results/data_jan2019_script/flakeglue_clf/YoungJaeShinSamples/4/train_val_split.p'
        to_load = pickle.load(open(label_file, 'rb'))
        self.labels = to_load['%s_labels'%(subset)]
        self.data_path = os.path.join(data_dir_prefix, subset)
        self.transforms = transform
        self.img_size = img_size
        self.num_imgs = len(self.labels)

        fea_name = '/nfs/bigmind/add_disk0/boyu/Projects/material_segmentation/results/data_jan2019_script/flakeglue_clf/YoungJaeShinSamples/4/train_val_data.p'
        fea_to_load = pickle.load(open(fea_name, 'rb'))
        self.all_feas = fea_to_load['%s_feats'%subset]

    def __getitem__(self, index):
        img_name = os.path.join(self.data_path, '%d.png'%index)
        img = Image.open(img_name)
        if self.transforms:
            img = self.transforms(img)
        return img, self.labels[index], self.all_feas[index]

    def __len__(self):
        return self.num_imgs

class AllDataLoder(DD.Dataset):
    def __init__(self, data_dir_prefix, transform=None, img_size=256):
        super(AllDataLoder, self).__init__()
        label_file = '/nfs/bigmind/add_disk0/boyu/Projects/material_segmentation/results/data_jan2019_script/flakeglue_clf/YoungJaeShinSamples/4/train_val_split.p'
        to_load = pickle.load(open(label_file, 'rb'))
        self.labels = to_load['train_labels'] + to_load['val_labels']
        self.train_data_path = os.path.join(data_dir_prefix, 'train')
        self.val_data_path = os.path.join(data_dir_prefix, 'val')
        # self.data_path = os.path.join(data_dir_prefix, subset)
        fea_name = '/nfs/bigmind/add_disk0/boyu/Projects/material_segmentation/results/data_jan2019_script/flakeglue_clf/YoungJaeShinSamples/4/train_val_data.p'
        fea_to_load = pickle.load(open(fea_name, 'rb'))
        self.all_feas = np.concatenate([fea_to_load['train_feats'], fea_to_load['val_feats']])

        self.transforms = transform
        self.img_size = img_size
        self.num_imgs = len(self.labels)
        assert self.all_feas.shape[0] == self.num_imgs
        self.train_num_imgs = len(to_load['train_labels'])

    def __getitem__(self, index):
        if index < self.train_num_imgs:
            img_name = os.path.join(self.train_data_path, '%d.png'%index)
        else:
            img_name = os.path.join(self.val_data_path, '%d.png'%(index-self.train_num_imgs))
        img = Image.open(img_name)
        if self.transforms:
            img = self.transforms(img)
        return img, self.labels[index], self.all_feas[index]

    def __len__(self):
        return self.num_imgs


resize = transforms.Compose([transforms.ToTensor()])
trvalSet = AllDataLoder(data_dir_prefix=opt.data_dir_prefix, transform=resize)
trSet = DataLoder(data_dir_prefix=opt.data_dir_prefix, subset='train', transform=resize)
valSet = DataLoder(data_dir_prefix=opt.data_dir_prefix, subset='val', transform=resize)

trvalLD = DD.DataLoader(trvalSet, batch_size=opt.batchSize,
       sampler=DD.sampler.RandomSampler(trvalSet),
       num_workers=opt.workers, pin_memory=True)
trLD = DD.DataLoader(trSet, batch_size=opt.batchSize,
       sampler=DD.sampler.RandomSampler(trSet),
       num_workers=opt.workers, pin_memory=True)
valLD = DD.DataLoader(valSet, batch_size=opt.batchSize,
       sampler=DD.sampler.SequentialSampler(valSet),
       num_workers=opt.workers, pin_memory=True)

# ------------ training ------------ #
if opt.evaluate == 1:
    doTraining = False
    doTesting = True
elif opt.evaluate == 0:
    doTraining = True
    doTesting = False

iter_mark=0
train_losses = []
val_losses = []
val_accs = []


for epoch in range(opt.epoch_iter):
    train_loss = torch.zeros(1).cuda()
    train_amount = 0
    val_loss = torch.zeros(1).cuda()
    val_amount = 0
    gc.collect() # collect garbage
    encoders.train()
    decoders.train()
    # for dataroot in TrainingData:
    if not doTraining:
        break
    for batch_idx, (input_img, input_labels, input_feas) in enumerate(trvalLD, 0):
        gc.collect() # collect garbage
        ### prepare data ###
        input_data = input_img.cuda()
        target_data = input_img
        # target_label = class_label.float()
        # input_var = torch.autograd.Variable(input_data).cuda()
        # target_var = torch.autograd.Variable(target_data).cuda()
        # target_label = torch.autograd.Variable(target_label, requires_grad=False).cuda()
        updator_encoders.zero_grad()
        encoders.zero_grad()
        updator_decoders.zero_grad()
        decoders.zero_grad()
        
        criterionRecon.zero_grad()
        
        fea = encoders(input_data)
        recons = decoders(fea)
        # reconstruction loss
        loss_recon = criterionRecon(recons, input_data, weight=1)
        loss_recon.backward(retain_graph=True)

        updator_decoders.step()
        updator_encoders.step()

        batch_size_t = input_img.size(0)
        train_loss += loss_recon.data[0] * batch_size_t
        iter_mark+=1
        train_amount += batch_size_t

        print('Iteration[%d] recon loss: %.4f'
            % (iter_mark, loss_recon.data[0]))

    train_loss /= train_amount
    train_losses.append(train_loss.cpu().numpy())

    # # save images after iterating a dataroot
    misc.visualizeAsImages(input_img.data.clone(),
        opt.dirImageoutput,
        filename='iter_'+str(epoch)+'_input_', n_sample=49, nrow=7, normalize=False)
    misc.visualizeAsImages(recons.data.clone(),
        opt.dirImageoutput,
        filename='iter_'+str(epoch)+'_output_', n_sample=49, nrow=7, normalize=False)

    if doTraining:
        print('====> Epoch: {} Average training loss:'.format(epoch))
        print(train_loss)
        # do checkpointing
        torch.save(encoders.state_dict(), '%s/model_epoch_%d_encoders.pth' % (opt.dirCheckpoints, epoch))
        torch.save(decoders.state_dict(), '%s/model_epoch_%d_decoders.pth' % (opt.dirCheckpoints, epoch))

    # evaluate on validation set
    if epoch % 1 == 0:
        # gt_labels = []
        # feats = []
        for batch_idx, (input_img, input_labels, input_feas) in enumerate(valLD, 0):
            gc.collect() # collect garbage
            ### prepare data ###
            input_data = input_img.cuda()
            input_labels = list(input_labels.numpy())
            # gt_labels.extend(input_labels)
            
            fea = encoders(input_data)
            recons = decoders(fea)
            loss_recon = criterionRecon(recons, input_data, weight=1)
            
            batch_size_t = input_img.size(0)
            val_loss += loss_recon.data[0] * batch_size_t
            val_amount += batch_size_t

            # fea = fea.data.cpu().numpy()
            # fea = np.reshape(fea, [batch_size_t, -1])
            # feats.append(fea)

        val_loss /= val_amount
        val_losses.append(val_loss.cpu().numpy())

        print(val_loss)
        misc.visualizeAsImages(input_img.data.clone(),
            opt.dirImageoutput,
            filename='test_iter_'+str(epoch)+'_input_', n_sample=49, nrow=7, normalize=False)
        misc.visualizeAsImages(recons.data.clone(),
            opt.dirImageoutput,
            filename='test_iter_'+str(epoch)+'_output_', n_sample=49, nrow=7, normalize=False)


if doTesting:
    encoders.load_state_dict(torch.load('%s/model_epoch_%d_encoders.pth' % (opt.dirCheckpoints, opt.modelPathEpoch)))
    decoders.load_state_dict(torch.load('%s/model_epoch_%d_decoders.pth' % (opt.dirCheckpoints, opt.modelPathEpoch)))
    # ------------ testing ------------ #
    encoders.eval()
    decoders.eval()
    
    # extract train features
    train_labels = []
    train_feats = []
    for batch_idx, (input_img, input_labels, input_feas) in enumerate(trLD, 0):
        gc.collect() # collect garbage
        ### prepare data ###
        input_data = input_img.cuda()
        input_labels = list(input_labels.numpy())
        train_labels.extend(input_labels)
        
        fea = encoders(input_data)
        recons = decoders(fea)
        loss_recon = criterionRecon(recons, input_data, weight=1)
        
        batch_size_t = input_img.size(0)
        val_loss += loss_recon.data[0] * batch_size_t
        val_amount += batch_size_t

        fea = fea.data.cpu().numpy()
        fea = np.reshape(fea, [batch_size_t, -1])
        input_feas = input_feas.numpy()
        if opt.handfea:
            fea = np.concatenate([fea, input_feas], 1)
        train_feats.append(fea)
    train_feats = np.concatenate(train_feats)

    val_labels = []
    val_feats = []
    for batch_idx, (input_img, input_labels, input_feas) in enumerate(valLD, 0):
        gc.collect() # collect garbage
        ### prepare data ###
        input_data = input_img.cuda()
        input_labels = list(input_labels.numpy())
        val_labels.extend(input_labels)
        
        fea = encoders(input_data)
        recons = decoders(fea)
        loss_recon = criterionRecon(recons, input_data, weight=1)
        
        batch_size_t = input_img.size(0)
        val_loss += loss_recon.data[0] * batch_size_t
        val_amount += batch_size_t

        fea = fea.data.cpu().numpy()
        fea = np.reshape(fea, [batch_size_t, -1])
        input_feas = input_feas.numpy()
        if opt.handfea:
            fea = np.concatenate([fea, input_feas], 1)
        val_feats.append(fea)
    val_feats = np.concatenate(val_feats)

    # normalize fea
    if opt.normfea:
        mean_feat = np.mean(train_feats, axis=0, keepdims=True)
        train_feats -= mean_feat
        train_feats = train_feats / np.linalg.norm(train_feats, 2, axis=1, keepdims=True)
        # train_feats = train_feats / np.sum(train_feats, 1, keepdims=True)
        val_feats -= mean_feat
        val_feats = val_feats / np.linalg.norm(val_feats, 2, axis=1, keepdims=True)
        # val_feats = val_feats / np.sum(val_feats, 1, keepdims=True)


    from sklearn.svm import LinearSVC, SVC
    C = opt.C
    method = 'linearsvm'
    clf = LinearSVC(random_state=0, tol=1e-5, C=C, max_iter=5e4)
    clf.fit(train_feats, train_labels)

    train_pred_cls = clf.predict(train_feats)
    train_pred_scores = clf.decision_function(train_feats)
    pred_cls = clf.predict(val_feats)
    pred_scores = clf.decision_function(val_feats)
    
    from sklearn.metrics import accuracy_score
    train_acc = accuracy_score(train_labels, train_pred_cls)
    val_acc = accuracy_score(val_labels, pred_cls)
    print('%s-%f: train: %.4f, val: %.4f' %(method, C, train_acc, val_acc))