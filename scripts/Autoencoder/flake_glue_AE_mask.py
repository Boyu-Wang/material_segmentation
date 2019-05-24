"""
Train an auto encoder for flake/glue, use bottleneck feature for classification
The AE is trained to minimize loss within the mask.
Input could be rgb, gray, hsv, or with addition mask

By: Boyu Wang (boywang@cs.stonybrook.edu)
Created Data: 8 May 2019
Last Modified Date: 8 May 2019
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
import torchvision.utils as vutils
import torch.utils.data as DD
from torch.autograd import Variable
from torch.autograd import gradcheck
from torch.autograd import Function
import misc
from dataloader import LabelDataLoader, AllDataLoader, completeLabelDataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
parser.add_argument('--ngf', type=int, default=32)
parser.add_argument('--ndf', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--gpuid', type=int, default=0, help='ids of GPUs to use')
parser.add_argument('--modelPathEpoch', type=int, default=499, help="model epoch to load")
parser.add_argument('--modelPath', type=str, default='', help="model path to load")
parser.add_argument('--manualSeed', type=int, default=543, help='manual seed')
parser.add_argument('--epoch_iter', type=int,default=500, help='number of epochs on entire dataset')
parser.add_argument('--start_epoch', type=int,default=0, help='start epoch')
parser.add_argument('--data_type', type=str,default='allexp', help='data_type: allexp, allsubexp, singlesubexp')
parser.add_argument('--input_type', type=str,default='rgb', help='input type: rgb, gray, hsv, rgb-gray-hsv')
parser.add_argument('--input_with_mask', type=int,default=1, help='whether to input mask')
parser.add_argument('--evaluate', type=int, default=0, help ='1 for evaluate')
parser.add_argument('--handfea', type=int, default=0, help ='1 use handcrafted features')
parser.add_argument('--normfea', type=int, default=1, help ='1 to normalize feature')
parser.add_argument('--C', type=float, default=10, help ='C for SVM')
opt = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpuid)
# misc
opt.output_dir_prefix = '../../results/AE_%s_input-%s_inputmask-%d/'%(opt.data_type, opt.input_type, opt.input_with_mask)
# load encoder model which is pretrained on reconsturction
# opt.modelPath = opt.output_dir_prefix + 'checkpoints/model_epoch_99'
opt.dirCheckpoints = opt.output_dir_prefix + 'checkpoints'
opt.dirImageoutput = opt.output_dir_prefix + 'images'
# opt.dirLogoutput = opt.output_dir_prefix + 'logs'
opt.dirTestingoutput = opt.output_dir_prefix + 'testing'


opt.imgSize = 256
opt.cuda = True
opt.use_dropout = 0
opt.zdim = 256
opt.use_gpu = True
opt.nc = 0
if 'rgb' in opt.input_type:
    opt.nc += 3
if 'gray' in opt.input_type:
    opt.nc += 1
if 'hsv' in opt.input_type:
    opt.nc += 3
if 'contrast' in opt.input_type:
    opt.nc += 2

opt.nc_out = opt.nc
if opt.input_with_mask:
    opt.nc += 1
print(opt)

try:
    os.makedirs(opt.dirCheckpoints)
except OSError:
    pass
try:
    os.makedirs(opt.dirImageoutput)
except OSError:
    pass
# try:
#     os.makedirs(opt.dirLogoutput)
# except OSError:
#     pass
try:
    os.makedirs(opt.dirTestingoutput)
except OSError:
    pass


# if opt.manualSeed is None:
#     opt.manualSeed = random.randint(1, 10000)
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


if not opt.modelPath=='' or opt.start_epoch > 0:
    print('Reload previous model at: %d' %opt.start_epoch)
    # encoders.load_state_dict(torch.load(opt.modelPath+'_encoders.pth'))
    # decoders.load_state_dict(torch.load(opt.modelPath+'_decoders.pth'))
    encoders.load_state_dict(torch.load(opt.dirCheckpoints + '/model_epoch_%d_encoders.pth'%opt.start_epoch))
    decoders.load_state_dict(torch.load(opt.dirCheckpoints + '/model_epoch_%d_decoders.pth'%(opt.start_epoch)))
else:
    print('No previous model found, initializing model weight.')
    encoders.apply(misc.weights_init)
    decoders.apply(misc.weights_init)


updator_encoders = optim.Adam(encoders.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
updator_decoders = optim.Adam(decoders.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# updator_classify = optim.Adam(mlp_classify.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

criterionRecon = models.MaskABSLoss()
# criterionClassify = nn.BCEWithLogitsLoss()
# criterionClassify = criterionClassify.cuda()


all_img_path = '../../results/data_jan2019_script/ae_img'
all_mask_path = '../../results/data_jan2019_script/ae_mask'
all_contrast_path = '../../results/data_jan2019_script/ae_contrast'
# label_img_path = '../../results/data_jan2019_script/ae_label_img'
# label_mask_path = '../../results/data_jan2019_script/ae_label_mask'
# label_contrast_path = '../../results/data_jan2019_script/ae_label_contrast'
label_img_path = '../../results/data_jan2019_script/ae_complete_label_img'
label_mask_path = '../../results/data_jan2019_script/ae_complete_label_mask'
label_contrast_path = '../../results/data_jan2019_script/ae_complete_label_contrast'

resize = transforms.Compose([transforms.ToTensor()])
trvalSet = AllDataLoader(data_type=opt.data_type, img_dir_prefix=all_img_path, mask_dir_prefix=all_mask_path, contrast_dir_prefix=all_contrast_path, input_type=opt.input_type, input_with_mask=opt.input_with_mask, transform=resize)
# trSet = LabelDataLoader(img_dir_prefix=label_img_path, mask_dir_prefix=label_mask_path, contrast_dir_prefix=label_contrast_path, input_type=opt.input_type, input_with_mask=opt.input_with_mask, subset='train', transform=resize)
# valSet = LabelDataLoader(img_dir_prefix=label_img_path, mask_dir_prefix=label_mask_path, contrast_dir_prefix=label_contrast_path, input_type=opt.input_type, input_with_mask=opt.input_with_mask, subset='val', transform=resize)
trSet = completeLabelDataLoader(img_dir_prefix=label_img_path, mask_dir_prefix=label_mask_path, contrast_dir_prefix=label_contrast_path, input_type=opt.input_type, input_with_mask=opt.input_with_mask, subset='train', transform=resize)
valSet = completeLabelDataLoader(img_dir_prefix=label_img_path, mask_dir_prefix=label_mask_path, contrast_dir_prefix=label_contrast_path, input_type=opt.input_type, input_with_mask=opt.input_with_mask, subset='val', transform=resize)

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


for epoch in range(opt.start_epoch, opt.epoch_iter):
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
    for batch_idx, (input_img, mask_img) in enumerate(trvalLD, 0):
        gc.collect() # collect garbage
        ### prepare data ###
        input_data = input_img.cuda()
        target_data = input_img
        mask_img = mask_img.cuda()
        updator_encoders.zero_grad()
        updator_decoders.zero_grad()

        # criterionRecon.zero_grad()
        
        fea = encoders(input_data)
        recons = decoders(fea)
        # reconstruction loss
        # print(recons.size())
        # print(fea.size())
        loss_recon = criterionRecon(recons, input_data, mask_img)
        loss_recon.backward(retain_graph=True)

        updator_decoders.step()
        updator_encoders.step()

        batch_size_t = input_img.size(0)
        train_loss += loss_recon.data.item() * batch_size_t
        iter_mark+=1
        train_amount += batch_size_t

        print('Iteration[%d] recon loss: %.4f'
            % (iter_mark, loss_recon.data.item()))

    train_loss /= train_amount
    # train_losses.append(train_loss.cpu().numpy())

    # # save images after iterating a dataroot
    if opt.nc == 3 or opt.nc == 1:
        misc.visualizeAsImages(input_img.data.clone(),
            opt.dirImageoutput,
            filename='iter_'+str(epoch)+'_input_', n_sample=49, nrow=7, normalize=False)
        misc.visualizeAsImages(recons.data.clone(),
            opt.dirImageoutput,
            filename='iter_'+str(epoch)+'_output_', n_sample=49, nrow=7, normalize=False)
    if opt.input_with_mask and (opt.nc==4 or opt.nc==2):
        misc.visualizeAsImages(input_img[:,:-1].data.clone(),
                               opt.dirImageoutput,
                               filename='iter_' + str(epoch) + '_input_', n_sample=49, nrow=7, normalize=False)
        misc.visualizeAsImages(recons[:,:-1].data.clone(),
                               opt.dirImageoutput,
                               filename='iter_' + str(epoch) + '_output_', n_sample=49, nrow=7, normalize=False)

    if doTraining:
        print('====> Epoch: {} Average training loss:'.format(epoch))
        print(train_loss)
        # do checkpointing
        torch.save(encoders.state_dict(), '%s/model_epoch_%d_encoders.pth' % (opt.dirCheckpoints, epoch))
        torch.save(decoders.state_dict(), '%s/model_epoch_%d_decoders.pth' % (opt.dirCheckpoints, epoch))

    # evaluate on validation set
    """
    if epoch % 1 == 0:
        # gt_labels = []
        # feats = []
        for batch_idx, (input_img, mask_img, input_labels, input_feas) in enumerate(valLD, 0):
            gc.collect() # collect garbage
            ### prepare data ###
            input_data = input_img.cuda()
            input_labels = list(input_labels.numpy())
            mask_img = mask_img.cuda()
            # gt_labels.extend(input_labels)
            
            fea = encoders(input_data)
            recons = decoders(fea)
            loss_recon = criterionRecon(recons, input_data, mask_img)
            
            batch_size_t = input_img.size(0)
            val_loss += loss_recon.data.item() * batch_size_t
            val_amount += batch_size_t

            # fea = fea.data.cpu().numpy()
            # fea = np.reshape(fea, [batch_size_t, -1])
            # feats.append(fea)

        val_loss /= val_amount
        # val_losses.append(val_loss.cpu().numpy())

        print(val_loss)
        if opt.nc == 3 or opt.nc == 1:
            misc.visualizeAsImages(input_img.data.clone(),
                opt.dirImageoutput,
                filename='test_iter_'+str(epoch)+'_input_', n_sample=49, nrow=7, normalize=False)
            misc.visualizeAsImages(recons.data.clone(),
                opt.dirImageoutput,
                filename='test_iter_'+str(epoch)+'_output_', n_sample=49, nrow=7, normalize=False)
        if opt.input_with_mask and (opt.nc == 4 or opt.nc == 2):
            misc.visualizeAsImages(input_img[:, :-1].data.clone(),
                                   opt.dirImageoutput,
                                   filename='test_iter_' + str(epoch) + '_input_', n_sample=49, nrow=7, normalize=False)
            misc.visualizeAsImages(recons[:, :-1].data.clone(),
                                   opt.dirImageoutput,
                                   filename='test_iter_' + str(epoch) + '_output_', n_sample=49, nrow=7, normalize=False)
    """
if doTesting:
    print('loading pretrained model')
    encoders.load_state_dict(torch.load('%s/model_epoch_%d_encoders.pth' % (opt.dirCheckpoints, opt.modelPathEpoch)))
    decoders.load_state_dict(torch.load('%s/model_epoch_%d_decoders.pth' % (opt.dirCheckpoints, opt.modelPathEpoch)))
    # ------------ testing ------------ #
    encoders.eval()
    decoders.eval()
    
    # extract train features
    train_labels = []
    train_feats = []
    train_handfeats = []
    for batch_idx, (input_img, mask_img, input_labels, input_feas) in enumerate(trLD, 0):
        gc.collect() # collect garbage
        ### prepare data ###
        input_data = input_img.cuda()
        input_labels = list(input_labels.numpy())
        train_labels.extend(input_labels)
        
        fea = encoders(input_data)
        recons = decoders(fea)
        # loss_recon = criterionRecon(recons, input_data, weight=1)
        
        batch_size_t = input_img.size(0)

        fea = fea.data.cpu().numpy()
        fea = np.reshape(fea, [batch_size_t, -1])
        input_feas = input_feas.numpy()
        train_handfeats.append(input_feas)
        # if opt.handfea:
        #     fea = np.concatenate([fea, input_feas], 1)
        train_feats.append(fea)
    train_feats = np.concatenate(train_feats)
    train_handfeats = np.concatenate(train_handfeats)

    val_labels = []
    val_feats = []
    val_handfeats = []
    for batch_idx, (input_img, mask_img, input_labels, input_feas) in enumerate(valLD, 0):
        gc.collect() # collect garbage
        ### prepare data ###
        input_data = input_img.cuda()
        input_labels = list(input_labels.numpy())
        val_labels.extend(input_labels)
        
        fea = encoders(input_data)
        recons = decoders(fea)
        # loss_recon = criterionRecon(recons, input_data, weight=1)
        
        batch_size_t = input_img.size(0)

        fea = fea.data.cpu().numpy()
        fea = np.reshape(fea, [batch_size_t, -1])
        input_feas = input_feas.numpy()
        val_handfeats.append(input_feas)
        # if opt.handfea:
        #     fea = np.concatenate([fea, input_feas], 1)
        val_feats.append(fea)
    val_feats = np.concatenate(val_feats)
    val_handfeats = np.concatenate(val_handfeats)

    # normalize handfeats
    mean_handfeat = np.mean(train_handfeats, axis=0, keepdims=True)
    std_handfeat = np.std(train_handfeats, axis=0, keepdims=True)
    train_handfeats -= mean_handfeat
    train_handfeats = train_handfeats / std_handfeat
    val_handfeats -= mean_handfeat
    val_handfeats = val_handfeats / std_handfeat

    # normalize fea
    if opt.normfea:
        mean_feat = np.mean(train_feats, axis=0, keepdims=True)
        # std_feat = np.std(train_feats, axis=0, keepdims=True)
        train_feats -= mean_feat
        # train_feats = train_feats / std_feat
        train_feats = train_feats / np.linalg.norm(train_feats, 2, axis=1, keepdims=True)
        # train_feats = train_feats / np.sum(train_feats, 1, keepdims=True)
        val_feats -= mean_feat
        # val_feats = val_feats / std_feat
        val_feats = val_feats / np.linalg.norm(val_feats, 2, axis=1, keepdims=True)
        # val_feats = val_feats / np.sum(val_feats, 1, keepdims=True)


    if opt.handfea:
        train_feats = np.concatenate([train_feats, train_handfeats], 1)
        val_feats = np.concatenate([val_feats, val_handfeats], 1)

        if opt.normfea:
            mean_feat = np.mean(train_feats, axis=0, keepdims=True)
            std_feat = np.std(train_feats, axis=0, keepdims=True)
            train_feats -= mean_feat
            # train_feats = train_feats / std_feat
            train_feats = train_feats / np.linalg.norm(train_feats, 2, axis=1, keepdims=True)
            # train_feats = train_feats / np.sum(train_feats, 1, keepdims=True)
            val_feats -= mean_feat
            # val_feats = val_feats / std_feat
            val_feats = val_feats / np.linalg.norm(val_feats, 2, axis=1, keepdims=True)
            # val_feats = val_feats / np.sum(val_feats, 1, keepdims=True)

    from sklearn.svm import LinearSVC, SVC
    C = opt.C

    Cs = [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 50]
    for C in Cs:
        method = 'linearsvm'
        clf = LinearSVC(random_state=0, tol=1e-5, C=C, max_iter=5e4)
        clf.fit(train_feats, train_labels)

        train_pred_cls = clf.predict(train_feats)
        train_pred_scores = clf.decision_function(train_feats)
        val_pred_cls = clf.predict(val_feats)
        val_pred_scores = clf.decision_function(val_feats)

        # from sklearn.metrics import accuracy_score
        # train_acc = accuracy_score(train_labels, train_pred_cls)
        # val_acc = accuracy_score(val_labels, val_pred_cls)
        # print('%s-%f: train: %.4f, val: %.4f' %(method, C, train_acc, val_acc))


        from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix
        train_acc = accuracy_score(train_labels, train_pred_cls)
        val_acc = accuracy_score(val_labels, val_pred_cls)
        train_conf = confusion_matrix(train_labels, train_pred_cls)
        train_conf = train_conf / np.sum(train_conf, 1, keepdims=True)
        val_conf = confusion_matrix(val_labels, val_pred_cls)
        val_conf = val_conf / np.sum(val_conf, 1, keepdims=True)
        print(train_conf)
        print(val_conf)
        # calculate map:
        uniquelabels = [0, 1, 2]
        train_aps = []
        val_aps = []
        for l in uniquelabels:
            l_train_labels = [_ == l for _ in train_labels]
            l_val_labels = [_ == l for _ in val_labels]
            train_aps.append(average_precision_score(l_train_labels, train_pred_scores[:, l]))
            val_aps.append(average_precision_score(l_val_labels, val_pred_scores[:, l]))

        # train_aps = average_precision_score(train_labels, train_pred_scores)
        # val_aps = average_precision_score(val_labels, val_pred_scores)

        print(train_aps)
        print(val_aps)
        print('%s-%f: train: %.4f, val: %.4f, ap train: %4f, ap val: %4f' % (method, C, train_acc, val_acc, np.mean(train_aps), np.mean(val_aps)))
