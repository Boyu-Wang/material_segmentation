"""
Train an auto encoder for graphene/non-graphene, use bottleneck feature for classification
The AE is trained to minimize loss within the mask.
Input could be rgb, gray, hsv, or with addition mask

"""


import numpy as np
from PIL import Image
import cv2
import argparse
import os
import scipy
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from skimage.morphology import disk
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
from graphene_dataloader import AllDataLoader, completeLabelDataLoader


def getPrecisionAtRecall(precision, recall, rate=0.95):
    # find the recall which is the first one that small or equal to rate.
    for id, r in enumerate(recall):
        if r <= rate:
            break
    return precision[id]



parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=200, help='input batch size')
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
parser.add_argument('--data_type', type=str,default='nov', help='data_type: sep-oct, sep-oct-nov, nov')
parser.add_argument('--input_type', type=str,default='rgb-contrast-bg', help='input type: rgb, gray, hsv, rgb-gray-hsv')
parser.add_argument('--hand_color_fea', type=str,default='twosub-contrast-bg-shape', help='handcrafted features type: contrast-bg-shape, innercontrast-bg-shape, subsegment-contrast-bg-shape, subsegment-bg-shape, twosub-contrast-bg-shape')
parser.add_argument('--input_with_mask', type=int,default=1, help='whether to input mask')
parser.add_argument('--evaluate', type=int, default=0, help ='1 for evaluate')
parser.add_argument('--handfea', type=int, default=0, help ='1 use handcrafted features')
parser.add_argument('--handfeaonly', type=int, default=0, help ='1 use handcrafted features only')
parser.add_argument('--normfea', type=int, default=1, help ='1 to normalize feature')
parser.add_argument('--C', type=float, default=10, help ='C for SVM')
parser.add_argument('--zdim', type=int, default=256, help ='bottleneck dimension')
opt = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpuid)
# misc
opt.output_dir_prefix = '../../results/graphene_AE_%s_input-%s_inputmask-%d_dim-%d/'%(opt.data_type, opt.input_type, opt.input_with_mask, opt.zdim)
# load encoder model which is pretrained on reconsturction
opt.pretrained_modelPath = '../../results/AE_allexp_input-%s_inputmask-%d/'%(opt.input_type, opt.input_with_mask) + 'checkpoints/model_epoch_499'

opt.dirCheckpoints = opt.output_dir_prefix + 'checkpoints'
opt.dirImageoutput = opt.output_dir_prefix + 'images'
# opt.dirLogoutput = opt.output_dir_prefix + 'logs'
opt.dirTestingoutput = opt.output_dir_prefix + 'testing'


opt.imgSize = 256
opt.cuda = True
opt.use_dropout = 0
# opt.zdim = 256
opt.use_gpu = True
opt.nc = 0
if 'rgb' in opt.input_type:
    opt.nc += 3
if 'gray' in opt.input_type:
    opt.nc += 1
if 'hsv' in opt.input_type:
    opt.nc += 3
if 'contrast' in opt.input_type:
    opt.nc += 3
if 'bg' in opt.input_type:
    opt.nc += 3


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
    if opt.input_type == 'rbg' and opt.zdim == 256:
        print('No previous model found, initializing from previous experiment: ', opt.pretrained_modelPath)
        encoders.load_state_dict(torch.load(opt.pretrained_modelPath + '_encoders.pth'))
        decoders.load_state_dict(torch.load(opt.pretrained_modelPath + '_decoders.pth'))
    else:
        print('No previous model found, initializing from random')
        encoders.apply(misc.weights_init)
        decoders.apply(misc.weights_init)

updator_encoders = optim.Adam(encoders.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
updator_decoders = optim.Adam(decoders.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# updator_classify = optim.Adam(mlp_classify.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

criterionRecon = models.MaskABSLoss()
# criterionClassify = nn.BCEWithLogitsLoss()
# criterionClassify = criterionClassify.cuda()

all_img_path = []
all_mask_path = []
all_contrast_path = []
all_bg_path = []
label_img_path = []
label_mask_path = []
label_contrast_path = []
label_bg_path = []

if 'sep' in opt.data_type:
    all_img_path.append('../../results/data_sep2019_script/ae_img')
    all_mask_path.append('../../results/data_sep2019_script/ae_mask')
    all_contrast_path.append('../../results/data_sep2019_script/ae_contrast')
    label_img_path.append('../../results/data_sep2019_script/ae_complete_label_img')
    label_mask_path.append('../../results/data_sep2019_script/ae_complete_label_mask')
    label_contrast_path.append('../../results/data_sep2019_script/ae_complete_label_contrast')

if 'oct' in opt.data_type:
    all_img_path.append('../../results/10222019G wtih Suji_script/ae_img')
    all_mask_path.append('../../results/10222019G wtih Suji_script/ae_mask')
    all_contrast_path.append('../../results/10222019G wtih Suji_script/ae_contrast')
    label_img_path.append('../../results/10222019G wtih Suji_script/ae_complete_label_img')
    label_mask_path.append('../../results/10222019G wtih Suji_script/ae_complete_label_mask')
    label_contrast_path.append('../../results/10222019G wtih Suji_script/ae_complete_label_contrast')

if 'nov' in opt.data_type:
    all_img_path.append('../../results/data_111x_individual_script/ae_img')
    all_mask_path.append('../../results/data_111x_individual_script/ae_mask')
    all_contrast_path.append('../../results/data_111x_individual_script/ae_contrast')
    all_bg_path.append('../../results/data_111x_individual_script/ae_bg')
    # label_img_path.append('../../results/data_111x_individual_script/ae_complete_label_img')
    # label_mask_path.append('../../results/data_111x_individual_script/ae_complete_label_mask')
    # label_contrast_path.append('../../results/data_111x_individual_script/ae_complete_label_contrast')
    # label_bg_path.append('../../results/data_111x_individual_script/ae_complete_label_bg')
    label_img_path.append('../../results/data_111x_individual_script/ae_complete_doublecheck_label_img')
    label_mask_path.append('../../results/data_111x_individual_script/ae_complete_doublecheck_label_mask')
    label_contrast_path.append('../../results/data_111x_individual_script/ae_complete_doublecheck_label_contrast')
    label_bg_path.append('../../results/data_111x_individual_script/ae_complete_doublecheck_label_bg')



resize = transforms.Compose([transforms.ToTensor()])
trvalSet = AllDataLoader(data_type=opt.data_type, img_dir_prefixs=all_img_path, mask_dir_prefixs=all_mask_path, contrast_dir_prefixs=all_contrast_path, bg_dir_prefixs=all_bg_path, input_type=opt.input_type, input_with_mask=opt.input_with_mask, transform=resize)
trSet = completeLabelDataLoader(data_type=opt.data_type, img_dir_prefixs=label_img_path, mask_dir_prefixs=label_mask_path, contrast_dir_prefixs=label_contrast_path, bg_dir_prefixs=label_bg_path, input_type=opt.input_type, input_with_mask=opt.input_with_mask, subset='train', transform=resize, color_fea=opt.hand_color_fea)
valSet = completeLabelDataLoader(data_type=opt.data_type, img_dir_prefixs=label_img_path, mask_dir_prefixs=label_mask_path, contrast_dir_prefixs=label_contrast_path, bg_dir_prefixs=label_bg_path, input_type=opt.input_type, input_with_mask=opt.input_with_mask, subset='val', transform=resize, color_fea=opt.hand_color_fea)
# labelSet = completeLabelDataLoader(data_type=opt.data_type, img_dir_prefixs=label_img_path, mask_dir_prefixs=label_mask_path, contrast_dir_prefixs=label_contrast_path, bg_dir_prefixs=label_bg_path, input_type=opt.input_type, input_with_mask=opt.input_with_mask, subset='all', transform=resize)
# labelSet = completeLabelDataLoader(data_type=opt.data_type, img_dir_prefixs=label_img_path, mask_dir_prefixs=label_mask_path, contrast_dir_prefixs=label_contrast_path, bg_dir_prefixs=label_bg_path, input_type=opt.input_type, input_with_mask=opt.input_with_mask, subset='all', transform=resize, color_fea='innercontrast-bg-shape')
# labelSet = completeLabelDataLoader(data_type=opt.data_type, img_dir_prefixs=label_img_path, mask_dir_prefixs=label_mask_path, contrast_dir_prefixs=label_contrast_path, bg_dir_prefixs=label_bg_path, input_type=opt.input_type, input_with_mask=opt.input_with_mask, subset='all', transform=resize, color_fea='subsegment-contrast-bg-shape')
labelSet = completeLabelDataLoader(data_type=opt.data_type, img_dir_prefixs=label_img_path, mask_dir_prefixs=label_mask_path, contrast_dir_prefixs=label_contrast_path, bg_dir_prefixs=label_bg_path, input_type=opt.input_type, input_with_mask=opt.input_with_mask, subset='all', transform=resize, color_fea=opt.hand_color_fea)

trvalLD = DD.DataLoader(trvalSet, batch_size=opt.batchSize,
       sampler=DD.sampler.RandomSampler(trvalSet),
       num_workers=opt.workers, pin_memory=True)
trLD = DD.DataLoader(trSet, batch_size=opt.batchSize,
       sampler=DD.sampler.RandomSampler(trSet),
       num_workers=opt.workers, pin_memory=True)
valLD = DD.DataLoader(valSet, batch_size=opt.batchSize,
       sampler=DD.sampler.SequentialSampler(valSet),
       num_workers=opt.workers, pin_memory=True)
labelLD = DD.DataLoader(labelSet, batch_size=opt.batchSize,
       sampler=DD.sampler.SequentialSampler(labelSet),
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
    # if opt.input_with_mask and (opt.nc==4 or opt.nc==2):
    if opt.input_with_mask:
        misc.visualizeAsImages(input_img[:,:3].data.clone(),
                               opt.dirImageoutput,
                               filename='iter_' + str(epoch) + '_input_', n_sample=49, nrow=7, normalize=False)
        misc.visualizeAsImages(recons[:,:3].data.clone(),
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
    all_labels = []
    all_feats = []
    all_handfeats = []
    all_img_names = []
    # for batch_idx, (input_img, mask_img, input_labels, input_feas) in enumerate(trLD, 0):
    for batch_idx, (input_img, mask_img, input_labels, input_feas, input_img_names) in enumerate(labelLD, 0):
        gc.collect() # collect garbage
        ### prepare data ###
        input_data = input_img.cuda()
        input_labels = list(input_labels.numpy())
        all_labels.extend(input_labels)
        print(input_data.size())
        fea = encoders(input_data)
        recons = decoders(fea)
        # loss_recon = criterionRecon(recons, input_data, weight=1)
        
        batch_size_t = input_img.size(0)

        fea = fea.data.cpu().numpy()
        fea = np.reshape(fea, [batch_size_t, -1])
        input_feas = input_feas.numpy()
        print(input_feas.shape)
        all_handfeats.append(input_feas)
        # if opt.handfea:
        #     fea = np.concatenate([fea, input_feas], 1)
        all_feats.append(fea)
        all_img_names.append(input_img_names)
    all_feats = np.concatenate(all_feats)
    all_handfeats = np.concatenate(all_handfeats)
    all_img_names = np.concatenate(all_img_names)
    # print(all_img_names)
    # val_labels = []
    # val_feats = []
    # val_handfeats = []
    # for batch_idx, (input_img, mask_img, input_labels, input_feas) in enumerate(valLD, 0):
    #     gc.collect() # collect garbage
    #     ### prepare data ###
    #     input_data = input_img.cuda()
    #     input_labels = list(input_labels.numpy())
    #     val_labels.extend(input_labels)
        
    #     fea = encoders(input_data)
    #     recons = decoders(fea)
    #     # loss_recon = criterionRecon(recons, input_data, weight=1)
        
    #     batch_size_t = input_img.size(0)

    #     fea = fea.data.cpu().numpy()
    #     fea = np.reshape(fea, [batch_size_t, -1])
    #     input_feas = input_feas.numpy()
    #     val_handfeats.append(input_feas)
    #     # if opt.handfea:
    #     #     fea = np.concatenate([fea, input_feas], 1)
    #     val_feats.append(fea)
    # val_feats = np.concatenate(val_feats)
    # val_handfeats = np.concatenate(val_handfeats)

    # normalize handfeats
    mean_handfeat = np.mean(all_handfeats, axis=0, keepdims=True)
    std_handfeat = np.std(all_handfeats, axis=0, keepdims=True)
    all_handfeats -= mean_handfeat
    all_handfeats = all_handfeats / std_handfeat
    # val_handfeats -= mean_handfeat
    # val_handfeats = val_handfeats / std_handfeat

    # normalize fea
    if opt.normfea:
        mean_feat = np.mean(all_feats, axis=0, keepdims=True)
        # std_feat = np.std(train_feats, axis=0, keepdims=True)
        all_feats -= mean_feat
        # train_feats = train_feats / std_feat
        all_feats = all_feats / np.linalg.norm(all_feats, 2, axis=1, keepdims=True)
        # train_feats = train_feats / np.sum(train_feats, 1, keepdims=True)
        # val_feats -= mean_feat
        # # val_feats = val_feats / std_feat
        # val_feats = val_feats / np.linalg.norm(val_feats, 2, axis=1, keepdims=True)
        # # val_feats = val_feats / np.sum(val_feats, 1, keepdims=True)

    if opt.handfeaonly:
        all_feats = all_handfeats
        # val_feats = val_handfeats

    if opt.handfea:
        all_feats = np.concatenate([all_feats, all_handfeats], 1)
        # val_feats = np.concatenate([val_feats, val_handfeats], 1)

        if opt.normfea:
            mean_feat = np.mean(all_feats, axis=0, keepdims=True)
            std_feat = np.std(all_feats, axis=0, keepdims=True)
            all_feats -= mean_feat
            # train_feats = train_feats / std_feat
            all_feats = all_feats / np.linalg.norm(all_feats, 2, axis=1, keepdims=True)
            # train_feats = train_feats / np.sum(train_feats, 1, keepdims=True)
            # val_feats -= mean_feat
            # # val_feats = val_feats / std_feat
            # val_feats = val_feats / np.linalg.norm(val_feats, 2, axis=1, keepdims=True)
            # # val_feats = val_feats / np.sum(val_feats, 1, keepdims=True)

    from sklearn.svm import LinearSVC, SVC
    # C = opt.C

    num_data = len(all_feats)
    # cross validation
    n_cross = 5
    graphene_idx = [i for i in range(num_data) if all_labels[i] == 1]
    others_idx = [i for i in range(num_data) if all_labels[i] == 0]
    print('num of graphene: %d, number of others: %d'%(len(graphene_idx), len(others_idx)))
    shuffle_graphene_idxes = np.random.RandomState(seed=123).permutation(len(graphene_idx))
    shuffle_others_idxes = np.random.RandomState(seed=123).permutation(len(others_idx))
    val_group_graphene_idxes = []
    val_group_others_idxes = []
    for ni in range(n_cross):
        tmp_idx = [i*n_cross + ni for i in range(len(graphene_idx) // n_cross+1) if i*n_cross + ni < len(graphene_idx)]
        val_group_graphene_idxes.append(tmp_idx)
        tmp_idx = [i*n_cross + ni for i in range(len(others_idx) // n_cross+1) if i*n_cross + ni < len(others_idx)]
        val_group_others_idxes.append(tmp_idx)

    methods = ['linearsvm', 'rbf']
    Cs = [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 50]
    # Cs = [2]

    clf_save_dir = opt.output_dir_prefix + 'doublecheck_test_handfea-%d_handfeaonly-%d_handcolorfea-%s'%(opt.handfea, opt.handfeaonly, opt.hand_color_fea)
    if not os.path.exists(clf_save_dir):
        os.makedirs(clf_save_dir)

    from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, precision_recall_curve
            
    for method in methods:
        for C in Cs:
            train_aps = []
            train_accs = []
            val_aps = []
            val_accs = []
            val_confs = np.zeros([2,2])
            val_pred_scores_all_fold = []
            l_val_labels_all_fold = []
            # split data
            test_clf_save_dir = os.path.join(clf_save_dir, '%s-%f'%(method, C))
            if not os.path.exists(test_clf_save_dir):
                os.makedirs(test_clf_save_dir)

            for ni in range(n_cross):
                val_graphene_feats = np.array([all_feats[graphene_idx[ij],:] for ij in val_group_graphene_idxes[ni]])
                tr_graphene_feats = np.array([all_feats[graphene_idx[ij],:] for ij in range(len(graphene_idx)) if ij not in val_group_others_idxes[ni]])
                val_others_feats = np.array([all_feats[others_idx[ij],:] for ij in val_group_others_idxes[ni] ])
                tr_others_feats = np.array([all_feats[others_idx[ij],:] for ij in range(len(others_idx)) if ij not in val_group_others_idxes[ni] ])

                val_graphene_names = [all_img_names[graphene_idx[ij]] for ij in val_group_graphene_idxes[ni]]
                val_others_names = [all_img_names[others_idx[ij]] for ij in val_group_others_idxes[ni]]
                val_names = val_graphene_names + val_others_names

                # print('train num of graphene: %d, number of others: %d'%(len(tr_graphene_feats), len(tr_others_feats)))
                # print('val num of graphene: %d, number of others: %d'%(len(val_graphene_feats), len(val_others_feats)))
                train_feats = np.concatenate([tr_graphene_feats, tr_others_feats])
                train_labels = np.concatenate([np.ones([len(tr_graphene_feats)]), np.zeros([len(tr_others_feats)])])

                val_feats = np.concatenate([val_graphene_feats, val_others_feats])
                val_labels = np.concatenate([np.ones([len(val_graphene_feats)]), np.zeros([len(val_others_feats)])])

                if method == 'linearsvm':
                    clf = LinearSVC(random_state=0, tol=1e-5, C=C, max_iter=5e4, class_weight={0:1, 1:10})
                elif method == 'rbf':
                    # compute gamma,
                    pair_dist = scipy.spatial.distance.pdist(train_feats, metric='euclidean')
                    gamma = 1.0 / np.mean(pair_dist)
                    clf = SVC(kernel='rbf', gamma=gamma, random_state=0, tol=1e-5, C=C, max_iter=5e4, class_weight={0:1, 1:10})
                else:
                    raise NotImplementedError
                clf.fit(train_feats, train_labels)

                train_pred_cls = clf.predict(train_feats)
                train_pred_scores = clf.decision_function(train_feats)
                val_pred_cls = clf.predict(val_feats)
                val_pred_scores = clf.decision_function(val_feats)

                # # save images
                # for imx in range(val_feats.shape[0]):
                #     ori_name = os.path.join('../../results/data_111x_individual_script/ae_complete_doublecheck_label_imgcontour', 'all', val_names[imx])
                #     if val_pred_scores[imx] > 0:
                #         new_name = os.path.join(test_clf_save_dir, 'gt_' + val_names[imx] + '_predict_%s_score_%.3f.png'%('graphene', val_pred_scores[imx]))
                #     else:
                #         new_name = os.path.join(test_clf_save_dir, 'gt_' + val_names[imx] + '_predict_%s_score_%.3f.png'%('others', val_pred_scores[imx]))
                #     os.system('cp %s %s'%(ori_name, new_name))


                from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix
                train_acc = accuracy_score(train_labels, train_pred_cls)
                train_accs.append(train_acc)
                val_acc = accuracy_score(val_labels, val_pred_cls)
                val_accs.append(val_acc)
                # train_conf = confusion_matrix(train_labels, train_pred_cls)
                # train_conf = train_conf / np.sum(train_conf, 1, keepdims=True)
                val_conf = confusion_matrix(val_labels, val_pred_cls)
                # val_conf = val_conf / np.sum(val_conf, 1, keepdims=True)
                val_confs += val_conf

                # calculate map:
                l_train_labels = [_ == 1 for _ in train_labels]
                l_val_labels = [_ == 1 for _ in val_labels]
                train_aps.append(average_precision_score(l_train_labels, train_pred_scores))
                val_aps.append(average_precision_score(l_val_labels, val_pred_scores))
                l_val_labels_all_fold.extend(l_val_labels)
                val_pred_scores_all_fold.append(val_pred_scores)

            # val_pred_scores_all_fold = np.concatenate(val_pred_scores_all_fold)
            # precision_l, recall_l, _ = precision_recall_curve(np.array(l_val_labels_all_fold, dtype=np.uint8), val_pred_scores_all_fold)
            # print(getPrecisionAtRecall(precision_l, recall_l, 0.90), getPrecisionAtRecall(precision_l, recall_l, 0.95) )
            
                # print(train_aps)
                # print(val_aps)
            print(val_confs)
            print('%s-%f: train: %.4f, val: %.4f, ap train: %4f, ap val: %4f' % (method, C, np.mean(train_accs), np.mean(val_accs), np.mean(train_aps), np.mean(val_aps)))

