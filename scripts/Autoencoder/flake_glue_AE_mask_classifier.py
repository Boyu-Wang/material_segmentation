"""
Use the pretrained auto encoder to extract features, add a classifier

By: Boyu Wang (boywang@cs.stonybrook.edu)
Created Data: 24 Apr 2019
Last Modified Date: 9 May 2019
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
import torchvision.transforms as transforms
import torch.utils.data as DD
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, precision_recall_curve

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
parser.add_argument('--modelPath', type=str, default='', help="model path to load")
parser.add_argument('--manualSeed', type=int, default=543, help='manual seed')
parser.add_argument('--epoch_iter', type=int, default=200, help='number of epochs on entire dataset')
parser.add_argument('--data_type', type=str, default='allexp', help='data_type: allexp, allsubexp, singlesubexp')
parser.add_argument('--input_type', type=str, default='rgb', help='input type: rgb, gray, hsv, rgb-gray-hsv')
parser.add_argument('--input_with_mask', type=int, default=1, help='whether to input mask')
parser.add_argument('--evaluate', type=int, default=0, help='1 for evaluate')
parser.add_argument('--ft', type=int, default=1, help='whether to finetune the autoencoder')
parser.add_argument('--fea_type', type=str, default='ae-hand', help='feature type: ae, hand, ae-hand')
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpuid)
# misc
opt.dirCheckpoints = '../../results/AE_%s_input-%s_inputmask-%d/complete_classifier_fea-%s_ft-%d' % (opt.data_type, opt.input_type, opt.input_with_mask, opt.fea_type, opt.ft)
opt.ae_modelPath = '../../results/AE_%s_input-%s_inputmask-%d/checkpoints/' % (opt.data_type, opt.input_type, opt.input_with_mask)

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

opt.fea_dim=0
if 'ae' in opt.fea_type:
    opt.fea_dim += opt.zdim
if 'hand' in opt.fea_type:
    opt.fea_dim += 51

print(opt)

try:
    os.makedirs(opt.dirCheckpoints)
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
# decoders = models.Decoders(opt)
mlp_classify = models.mlp_classify(opt, output_dim=3)


if opt.cuda:
    encoders.cuda()
    # decoders.cuda()
    mlp_classify.cuda()


print('load pretrained autoencoder at: ' + opt.ae_modelPath)
encoders.load_state_dict(torch.load(opt.ae_modelPath + 'model_epoch_499_encoders.pth'))
# decoders.load_state_dict(torch.load(opt.ae_modelPath + 'model_epoch_499_decoders.pth'))
mlp_classify.apply(misc.weights_init)

if opt.ft:
    updator_encoders = optim.Adam(encoders.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    # updator_decoders = optim.Adam(decoders.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

updator_classify = optim.Adam(mlp_classify.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# criterionRecon = models.MaskABSLoss()
criterionClassify = nn.CrossEntropyLoss(weight=torch.Tensor([1, 5, 2]).cuda())

# label_img_path = '../../results/data_jan2019_script/ae_label_img'
# label_mask_path = '../../results/data_jan2019_script/ae_label_mask'
# label_contrast_path = '../../results/data_jan2019_script/ae_contrast'
label_img_path = '../../results/data_jan2019_script/ae_complete_label_img'
label_mask_path = '../../results/data_jan2019_script/ae_complete_label_mask'
label_contrast_path = '../../results/data_jan2019_script/ae_complete_label_contrast'

resize = transforms.Compose([transforms.ToTensor()])
# trSet = LabelDataLoader(img_dir_prefix=label_img_path, mask_dir_prefix=label_mask_path, input_type=opt.input_type,
#                         input_with_mask=opt.input_with_mask, subset='train', transform=resize)
# valSet = LabelDataLoader(img_dir_prefix=label_img_path, mask_dir_prefix=label_mask_path, input_type=opt.input_type,
#                          input_with_mask=opt.input_with_mask, subset='val', transform=resize)
trSet = completeLabelDataLoader(img_dir_prefix=label_img_path, mask_dir_prefix=label_mask_path, contrast_dir_prefix=label_contrast_path, input_type=opt.input_type, input_with_mask=opt.input_with_mask, subset='train', transform=resize)
valSet = completeLabelDataLoader(img_dir_prefix=label_img_path, mask_dir_prefix=label_mask_path, contrast_dir_prefix=label_contrast_path, input_type=opt.input_type, input_with_mask=opt.input_with_mask, subset='val', transform=resize)

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

iter_mark = 0
train_losses = []
val_losses = []
val_accs = []

best_val_mAP = 0


# train_labels: [num], label is 0 based
# train_pred_scores: [num, nClass]
def compute_aps(train_labels, train_pred_scores, fig_save_name):
    num_class = train_pred_scores.shape[1]
    aps = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    legends = ['thick', 'thin', 'glue']

    for l in range(num_class):
        l_train_labels = [_ == l for _ in train_labels]
        aps.append(average_precision_score(l_train_labels, train_pred_scores[:, l]))
        precision_l, recall_l, _ = precision_recall_curve(np.array(l_train_labels, dtype=np.uint8), train_pred_scores[:, l])
        ax.plot(recall_l, precision_l, label=legends[l])
        print(l, getPrecisionAtRecall(precision_l, recall_l, 0.90), getPrecisionAtRecall(precision_l, recall_l, 0.95) )

    plt.legend()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    plt.savefig(fig_save_name, dpi=300)
    plt.close(fig)

    return aps, np.mean(aps)


def getPrecisionAtRecall(precision, recall, rate=0.95):
    # find the recall which is the first one that small or equal to rate.
    for id, r in enumerate(recall):
        if r <= rate:
            break
    return precision[id]


for epoch in range(opt.epoch_iter):
    train_loss = torch.zeros(1).cuda()
    train_amount = 0
    val_loss = torch.zeros(1).cuda()
    val_amount = 0
    gc.collect()  # collect garbage
    if opt.ft:
        encoders.train()
    else:
        encoders.eval()
    # decoders.train()
    mlp_classify.train()
    # for dataroot in TrainingData:
    if not doTraining:
        break

    train_gtlabels = []
    train_predlabels = []
    train_predscores = []
    for batch_idx, (input_img, mask_img, input_labels, input_feas) in enumerate(trLD, 0):
        gc.collect()  # collect garbage
        ### prepare data ###
        input_data = input_img.cuda()
        target_data = input_img
        mask_img = mask_img.cuda()
        if opt.ft:
            updator_encoders.zero_grad()
        updator_classify.zero_grad()

        # # map label from -1,1 to 0,1
        # input_labels = (input_labels + 1) / 2
        input_labels = input_labels.cuda().long()

        if opt.fea_type == 'ae':
            fea = encoders(input_data)
        elif opt.fea_type == 'hand':
            fea = input_feas.cuda().float()
        elif opt.fea_type == 'ae-hand':
            ae_fea = encoders(input_data)
            hand_fea = input_feas.cuda().float()
            fea = torch.cat([ae_fea, hand_fea], 1)
        else:
            raise NotImplementedError
        logits = mlp_classify(fea)

        loss_clf = criterionClassify(logits, input_labels)
        loss_clf.backward(retain_graph=True)

        updator_classify.step()
        if opt.ft:
            updator_encoders.step()

        batch_size_t = input_img.size(0)
        train_loss += loss_clf.data.item() * batch_size_t
        iter_mark += 1
        train_amount += batch_size_t

        train_gtlabels.append(input_labels.cpu().numpy())
        train_predlabels.append(logits.max(1)[1].data.cpu().numpy())
        train_predscores.append(logits.data.cpu().numpy())

    train_gtlabels = np.concatenate(train_gtlabels)
    train_predlabels = np.concatenate(train_predlabels)
    train_predscores = np.concatenate(train_predscores)
    train_acc = accuracy_score(train_gtlabels, train_predlabels)
    train_aps, train_mAP = compute_aps(train_gtlabels, train_predscores, os.path.join(opt.dirCheckpoints, 'train_prc.png'))
    train_loss /= train_amount
    train_losses.append(train_loss.cpu().numpy())
    print('====> Epoch: {} Average training loss:'.format(epoch))
    print(train_loss.data.item(), train_aps, train_mAP)

    # evaluate on validation set
    if epoch % 1 == 0:
        encoders.eval()
        mlp_classify.eval()
        val_gtlabels = []
        val_predlabels = []
        val_predscores = []
        for batch_idx, (input_img, mask_img, input_labels, input_feas) in enumerate(valLD, 0):
            gc.collect()  # collect garbage
            ### prepare data ###
            input_data = input_img.cuda()
            mask_img = mask_img.cuda()
            # input_labels = (input_labels + 1) / 2
            input_labels = input_labels.cuda().long()

            if opt.fea_type == 'ae':
                fea = encoders(input_data)
            elif opt.fea_type == 'hand':
                fea = input_feas.cuda().float()
            elif opt.fea_type == 'ae-hand':
                ae_fea = encoders(input_data)
                hand_fea = input_feas.cuda().float()
                fea = torch.cat([ae_fea, hand_fea], 1)
            else:
                raise NotImplementedError
            logits = mlp_classify(fea)

            loss_clf = criterionClassify(logits, input_labels)

            batch_size_t = input_img.size(0)
            val_loss += loss_clf.data.item() * batch_size_t
            val_amount += batch_size_t

            val_gtlabels.append(input_labels.cpu().numpy())
            val_predlabels.append(logits.max(1)[1].data.cpu().numpy())
            val_predscores.append(logits.data.cpu().numpy())

        val_loss /= val_amount
        val_losses.append(val_loss.cpu().numpy())
        val_gtlabels = np.concatenate(val_gtlabels)
        val_predlabels = np.concatenate(val_predlabels)
        val_predscores = np.concatenate(val_predscores)
        val_acc = accuracy_score(val_gtlabels, val_predlabels)
        val_aps, val_mAP = compute_aps(val_gtlabels, val_predscores, os.path.join(opt.dirCheckpoints, 'val_prc.png'))
        print(val_loss.data.item(), val_aps, val_mAP)
        is_best = val_mAP > best_val_mAP
        best_val_mAP = max(val_mAP, best_val_mAP)
        # do checkpointing
        torch.save(mlp_classify.state_dict(), '%s/model_epoch_%d_mlp.pth' % (opt.dirCheckpoints, epoch))
        if opt.ft:
            torch.save(encoders.state_dict(), '%s/model_epoch_%d_encoders.pth' % (opt.dirCheckpoints, epoch))
        if is_best:
            torch.save(mlp_classify.state_dict(), '%s/model_epoch_best_mlp.pth' % (opt.dirCheckpoints))
            if opt.ft:
                torch.save(encoders.state_dict(), '%s/model_epoch_best_encoders.pth' % (opt.dirCheckpoints))

if doTesting:
    print('loading pretrained model')
    if opt.ft:
        encoders.load_state_dict(torch.load('%s/model_epoch_best_encoders.pth' % (opt.dirCheckpoints)))
    mlp_classify.load_state_dict(torch.load('%s/model_epoch_best_mlp.pth' % (opt.dirCheckpoints)))
    # ------------ testing ------------ #
    encoders.eval()
    mlp_classify.eval()

    val_gtlabels = []
    val_predlabels = []
    val_predscores = []
    val_loss = torch.zeros(1).cuda()
    val_amount = 0
    for batch_idx, (input_img, mask_img, input_labels, input_feas) in enumerate(valLD, 0):
        gc.collect()  # collect garbage
        ### prepare data ###
        input_data = input_img.cuda()
        mask_img = mask_img.cuda()
        # input_labels = (input_labels + 1) / 2
        input_labels = input_labels.cuda().long()

        if opt.fea_type == 'ae':
            fea = encoders(input_data)
        elif opt.fea_type == 'hand':
            fea = input_feas.cuda().float()
        elif opt.fea_type == 'ae-hand':
            ae_fea = encoders(input_data)
            hand_fea = input_feas.cuda().float()
            fea = torch.cat([ae_fea, hand_fea], 1)
        else:
            raise NotImplementedError
        logits = mlp_classify(fea)

        loss_clf = criterionClassify(logits, input_labels)

        batch_size_t = input_img.size(0)
        val_loss += loss_clf.data.item() * batch_size_t
        val_amount += batch_size_t

        val_gtlabels.append(input_labels.cpu().numpy())
        val_predlabels.append(logits.max(1)[1].data.cpu().numpy())
        val_predscores.append(logits.data.cpu().numpy())

    val_loss /= val_amount
    val_losses.append(val_loss.cpu().numpy())
    val_gtlabels = np.concatenate(val_gtlabels)
    val_predlabels = np.concatenate(val_predlabels)
    val_predscores = np.concatenate(val_predscores)
    val_acc = accuracy_score(val_gtlabels, val_predlabels)
    val_aps, val_mAP = compute_aps(val_gtlabels, val_predscores, os.path.join(opt.dirCheckpoints, 'val_prc.png'))
    print(val_loss.data.item(), val_aps, val_mAP)
    val_conf = confusion_matrix(val_gtlabels, val_predlabels)
    val_conf = val_conf / np.sum(val_conf, 1, keepdims=True)
    print(val_conf)
