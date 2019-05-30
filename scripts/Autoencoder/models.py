import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# only calculate loss if mask is true
class MaskABSLoss(nn.Module):
    def __init__(self):
        super(MaskABSLoss, self).__init__()
        self.criterion = nn.L1Loss(reduction='none')

    # input: [B, C, H, W]
    # mask: [B, 1,  H, W]
    def forward(self, input, target, mask):
        mask = mask.squeeze(1)
        loss = self.criterion(input, target)
        loss = loss.mean(1) * mask
        # print(mask.shape)
        # print(mask.sum(1).sum(1))
        # add 1 to avoid mask become empty
        loss = loss.sum(1).sum(1) / (1+mask.sum(1).sum(1))
        loss = loss.mean()
        return loss


class Encoders(nn.Module):
    def __init__(self, opt):
        super(Encoders, self).__init__()
        self.ngpu = opt.ngpu
        self.opt = opt

        self.encoder = nn.Sequential()
        self.encoder.add_module('layer0_conv_%d_%d'%(opt.nc, opt.ndf), nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False))
        self.encoder.add_module('layer0_bn', nn.BatchNorm2d(opt.ndf))
        self.encoder.add_module('layer0_lrelu',nn.LeakyReLU(0.2, inplace=True))
        csize = opt.imgSize // 2
        cndf = opt.ndf

        layer_cnt = 1
        while csize > 4:
            in_feat = cndf
            out_feat = min(cndf * 2, 256)
            self.encoder.add_module('layer%d_conv_%d_%d'%(layer_cnt, in_feat, out_feat), nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            self.encoder.add_module('layer%d_bn'%(layer_cnt), nn.BatchNorm2d(out_feat))
            self.encoder.add_module('layer%d_lrelu'%(layer_cnt), nn.LeakyReLU(0.2, inplace=True))

            layer_cnt += 1
            cndf = out_feat
            csize = csize // 2

        # input size: [256, 4, 4]
        self.encoder.add_module('final_conv_%d_%d'%(cndf, opt.zdim), nn.Conv2d(cndf, opt.zdim, 4, 4, 0, bias=False))
        self.encoder.add_module('final_sigmoid', nn.Sigmoid())
        # self.encoder = nn.Sequential(
        #     # input is (nc) x 256 x 256
        #     nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(opt.ndf),
        #     nn.LeakyReLU(0.2, False),
        #     # input is (opt.ndf) x 128 x 128
        #     nn.Conv2d(opt.ndf, opt.ndf, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(opt.ndf),
        #     nn.LeakyReLU(0.2, False),
        #     # input is (opt.ndf) x 64 x 64
        #     nn.Conv2d(opt.ndf, opt.ndf, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(opt.ndf),
        #     nn.LeakyReLU(0.2, False),
        #     # state size. (ndf) x 32 x 32
        #     nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(opt.ndf * 2),
        #     nn.LeakyReLU(0.2, False),
        #     # state size. (ndf*2) x 16 x 16
        #     nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(opt.ndf * 4),
        #     nn.LeakyReLU(0.2, False),
        #     # state size. (ndf*4) x 8 x 8
        #     nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(opt.ndf * 8),
        #     nn.LeakyReLU(0.2, False),
        #     # state size. (ndf*8) x 4 x 4
        #     nn.Conv2d(opt.ndf * 8, opt.zdim, 4, 4, 0, bias=False),
        #     nn.Sigmoid()
        # )
    def forward(self, input):
        self.z = self.encoder(input).view(-1, self.opt.zdim)
        return self.z


class Decoders(nn.Module):
    def __init__(self, opt):
        super(Decoders, self).__init__()
        self.ngpu = opt.ngpu
        self.opt = opt
        self.decoder = nn.Sequential()

        cngf = 256
        self.decoder.add_module('layer0_convt_%d_%d' % (opt.zdim, cngf),
                                nn.ConvTranspose2d(opt.zdim, cngf, 4, 1, 0, bias=False))
        self.decoder.add_module('layer0_bn', nn.BatchNorm2d(cngf))
        self.decoder.add_module('layer0_lrelu', nn.ReLU(True))
        csize = 4

        layer_cnt = 1
        while csize < opt.imgSize // 2:
            in_feat = cngf
            out_feat = max(cngf//2, 32)
            self.decoder.add_module('layer%d_convt_%d_%d' % (layer_cnt, in_feat, out_feat),
                                    nn.ConvTranspose2d(in_feat, out_feat, 4, 2, 1, bias=False))
            self.decoder.add_module('layer%d_bn'%(layer_cnt), nn.BatchNorm2d(out_feat))
            self.decoder.add_module('layer%d_lrelu'%(layer_cnt), nn.ReLU(True))

            layer_cnt += 1
            csize = csize * 2
            cngf = out_feat

        self.decoder.add_module('final_convt_%d_%d' % (cngf, opt.nc),
                                nn.ConvTranspose2d(cngf, opt.nc, 4, 2, 1, bias=False))
        self.decoder.add_module('final_hardtanh', nn.Hardtanh(0,1))

        # self.decoder = nn.Sequential(
        #     # input is Z, going into a convolution
        #     nn.ConvTranspose2d(opt.zdim, opt.ngf * 8, 4, 1, 0, bias=False),
        #     nn.BatchNorm2d(opt.ngf * 8),
        #     nn.ReLU(True),
        #     # state size. (ngf*8) x 4 x 4
        #     nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(opt.ngf * 4),
        #     nn.ReLU(True),
        #     # state size. (ngf*4) x 8 x 8
        #     nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(opt.ngf * 2),
        #     nn.ReLU(True),
        #     # state size. (ngf*2) x 16 x 16
        #     nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(opt.ngf),
        #     nn.ReLU(True),
        #     # state size. (ngf) x 32 x 32
        #     nn.ConvTranspose2d(opt.ngf, opt.ngf, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(opt.ngf),
        #     nn.ReLU(True),
        #     # state size. (ngf) x 64 x 64
        #     nn.ConvTranspose2d(opt.ngf, opt.ngf, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(opt.ngf),
        #     nn.ReLU(True),
        #     # state size. (ngf) x 128 x 128
        #     nn.ConvTranspose2d(opt.ngf, opt.ngf, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(opt.ngf),
        #     nn.ReLU(True),
        #     # state size. (nc) x 256 x 256
        #     nn.ConvTranspose2d(opt.ngf, opt.nc, 3, 1, 1, bias=False),
        #     nn.Hardtanh(0,1)
        # )

    def forward(self, input):
        self.output = self.decoder(input.view(-1, self.opt.zdim, 1, 1))
        return self.output



class mlp_classify(nn.Module):
    def __init__(self, opt, output_dim=2):
        super(mlp_classify, self).__init__()
        self.main = nn.Sequential(nn.Linear(opt.fea_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, False),
            nn.Linear(32, output_dim))

    def forward(self, input):
        self.output = self.main(input)
        return self.output