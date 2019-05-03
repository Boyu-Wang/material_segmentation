import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
from torch.autograd import Function
from torch.autograd import Variable
from torch.autograd import gradcheck
from torch.autograd import Function
import numpy as np


class WeightABSLoss(nn.Module):
    def __init__(self,opt):
        super(WeightABSLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.opt=opt
    def forward(self, input, target, weight=1):
        w = torch.cuda.FloatTensor(1).fill_(weight)
        if self.opt.cuda:
            w.cuda()
        w = Variable(w, requires_grad=False)
        self.loss = w*self.criterion(input, target)
        return self.loss


class Encoders(nn.Module):
    def __init__(self, opt):
        super(Encoders, self).__init__()
        self.ngpu = opt.ngpu
        self.opt = opt
        self.encoder = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf),
            nn.LeakyReLU(0.2, False),
            # input is (opt.ndf) x 128 x 128
            nn.Conv2d(opt.ndf, opt.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf),
            nn.LeakyReLU(0.2, False),
            # input is (opt.ndf) x 64 x 64
            nn.Conv2d(opt.ndf, opt.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(opt.ndf * 8, opt.zdim, 4, 4, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        self.z = self.encoder(input).view(-1, self.opt.zdim)
        return self.z


class Decoders(nn.Module):
    def __init__(self, opt):
        super(Decoders, self).__init__()
        self.ngpu = opt.ngpu
        self.opt = opt
        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(opt.zdim, opt.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(opt.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(opt.ngf, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(opt.ngf, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 128 x 128
            nn.ConvTranspose2d(opt.ngf, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.ReLU(True),
            # state size. (nc) x 256 x 256
            nn.ConvTranspose2d(opt.ngf, opt.nc, 3, 1, 1, bias=False),
            nn.Hardtanh(0,1)
        )

    def forward(self, input):
        self.output = self.decoder(input.view(-1, self.opt.zdim, 1, 1))
        return self.output



class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.ngpu = opt.ngpu
        self.opt = opt
        self.encoder = nn.Sequential(
            # input is (nc*2) x 64 x 64
            nn.Conv2d(opt.nc*2, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 2),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 4),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 8),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(opt.ndf * 8, 1, 4, 4, 0, bias=False),
            # nn.Sigmoid()
        )
    def forward(self, input):
        self.z = self.encoder(input).view(-1, 1).squeeze(1)
        return self.z


# map 128*128 image to 64*64, to fit the pretrained encoders
class Encoders_128_preprocess(nn.Module):
    def __init__(self, opt):
        super(Encoders_128_preprocess, self).__init__()
        self.ngpu = opt.ngpu
        self.opt = opt
        self.downsample = nn.MaxPool2d(3,2,1)
        self.encoder = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(opt.ndf, 1, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(False),
            # state size. (nc) x 64 x 64
        )
    def forward(self, input, alpha):
        downsampled_input = self.downsample(input)
        output = downsampled_input * (1 - alpha) + alpha * self.encoder(input)
        return output


# map 64*64 to 128*128
class Decoders_128_postprocess(nn.Module):
    def __init__(self, opt):
        super(Decoders_128_postprocess, self).__init__()
        self.ngpu = opt.ngpu
        self.opt = opt
        self.upsample = nn.Upsample(scale_factor=2)
        self.decoder = nn.Sequential(
            # input size. (nc) x 64 x 64
            nn.ConvTranspose2d(opt.nc, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 128 x 128
            nn.ConvTranspose2d(opt.ngf, opt.nc, 3, 1, 1, bias=False),
            nn.Hardtanh(0,1)
        )

    def forward(self, input, alpha):
        upsampled_input = self.upsample(input)
        output = upsampled_input * (1 - alpha) + alpha * self.decoder(input)
        return output


# map 256*256 image to 128*128, to fit the pretrained encoders
class Encoders_256_preprocess(nn.Module):
    def __init__(self, opt):
        super(Encoders_256_preprocess, self).__init__()
        self.ngpu = opt.ngpu
        self.opt = opt
        self.downsample = nn.MaxPool2d(3,2,1)
        self.encoder = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, False),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(opt.ndf, 1, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(False),
            # state size. (nc) x 128 x 128
        )
    def forward(self, input, alpha):
        downsampled_input = self.downsample(input)
        output = downsampled_input * (1 - alpha) + alpha * self.encoder(input)
        return output


# map 128*128 to 256*256
class Decoders_256_postprocess(nn.Module):
    def __init__(self, opt):
        super(Decoders_256_postprocess, self).__init__()
        self.ngpu = opt.ngpu
        self.opt = opt
        self.upsample = nn.Upsample(scale_factor=2)
        self.decoder = nn.Sequential(
            # input size. (nc) x 128 x 128
            nn.ConvTranspose2d(opt.nc, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 256 x 256
            nn.ConvTranspose2d(opt.ngf, opt.nc, 3, 1, 1, bias=False),
            nn.Hardtanh(0,1)
        )

    def forward(self, input, alpha):
        upsampled_input = self.upsample(input)
        output = upsampled_input * (1 - alpha) + alpha * self.decoder(input)
        return output


class mlp_angle_reg(nn.Module):
    def __init__(self, opt):
        super(mlp_angle_reg, self).__init__()
        self.main = nn.Sequential(nn.Linear(opt.zdim, 128),
            nn.LeakyReLU(0.2, False),
            nn.Linear(128, 3),
            nn.ReLU(True))

    def forward(self, input):
        self.output = self.main(input)
        return self.output


class mlp_reflcunderlyparam_reg(nn.Module):
    def __init__(self, opt):
        super(mlp_reflcunderlyparam_reg, self).__init__()
        self.main = nn.Sequential(nn.Linear(opt.zdim, 128),
            nn.LeakyReLU(0.2, False),
            nn.Linear(128, 5),
            nn.ReLU(True))

    def forward(self, input):
        self.output = self.main(input)
        return self.output

class mlp_reflcunderlyparam_reg_norelu(nn.Module):
    def __init__(self, opt):
        super(mlp_reflcunderlyparam_reg_norelu, self).__init__()
        self.main = nn.Sequential(nn.Linear(opt.zdim, 128),
            nn.LeakyReLU(0.2, False),
            nn.Linear(128, 5))

    def forward(self, input):
        self.output = self.main(input)
        return self.output


# class mlp_reflc_reg(nn.Module):
#     def __init__(self, opt):
#         super(mlp_reflc_reg, self).__init__()
#         self.main = nn.Sequential(nn.Linear(opt.zdim, 256),
#             nn.LeakyReLU(0.2, False),
#             nn.Linear(256, 512),
#             nn.LeakyReLU(0.2, False),
#             nn.Linear(512, 1000),
#             nn.LeakyReLU(0.2, False),
#             nn.Linear(1000, 2000),
#             nn.ReLU(True))
#         self.w = torch.cat([-torch.ones(1000), torch.ones(1000)]).cuda()
#         self.w = Variable(self.w, requires_grad=False)

#     def forward(self, input):
#         # self.output = -1 * self.main(input)
#         # output = self.main(input).view([-1, 2, 1000])
#         self.output = self.w * self.main(input)
#         return self.output


class mlp_reflc_reg(nn.Module):
    def __init__(self, opt):
        super(mlp_reflc_reg, self).__init__()
        self.main = nn.Sequential(nn.Linear(opt.zdim, 256),
            nn.LeakyReLU(0.2, False),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, False),
            nn.Linear(512, 1000),
            nn.LeakyReLU(0.2, False),
            nn.Linear(1000, 1000),
            nn.ReLU(True))
        # self.w = torch.cat([-torch.ones(1000), torch.ones(1000)]).cuda()
        # self.w = Variable(self.w, requires_grad=False)

    def forward(self, input):
        # self.output = -1 * self.main(input)
        # output = self.main(input).view([-1, 2, 1000])
        self.output = -1 * self.main(input)
        return self.output


class mlp_reflc_reg_conv(nn.Module):
    def __init__(self, opt):
        super(mlp_reflc_reg_conv, self).__init__()
        self.opt = opt
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose1d(opt.zdim, opt.ngf * 8, 7, 1, 0, bias=False),
            nn.BatchNorm1d(opt.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 7
            nn.ConvTranspose1d(opt.ngf * 8, opt.ngf * 4, 5, 3, 1, bias=False),
            nn.BatchNorm1d(opt.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 21
            nn.ConvTranspose1d(opt.ngf * 4, opt.ngf * 2, 5, 3, 1, bias=False),
            nn.BatchNorm1d(opt.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 63 
            nn.ConvTranspose1d(opt.ngf * 2, opt.ngf, 5, 2, 2, bias=False),
            nn.BatchNorm1d(opt.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 125
            nn.ConvTranspose1d(opt.ngf, opt.ngf/2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ngf/2),
            nn.ReLU(True),
            # state size. (ngf/2) x 250
            nn.ConvTranspose1d(opt.ngf/2, opt.ngf/4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ngf/4),
            nn.ReLU(True),
            # state size. (ngf/4) x 500
            nn.ConvTranspose1d(opt.ngf/4, 1, 4, 2, 1, bias=False),
            nn.ReLU(True))
            # state size. (1) x 1000

    def forward(self, input):
        self.output = -1 * self.main(input.view([-1, self.opt.zdim, 1]))
        self.output = self.output.squeeze(1)
        return self.output


# first output length of 64, then 1000
class mlp_reflc_reg_conv_twostep(nn.Module):
    def __init__(self, opt):
        super(mlp_reflc_reg_conv_twostep, self).__init__()
        self.opt = opt
        self.main_1st = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose1d(opt.zdim, opt.ngf * 8, 7, 1, 0, bias=False),
            nn.BatchNorm1d(opt.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 7
            nn.ConvTranspose1d(opt.ngf * 8, opt.ngf * 4, 5, 3, 1, bias=False),
            nn.BatchNorm1d(opt.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 21
            nn.ConvTranspose1d(opt.ngf * 4, 1, 4, 3, 0, bias=False),
            # nn.BatchNorm1d(opt.ngf * 2),
            nn.ReLU(True))
            # state size. 1 x 64
        self.main_2nd = nn.Sequential(
            nn.ConvTranspose1d(1, opt.ngf*2, 1, 1, 0, bias=False),
            nn.BatchNorm1d(opt.ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose1d(opt.ngf * 2, opt.ngf, 3, 2, 2, bias=False),
            nn.BatchNorm1d(opt.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 125
            nn.ConvTranspose1d(opt.ngf, opt.ngf/2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ngf/2),
            nn.ReLU(True),
            # state size. (ngf/2) x 250
            nn.ConvTranspose1d(opt.ngf/2, opt.ngf/4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ngf/4),
            nn.ReLU(True),
            # state size. (ngf/4) x 500
            nn.ConvTranspose1d(opt.ngf/4, 1, 4, 2, 1, bias=False),
            nn.ReLU(True))
            # state size. (1) x 1000

    def forward(self, input):
        self.output_1st = -1 * self.main_1st(input.view([-1, self.opt.zdim, 1]))
        self.output_2nd = -1 * self.main_2nd(self.output_1st)
        self.output_1st = self.output_1st.squeeze(1)
        self.output_2nd = self.output_2nd.squeeze(1)
        return self.output_1st, self.output_2nd


# first output length of 64, then 1000
class mlp_reflc_reg_conv_twostep_norelu(nn.Module):
    def __init__(self, opt):
        super(mlp_reflc_reg_conv_twostep_norelu, self).__init__()
        self.opt = opt
        self.main_1st = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose1d(opt.zdim, opt.ngf * 8, 7, 1, 0, bias=False),
            nn.BatchNorm1d(opt.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 7
            nn.ConvTranspose1d(opt.ngf * 8, opt.ngf * 4, 5, 3, 1, bias=False),
            nn.BatchNorm1d(opt.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 21
            nn.ConvTranspose1d(opt.ngf * 4, 1, 4, 3, 0, bias=False),
            # nn.BatchNorm1d(opt.ngf * 2),
            # nn.ReLU(True),
            )
            # state size. 1 x 64
        self.main_2nd = nn.Sequential(
            nn.ConvTranspose1d(1, opt.ngf*2, 1, 1, 0, bias=False),
            nn.BatchNorm1d(opt.ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose1d(opt.ngf * 2, opt.ngf, 3, 2, 2, bias=False),
            nn.BatchNorm1d(opt.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 125
            nn.ConvTranspose1d(opt.ngf, opt.ngf/2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ngf/2),
            nn.ReLU(True),
            # state size. (ngf/2) x 250
            nn.ConvTranspose1d(opt.ngf/2, opt.ngf/4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ngf/4),
            nn.ReLU(True),
            # state size. (ngf/4) x 500
            nn.ConvTranspose1d(opt.ngf/4, 1, 4, 2, 1, bias=False),
            # nn.ReLU(True),
            )
            # state size. (1) x 1000

    def forward(self, input):
        self.output_1st = self.main_1st(input.view([-1, self.opt.zdim, 1]))
        self.output_2nd = self.main_2nd(self.output_1st)
        self.output_1st = self.output_1st.squeeze(1)
        self.output_2nd = self.output_2nd.squeeze(1)
        return self.output_1st, self.output_2nd


# first output length of 64, then 1000
class mlp_reflc_reg_mlp_twostep_norelu(nn.Module):
    def __init__(self, opt):
        super(mlp_reflc_reg_mlp_twostep_norelu, self).__init__()
        self.opt = opt
        self.main_1st = nn.Sequential(
            nn.Linear(opt.zdim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, False),
            nn.Linear(32, 64),
            )
            # state size. 64
        self.main_2nd = nn.Sequential(
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, False),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, False),
            nn.Linear(512, 1000),
            # nn.ReLU(True),
            )
            # state size. 1000

    def forward(self, input):
        self.output_1st = self.main_1st(input)
        self.output_2nd = self.main_2nd(self.output_1st)
        return self.output_1st, self.output_2nd


class mlp_reflc_bases(nn.Module):
    def __init__(self, opt):
        super(mlp_reflc_bases, self).__init__()
        # self.main = nn.Sequential(nn.Linear(opt.zdim, 256),
        #     nn.LeakyReLU(0.2, False),
        #     nn.Linear(256, 512),
        #     nn.LeakyReLU(0.2, False),
        #     nn.Linear(512, 1000),
        #     nn.LeakyReLU(0.2, False),
        #     nn.Linear(1000, opt.nbases))

        self.main = nn.Sequential(nn.Linear(opt.zdim, 256),
            nn.LeakyReLU(0.2, False),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, False),
            nn.Linear(128, opt.nbases))

    def forward(self, input):
        self.output = self.main(input)
        return self.output


class mlp_trans_reg(nn.Module):
    def __init__(self, opt):
        super(mlp_trans_reg, self).__init__()
        self.main = nn.Sequential(nn.Linear(opt.zdim, 256),
            nn.LeakyReLU(0.2, False),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, False),
            nn.Linear(512, 1000),
            nn.LeakyReLU(0.2, False),
            nn.Linear(1000, 1000),
            nn.ReLU(True))
        # self.w = torch.cat([-torch.ones(1000), torch.ones(1000)]).cuda()
        # self.w = Variable(self.w, requires_grad=False)

    def forward(self, input):
        # self.output = -1 * self.main(input)
        # output = self.main(input).view([-1, 2, 1000])
        self.output = self.main(input)
        return self.output


class mlp_classify(nn.Module):
    def __init__(self, opt):
        super(mlp_classify, self).__init__()
        self.main = nn.Sequential(nn.Linear(opt.zdim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, False),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, False),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, False),
            nn.Linear(64, 25))

    def forward(self, input):
        self.output = self.main(input)
        return self.output