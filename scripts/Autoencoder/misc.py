import torch
import torch.nn as nn
import torchvision.utils as vutils

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# sampe iamges
def visualizeAsImages(img_list, output_dir,
                      n_sample=4, id_sample=None, dim=-1,
                      filename='myimage', nrow=2,
                      normalize=False):
    if id_sample is None:
        images = img_list[0:n_sample,:,:,:]
    else:
        images = img_list[id_sample,:,:,:]
    if dim >= 0:
        images = images[:,dim,:,:].unsqueeze(1)
    vutils.save_image(images,
        '%s/%s'% (output_dir, filename+'.png'),
        nrow=nrow, normalize = normalize, padding=2)

