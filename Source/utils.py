import torch
import torchvision
import pandas as pd
import numpy as np 
import random
import string
import torchvision.utils
import os
import numpy
import glob
import cv2
#import lpips
import math
from PIL import Image, ImageOps
from torchvision import transforms, datasets
#loss_fn_alex = lpips.LPIPS(net='alex')
#loss_fn_vgg = lpips.LPIPS(net='vgg')


def random_string(length:int):
    return ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(length))

def psnr(img1,img2):
    mse = numpy.mean( (img1-img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX  = 255.0
    return 20* math.log10(PIXEL_MAX / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
    
def save_checkpoint(model, epoch:int, folder:str):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    file_name = str(epoch)+'epoch.pyt'
    file_name = os.path.join(folder,file_name)
    checkpoint = {
        'ED_state_dic' : model.ED.state_dict(),
        'DIS_state_dic' : model.DIS.state_dict(),
        'ED_optim' : model.optim_ED.state_dict(),
        'DIS_optim' : model.optim_DIS.state_dict(),
        'epoch':epoch
    }
    torch.save(checkpoint,file_name)

def model_from_checkpoint(model, checkpoint):
    """ Restores the hidden_net object from a checkpoint object """
    model.ED.load_state_dict(checkpoint['ED_state_dic'])
    model.optim_ED.load_state_dict(checkpoint['ED_optim'])
    model.DIS.load_state_dict(checkpoint['DIS_state_dic'])
    model.optim_DIS.load_state_dict(checkpoint['DIS_optim'])

def save_student(model,epoch:int,folder:str):
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_name = str(epoch)+'epoch_s.pyt'
    file_name = os.path.join(folder,file_name)

    checkpoint = {
            'Decoder' : model.decoder.state_dict(),
            'Dec_optim' : model.optim_ED.state_dict(),
            'epoch' : epoch
            }
    torch.save(checkpoint,file_name)

def student_from_checkpoint(model,checkpoint):
    model.decoder.load_state_dict(checkpoint['Decoder'])
    model.optim_ED.load_state_dict(checkpoint['Dec_optim'])


