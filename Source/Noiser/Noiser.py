import numpy as np
import torch.nn as nn
import torch
from Noiser.cutmix import Cutmix
from Noiser.mixup import Mixup
from Noiser.resize import Resize
class Noiser(nn.Module):

    def __init__(self,device,mode):
        super(Noiser,self).__init__()
        self.device = device
        self.cutmix = Cutmix()
        self.mixup = Mixup()
        self.resize = Resize()
        self.mode = mode

    def forward(self,images,sub_images,epoch):
        number = np.random.randint(3)
        if self.mode == 'teacher':
            # if number == 0 :
            noised_image = images
            # elif number == 1:
            #     noised_image = self.cutmix(images,sub_images,epoch)
            # elif number == 2:
            #     noised_image = self.mixup(images,sub_images,epoch)
            # else:
            #     noised_image = self.resize(images,epoch)
            return noised_image
        else:
            print('err')
            if number == 0 :
                noised_image = images
            elif number == 1:
                noised_image = self.cutmix(images,sub_images,epoch)
            elif number == 2:
                noised_image = self.mixup(images,sub_images,epoch)
            else:
                noised_image = self.resize(images,epoch)
            return noised_image
        

