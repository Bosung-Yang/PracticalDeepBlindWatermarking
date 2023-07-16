
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def random_float(min, max):
    """
    Return a random number
    :param min:
    :param max:
    :return:
    """
    return np.random.rand() * (max - min) + min

class Resize(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """
    def __init__(self, interpolation_method='nearest'):
        super(Resize, self).__init__()
        self.interpolation_method = interpolation_method

    def forward(self, encoded_image,epoch):
        resize_ratio_min = 0.1
        resize_ratio_max = 0.2

        if epoch > 150:
            resize_ratio_min = 0.1
            resize_ratio_max = 0.2
        if epoch > 250:
            resize_ratio_min = 0.2
            resize_ratio_max = 0.3
        resize_ratio = random_float(resize_ratio_min,resize_ratio_max)
        noised_image = encoded_image
        noised_image = F.interpolate(
                                    noised_image,
                                    scale_factor=(resize_ratio, resize_ratio),
                                    mode=self.interpolation_method)

        return noised_image