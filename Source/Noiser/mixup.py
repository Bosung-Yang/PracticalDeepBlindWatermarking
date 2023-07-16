import torch.nn as nn
import torch
import numpy as np

def random_float(min, max):
    return np.random.rand() * (max - min) + min

def get_random_rectangle_inside(image, height_ratio_range, width_ratio_range):
    
    image_height = image.shape[2]
    image_width = image.shape[3]

    remaining_height = int(np.rint(random_float(height_ratio_range[0], height_ratio_range[1]) * image_height))
    remaining_width = int(np.rint(random_float(width_ratio_range[0], width_ratio_range[0]) * image_width))

    if remaining_height == image_height:
        height_start = 0
    else:
        height_start = np.random.randint(0, image_height - remaining_height)

    if remaining_width == image_width:
        width_start = 0
    else:
        width_start = np.random.randint(0, image_width - remaining_width)

    return height_start, height_start+remaining_height, width_start, width_start+remaining_width



class Mixup(nn.Module):
    def __init__(self):
        super(Mixup,self).__init__()
        self.height_ratio_range = (0.1,0.25)
        self.width_ratio_range = (0.1,0.25)

    def forward(self,image,subimage,epoch):
        encoded_image = image
        noise = subimage

        noise_mask = torch.zeros_like(image)
   
        if epoch > 150:
            self.height_ratio_range = (0.1,0.2)
            self.width_ratio_range = (0.1,0.2)
        if epoch > 250:
            self.height_ratio_range = (0.1,0.3)
            self.width_ratio_range = (0.1,0.3)


        h_start, h_end, w_start, w_end = get_random_rectangle_inside(image=encoded_image,
                                                                     height_ratio_range=self.height_ratio_range,
                                                                     width_ratio_range=self.width_ratio_range)

        noise_mask[:,:,h_start:h_end,w_start:w_end] = 0.5
    
        noised_image = encoded_image * (1 - noise_mask) + noise * (noise_mask)

        return noised_image


