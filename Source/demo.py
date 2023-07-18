import gradio

import torch
import torch.nn
import numpy as np
import pandas as pd
import os
import glob
import utils
from model import *
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision
from model.Basic_model import Basic
from Noiser.Noiser import Noiser

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

torch.set_num_threads(8)
checkpoint = torch.load('399epoch.pyt')
noiser = Noiser(device,'teacher')
model = Basic(device,noiser)
utils.model_from_checkpoint(model,checkpoint)

def gr_encode(text,image):
    message = torch.Tensor(np.random.choice([0, 1], (1,
                                                        30))).to(device)
    #image_tensor = TF.to_tensor(image).to(device)
    #image_tensor = image_tensor * 2 -1
    #image_tensor.unsqueeze_(0)

    image = np.transpose(image,[2,0,1])

    image = torch.Tensor(image).to(device)
    image = image.unsqueeze_(0)

    encoded_image = model.ED.encoder(image,message)
    message_detached = message.detach().cpu().numpy()
    message_string = ''.join(str(int(x)) for x in message_detached[0])
    encoded_image = encoded_image.squeeze(0).detach().cpu().numpy()
    encoded_image = np.transpose(encoded_image,[1,2,0])
    print(encoded_image.shape)
    print(encoded_image)
    encoded_image = encoded_image - 127.5
    encoded_image = encoded_image/255
    print(encoded_image)
    return encoded_image, message_string

def main():


    demo = gradio.Interface(gr_encode, 
    inputs = ['text',gradio.Image(shape=(128,128))],
    outputs = ['image','text']
    )

    demo.launch()

if __name__ == '__main__':
    main()