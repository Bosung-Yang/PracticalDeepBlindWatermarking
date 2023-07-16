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
from teacher.Basic_model import Basic
from Noiser.Noiser import Noiser
import time
import bchlib

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def bitstring_to_bytearray(data):
    is_first = True
    
    while len(data)%8!=0:
        data = np.append(data,0)
    while len(data) >=8:
        substring = ''
        for i in range(8):
            substring += str(data[i])
        
        if is_first:
            btye_array = bytearray([int(substring,2)])
            is_first= False
        
        else:
            btye_array += bytearray([int(substring,2)])
        data = data[8:]
        
    if (list(data)):
        substring = ''
        for i in range(len(data)):
            substring += str(data[i])
                    
    return btye_array

def randomCrop(img, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img

def main():

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.set_num_threads(8)
    exp_name = 'coco30bit/'
    data_path = './result/checkpoints/'
    image_path = './out/'+exp_name
    checkpoint = torch.load(data_path+'399epoch.pyt')
    noiser = Noiser(device,'teacher')
    model = Basic(device,noiser)
    utils.model_from_checkpoint(model,checkpoint)

    BCH_POLYNOMIAL = 137
    BCH_BITS = 5
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    images = glob.glob(image_path+'/encode/'+'*c*')

    df = pd.read_csv(image_path+'/message.csv')

    errors = []
    # identity
    images = glob.glob(image_path+'/encode/'+'*hidden*')
    errors = []
    start = time.time()
    
    for image in images:
        image_num = image.split('/')[5][:12]
        find = df[df.image == int(image_num)]

        message_string = str(find.message.values[0])
        message_string = message_string.zfill(120)

        image_pil = Image.open(image)
        image = image_pil.resize((256,256))
        img1 = image.crop((0,0,128,128))
        img2 = image.crop((128,0,256,128))
        img3 = image.crop((0,128,128,256))
        img4 = image.crop((128,128,256,256))

        image_tensor = TF.to_tensor(image).to(device)
        image_tensor = image_tensor * 2 -1
        image_tensor.unsqueeze_(0)

        img1_tensor = TF.to_tensor(img1).to(device)
        img1_tensor = img1_tensor * 2 -1
        img1_tensor.unsqueeze_(0)

        img2_tensor = TF.to_tensor(img2).to(device)
        img2_tensor = img2_tensor * 2 -1
        img2_tensor.unsqueeze_(0)

        img3_tensor = TF.to_tensor(img3).to(device)
        img3_tensor = img3_tensor * 2 -1
        img3_tensor.unsqueeze_(0)

        img4_tensor = TF.to_tensor(img4).to(device)
        img4_tensor = img4_tensor * 2 -1
        img4_tensor.unsqueeze_(0)

        model.ED.eval()

        decoded_messages = model.ED.decoder(img1_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])


        decoded_messages = model.ED.decoder(img2_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string += ''.join(str(int(x)) for x in decoded_rounded[0])


        decoded_messages = model.ED.decoder(img3_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string += ''.join(str(int(x)) for x in decoded_rounded[0])

        decoded_messages = model.ED.decoder(img4_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string += ''.join(str(int(x)) for x in decoded_rounded[0])
 
        decoded_string = decoded_string[:-16]
        decoded_barray = bitstring_to_bytearray(decoded_string)
        data, ecc = decoded_barray[:-bch.ecc_bytes], decoded_barray[-bch.ecc_bytes:]

        decoded_data = bch.decode(data,ecc)[1]
        decoded_mesg = bytearray_to_bits(decoded_data)

        
        xor = 0
        for idx in range(56):
            if int(message_string[idx]) != int(decoded_mesg[idx]):
                xor +=1
        errors.append(xor/56)

    print('IDENTITY : ',1-sum(errors)/len(errors))


    # cutmix
    images = glob.glob(image_path+'/encode/'+'*crop*')
    errors = []
    start = time.time()
    
    for image in images:
        image_num = image.split('/')[5][:12]
        find = df[df.image == int(image_num)]

        message_string = str(find.message.values[0])
        message_string = message_string.zfill(120)

        image_pil = Image.open(image)
        image = image_pil.resize((256,256))
        img1 = image.crop((0,0,128,128))
        img2 = image.crop((128,0,256,128))
        img3 = image.crop((0,128,128,256))
        img4 = image.crop((128,128,256,256))

        image_tensor = TF.to_tensor(image).to(device)
        image_tensor = image_tensor * 2 -1
        image_tensor.unsqueeze_(0)

        img1_tensor = TF.to_tensor(img1).to(device)
        img1_tensor = img1_tensor * 2 -1
        img1_tensor.unsqueeze_(0)

        img2_tensor = TF.to_tensor(img2).to(device)
        img2_tensor = img2_tensor * 2 -1
        img2_tensor.unsqueeze_(0)

        img3_tensor = TF.to_tensor(img3).to(device)
        img3_tensor = img3_tensor * 2 -1
        img3_tensor.unsqueeze_(0)

        img4_tensor = TF.to_tensor(img4).to(device)
        img4_tensor = img4_tensor * 2 -1
        img4_tensor.unsqueeze_(0)

        model.ED.eval()

        decoded_messages = model.ED.decoder(img1_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])


        decoded_messages = model.ED.decoder(img2_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string += ''.join(str(int(x)) for x in decoded_rounded[0])


        decoded_messages = model.ED.decoder(img3_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string += ''.join(str(int(x)) for x in decoded_rounded[0])

        decoded_messages = model.ED.decoder(img4_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string += ''.join(str(int(x)) for x in decoded_rounded[0])
 
        decoded_string = decoded_string[:-16]
        decoded_barray = bitstring_to_bytearray(decoded_string)
        data, ecc = decoded_barray[:-bch.ecc_bytes], decoded_barray[-bch.ecc_bytes:]

        decoded_data = bch.decode(data,ecc)[1]
        decoded_mesg = bytearray_to_bits(decoded_data)

        
        xor = 0
        for idx in range(56):
            if int(message_string[idx]) != int(decoded_mesg[idx]):
                xor +=1
        errors.append(xor/56)

    print('Cutmix : ',1-sum(errors)/len(errors))

def bytearray_to_bits(x):
    result = []
    for i in x:
        bits = bin(i)[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])

    return result
    
if __name__ == '__main__':
    main()
