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
        message_string = message_string.zfill(20)

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
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))

        decoded_messages = model.ED.decoder(img2_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))

        decoded_messages = model.ED.decoder(img3_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))

        decoded_messages = model.ED.decoder(img4_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))
    end = time.time()
    print(end-start)
    print('IDENTITY : ',1-sum(errors)/len(errors))

    # cutmix
    images = glob.glob(image_path+'/encode/'+'*cutmix*')
    errors = []
    i1 = []
    i2 = []
    i3 = []
    i4 = []
    for image in images:
        image_num = image.split('/')[5][:12]
        find = df[df.image == int(image_num)]

        message_string = str(find.message.values[0])
        message_string = message_string.zfill(30)

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
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))
        i1.append(xor/len(message_string))

        decoded_messages = model.ED.decoder(img2_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))
        i2.append(xor/len(message_string))

        decoded_messages = model.ED.decoder(img3_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))
        i3.append(xor/len(message_string))

        decoded_messages = model.ED.decoder(img4_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))
        i4.append(xor/len(message_string))
    print('Cutmix : ',1-sum(errors)/len(errors))

    # Mixup
    images = glob.glob(image_path+'/encode/'+'*mixup*')
    errors = []
    for image in images:
        image_num = image.split('/')[5][:12]
        find = df[df.image == int(image_num)]

        message_string = str(find.message.values[0])
        message_string = message_string.zfill(20)

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
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))

        decoded_messages = model.ED.decoder(img2_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))

        decoded_messages = model.ED.decoder(img3_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))

        decoded_messages = model.ED.decoder(img4_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))
    print('Mixup : ',1-sum(errors)/len(errors))
    # resize
    images = glob.glob(image_path+'/encode/'+'*resize*')
    errors = []
    for image in images:
        image_num = image.split('/')[5][:12]
        find = df[df.image == int(image_num)]

        message_string = str(find.message.values[0])
        message_string = message_string.zfill(20)

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
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))

        decoded_messages = model.ED.decoder(img2_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))

        decoded_messages = model.ED.decoder(img3_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))

        decoded_messages = model.ED.decoder(img4_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))
    print('Resize : ',1-sum(errors)/len(errors))

    #JPEG
    images = glob.glob(image_path+'/encode/'+'*jpeg*')
    errors = []
    for image in images:
        image_num = image.split('/')[5][:12]
        find = df[df.image == int(image_num)]

        message_string = str(find.message.values[0])
        message_string = message_string.zfill(20)

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
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))

        decoded_messages = model.ED.decoder(img2_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))

        decoded_messages = model.ED.decoder(img3_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))

        decoded_messages = model.ED.decoder(img4_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))
    print('JPEG : ',1-sum(errors)/len(errors))
    #JPEG
    images = glob.glob(image_path+'/encode/'+'*crop*')
    errors = []
    for image in images:
        image_num = image.split('/')[5][:12]
        find = df[df.image == int(image_num)]

        message_string = str(find.message.values[0])
        message_string = message_string.zfill(20)

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
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))

        decoded_messages = model.ED.decoder(img2_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))

        decoded_messages = model.ED.decoder(img3_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))

        decoded_messages = model.ED.decoder(img4_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))
    print('Crop : ',1-sum(errors)/len(errors))

    #JPEG
    images = glob.glob(image_path+'/encode/'+'*blur*')
    errors = []
    for image in images:
        image_num = image.split('/')[5][:12]
        find = df[df.image == int(image_num)]

        message_string = str(find.message.values[0])
        message_string = message_string.zfill(20)

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
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))

        decoded_messages = model.ED.decoder(img2_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))

        decoded_messages = model.ED.decoder(img3_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))

        decoded_messages = model.ED.decoder(img4_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])
        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
        errors.append(xor/len(message_string))
    print('blur : ',1-sum(errors)/len(errors))
if __name__ == '__main__':
    main()
