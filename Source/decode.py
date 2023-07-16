import torch
import torch.nn
import numpy as np
import pandas as pd
import os
import glob
import utils
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision
from model.Basic_model import Basic
from Noiser.Noiser import Noiser
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
        print('cuda')
    else:
        device = torch.device('cpu')
    torch.set_num_threads(8)
    exp_name = 'coco30bit/'
    data_path = './result/checkpoints/'
    image_path = './out/'+exp_name
    checkpoint = torch.load('399epoch.pyt')   
    noiser = Noiser(device,'teacher')
    model = Basic(device,noiser)
    utils.model_from_checkpoint(model,checkpoint)
    
    images = glob.glob(image_path+'/encode/'+'*c*')
    
    df = pd.read_csv(image_path+'/message.csv')
    
    errors = []
    # identity
    images = glob.glob(image_path+'/encode/'+'*hidden*')
    errors = []
    error_per_bit = [0 for i in range(30)]
    total_xor = 0
    for image in images:
        image_num = image.split('\\')[-1][:12]
        find = df[df.image == int(image_num)]

        message_string = str(find.message.values[0]) 
        message_string = message_string.zfill(30)

        image_pil = Image.open(image)
        image = image_pil.resize((128,128))
        image = image_pil
        
        image_tensor = TF.to_tensor(image).to(device)
        image_tensor = image_tensor * 2 -1
        image_tensor.unsqueeze_(0)

        model.ED.eval()
        decoded_messages = model.ED.decoder(image_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])

        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
                total_xor += 1
                error_per_bit[idx] +=1
        errors.append(xor/len(message_string))
    print('IDENTITY : ',1-sum(errors)/len(errors))
    print(error_per_bit)
    print(total_xor)

    # cutmix
    error_per_bit = [0 for i in range(30)]
    images = glob.glob(image_path+'/encode/'+'*cutmix*')
    errors = []
    for image in images:
        image_num = image.split('/')[5][:12]

        find = df[df.image == int(image_num)]

        message_string = str(find.message.values[0]) 
        message_string = message_string.zfill(30)

        image_pil = Image.open(image)
        image = image_pil.resize((128,128))
        image = image_pil
        
        image_tensor = TF.to_tensor(image).to(device)
        image_tensor = image_tensor * 2 -1
        image_tensor.unsqueeze_(0)

        model.ED.eval()
        decoded_messages = model.ED.decoder(image_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])

        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
                error_per_bit[idx] +=1
        errors.append(xor/len(message_string))
    print('CUTMIX : ',1-sum(errors)/len(errors))
    print(error_per_bit)
    
    # Mixup
    images = glob.glob(image_path+'/encode/'+'*mixup*')
    errors = []
    error_per_bit = [0 for i in range(30)]
    for image in images:
        image_num = image.split('/')[5][:12]

        find = df[df.image == int(image_num)]

        message_string = str(find.message.values[0]) 
        message_string = message_string.zfill(20)

        image_pil = Image.open(image)
        image = image_pil.resize((128,128))
        image = image_pil
        
        image_tensor = TF.to_tensor(image).to(device)
        image_tensor = image_tensor * 2 -1
        image_tensor.unsqueeze_(0)

        model.ED.eval()
        decoded_messages = model.ED.decoder(image_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])

        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
                error_per_bit[idx] +=1
        errors.append(xor/len(message_string))
    print('MIXUP : ',sum(errors)/len(errors))
    print(error_per_bit)

    # resize
    images = glob.glob(image_path+'/encode/'+'*resize*')
    errors = []
    error_per_bit = [0 for i in range(30)]
    for image in images:
        image_num = image.split('/')[5][:12]

        find = df[df.image == int(image_num)]

        message_string = str(find.message.values[0]) 
        message_string = message_string.zfill(20)

        image_pil = Image.open(image)
        image = image_pil.resize((128,128))
        image = image_pil
        
        image_tensor = TF.to_tensor(image).to(device)
        image_tensor = image_tensor * 2 -1
        image_tensor.unsqueeze_(0)

        model.ED.eval()
        decoded_messages = model.ED.decoder(image_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])

        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
                error_per_bit[idx] +=1
        errors.append(xor/len(message_string))
    print('RESIZE : ',1-sum(errors)/len(errors))
    print(error_per_bit)

    #JPEG
    images = glob.glob(image_path+'/encode/'+'*jpeg*')
    errors = []
    error_per_bit = np.array([0 for i in range(30)])
    for image in images:
        image_num = image.split('/')[5][:12]

        find = df[df.image == int(image_num)]

        message_string = str(find.message.values[0]) 
        message_string = message_string.zfill(20)

        image_pil = Image.open(image)
        image = image_pil.resize((128,128))
        image = image_pil
        
        image_tensor = TF.to_tensor(image).to(device)
        image_tensor = image_tensor * 2 -1
        image_tensor.unsqueeze_(0)

        model.ED.eval()
        decoded_messages = model.ED.decoder(image_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])

        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
                error_per_bit[idx] +=1
        errors.append(xor/len(message_string))
    print('JPEG : ',1-sum(errors)/len(errors))
    print(1-error_per_bit/1000)

    #JPEG
    images = glob.glob(image_path+'/encode/'+'*crop*')
    errors = []
    error_per_bit = [0 for i in range(30)]
    for image in images:
        image_num = image.split('/')[5][:12]

        find = df[df.image == int(image_num)]

        message_string = str(find.message.values[0]) 
        message_string = message_string.zfill(20)

        image_pil = Image.open(image)
        image = image_pil.resize((128,128))
        image = image_pil
        
        image_tensor = TF.to_tensor(image).to(device)
        image_tensor = image_tensor * 2 -1
        image_tensor.unsqueeze_(0)

        model.ED.eval()
        decoded_messages = model.ED.decoder(image_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])

        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
                error_per_bit[idx] +=1
        errors.append(xor/len(message_string))
    print('Crop : ',1-sum(errors)/len(errors))
    print(error_per_bit)

    #JPEG
    images = glob.glob(image_path+'/encode/'+'*blur*')
    errors = []
    error_per_bit = [0 for i in range(30)]
    for image in images:
        image_num = image.split('/')[5][:12]

        find = df[df.image == int(image_num)]

        message_string = str(find.message.values[0]) 
        message_string = message_string.zfill(20)

        image_pil = Image.open(image)
        image = image_pil.resize((128,128))
        image = image_pil
        
        image_tensor = TF.to_tensor(image).to(device)
        image_tensor = image_tensor * 2 -1
        image_tensor.unsqueeze_(0)

        model.ED.eval()
        decoded_messages = model.ED.decoder(image_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])

        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
                error_per_bit[idx] +=1
        errors.append(xor/len(message_string))
    print('blur : ',1-sum(errors)/len(errors))  
    print(error_per_bit)  
    
    images = glob.glob(image_path+'/encode/'+'*gn*')
    errors = []
    error_per_bit = [0 for i in range(30)]
    for image in images:
        image_num = image.split('/')[5][:12]

        find = df[df.image == int(image_num)]

        message_string = str(find.message.values[0]) 
        message_string = message_string.zfill(20)

        image_pil = Image.open(image)
        image = image_pil.resize((128,128))
        image = image_pil
        
        image_tensor = TF.to_tensor(image).to(device)
        image_tensor = image_tensor * 2 -1
        image_tensor.unsqueeze_(0)

        model.ED.eval()
        decoded_messages = model.ED.decoder(image_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])

        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
                error_per_bit[idx] +=1
        errors.append(xor/len(message_string))
    print('Gassian Nosise : ',1-sum(errors)/len(errors))  
    print(error_per_bit)    


    images = glob.glob(image_path+'/encode/'+'*sp*')
    errors = []
    error_per_bit = [0 for i in range(30)]
    for image in images:
        image_num = image.split('/')[5][:12]

        find = df[df.image == int(image_num)]

        message_string = str(find.message.values[0]) 
        message_string = message_string.zfill(20)

        image_pil = Image.open(image)
        image = image_pil.resize((128,128))
        image = image_pil
        
        image_tensor = TF.to_tensor(image).to(device)
        image_tensor = image_tensor * 2 -1
        image_tensor.unsqueeze_(0)

        model.ED.eval()
        decoded_messages = model.ED.decoder(image_tensor)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        decoded_string = ''.join(str(int(x)) for x in decoded_rounded[0])

        xor = 0
        for idx in range(len(message_string)):
            if message_string[idx] != decoded_string[idx]:
                xor+=1
                error_per_bit[idx] +=1
        errors.append(xor/len(message_string))
    print('Salt & Pepper : ',1-sum(errors)/len(errors))  
    print(error_per_bit)  
if __name__ == '__main__':
    main()
