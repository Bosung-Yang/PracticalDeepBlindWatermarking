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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
def save_images(original_images, watermarked_images, file_name, folder, resize_to=None):
    images = original_images[:original_images.shape[0], :, :, :].cpu()
    watermarked_images = watermarked_images[:watermarked_images.shape[0], :, :, :].cpu()
    
    # scale values to range [0, 1] from original range of [-1, 1]
    images = (images + 1) / 2
    watermarked_images = (watermarked_images + 1) / 2

    filename = os.path.join(folder, file_name+'_origin.png')
    torchvision.utils.save_image(images, filename)
    
    filename = os.path.join(folder, file_name+'_hidden.png')
    torchvision.utils.save_image(watermarked_images, filename)

def randomCrop(img, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img

def main():
    torch.set_num_threads(8)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    exp_name = 'coco30bit'
    data_path = './result/checkpoints/'

    checkpoint = torch.load(data_path+'399epoch.pyt')
    noiser = Noiser(device,'teacher')
    model = Basic(device,noiser)

    utils.model_from_checkpoint(model,checkpoint)

    image_path = '../data/COCO/test/0/'
    images = glob.glob(image_path+'*')
    file_num = 0
    df = pd.DataFrame()
    num_of_image = []
    encoded_message = []
    start = time.time()
    for image in images:
        #print(image)
        image_num = image.split('/')[5][:12]
        num_of_image.append(image_num)

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

        message = torch.Tensor(np.random.choice([0, 1], (img1_tensor.shape[0],
                                                        30))).to(device)

        message_detached = message.detach().cpu().numpy()
        message_string = ''.join(str(int(x)) for x in message_detached[0])
        encoded_message.append(str(message_string))

        model.ED.eval()
        encoded_images1 = model.ED.encoder(img1_tensor, message)
        encoded_images2 = model.ED.encoder(img2_tensor, message)
        encoded_images3 = model.ED.encoder(img3_tensor, message)
        encoded_images4 = model.ED.encoder(img4_tensor, message)

        encoded_images = torch.cat(
            [torch.cat([encoded_images1,encoded_images2],dim=-1),
            torch.cat([encoded_images3,encoded_images4],dim=-1),
            ],dim=2
        )
        #print(encoded_images.shape)
        #losses, (encoded_images, noised_images, decoded_messages) = hidden_net.validate_on_batch([image_tensor, message])
        save_images(image_tensor.cpu(), encoded_images.cpu(),str(image_num),'./out/'+exp_name+'/encode/')
    end = time.time()
    print('encoding time : ', end-start)
    df['image'] = num_of_image
    df['message'] = encoded_message
    df.to_csv('./out/'+exp_name+'/message.csv')



if __name__ == '__main__':
    main()
