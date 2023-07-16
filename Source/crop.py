import cv2 as cv
import os
import glob
import random
import numpy as np

def crop():
    sub_image_folder = '../data/COCO/val/0/'
    image_folder = './out/coco30bit/encode/'
    result_folder = './out/coco30bit/encode/'
    
    images = glob.glob(image_folder+'*hidden*')
    
    for image in images:
        image_name = image.split('\\')[-1][:12]
        img = cv.imread(image)
        # interpolation = INTER_NEAREST INTER_LINEAR INTER_LINEAR_EXACT INTER_CUBIC INTER_AREA INTER_LANCZOS4
        f = 13
        x = np.random.randint(f)
        y = np.random.randint(f)
        #img = img[x:128+f-x,y:128+f-y]
        img[0:x] = 0
        img[128+f-x:] = 0
        img[0:y] = 0
        img[128+f-y:] = 0        
        cv.imwrite(result_folder+image_name+'crop.png',img)
    
if __name__ == '__main__':
    crop()
