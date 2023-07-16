import cv2 as cv
import os
import glob
import random
import numpy as np


def main():
    sub_image_folder = '../data/COCO/val/0/'
    image_folder = './out/coco30bit/encode/'
    result_folder = './out/coco30bit/encode/'
    random.seed()
    images = glob.glob(image_folder+'*hidden*')
    sub_images = glob.glob(sub_image_folder+'*')
    
    for image in images:
        image_name = image.split('/')[4][:12]
        mixed = cv.imread(image)
        sub = random.choice(sub_images)
        target = cv.imread(sub)
        while target.shape[0]<=200 or target.shape[1]<=200:
            sub = random.choice(sub_images)
            target = cv.imread(sub)
        x = np.random.randint(128-14*5)
        y = np.random.randint(128-14*5)
        mixed[x:x+13*5,y:y+13*5] = mixed[x:x+13*5,y:y+13*5]//2 + target[100:100+13*5,100:100+13*5]//2
        print(image,sub)
        cv.imwrite(result_folder+image_name+'_mixup.png',mixed)
        print(result_folder+image_name+'_mixup.jpg')
    
if __name__ == '__main__':
    main()

