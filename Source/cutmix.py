import cv2 as cv
import os
import glob
import random
import numpy as np

def main():
    sub_image_folder = '../val/0/'
    image_folder = './out/coco30bit/encode/'
    result_folder = './out/coco30bit/encode/'
    
    images = glob.glob(image_folder+'*hidden*')
    sub_images = glob.glob(sub_image_folder+'*')
    
    for image in images:
        image_name = image.split('/')[-1][:12]
        mixed = cv.imread(image)
        sub = random.choice(sub_images)
        target = cv.imread(sub)
        while target.shape[0] < 200 or target.shape[1] < 200:
            sub = random.choice(sub_images)
            target = cv.imread(sub)

        x = np.random.randint(50-14)
        y = np.random.randint(50-14)

        mixed[x:x+13,y:y+13] = target[100:113,100:113]
        print(image,sub)
        a = cv.imwrite(result_folder+image_name+'_cutmix1.png',mixed)
        print(a)
        print(result_folder+image_name+'_cutmix1.jpg')
    
if __name__ == '__main__':
    main()
