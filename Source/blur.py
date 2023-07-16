import cv2 as cv
import os
import glob
import random
import numpy as np

def main():
    sub_image_folder = '../data/COCO/val/0/'
    image_folder = './out/coco30bit/encode/'
    result_folder = './out/coco30bit/encode/'
    
    images = glob.glob(image_folder+'*hidden*')
    sub_images = glob.glob(sub_image_folder+'*')
    
    for image in images:
        image_name = image.split('\\')[-1][:12]
        mixed = cv.imread(image)
        blur = cv.GaussianBlur(mixed, (1,1),0)

        
        cv.imwrite(result_folder+image_name+'_blur.png',blur)

    
if __name__ == '__main__':
    main()
