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
    
    for image in images:
        image_name = image.split('\\')[-1][:12]
        img = cv.imread(image)

        # interpolation = INTER_NEAREST INTER_LINEAR INTER_LINEAR_EXACT INTER_CUBIC INTER_AREA INTER_LANCZOS4
        img = cv.resize(img,dsize=(0,0), fx=0.90,fy=0.90, interpolation = cv.INTER_AREA)

        
        cv.imwrite(result_folder+image_name+'resize.png',img)
        print(result_folder+image_name+'resize.jpg')
    
if __name__ == '__main__':
    main()
