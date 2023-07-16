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
        encode_param=[int(cv.IMWRITE_JPEG_QUALITY),90]
        
        cv.imwrite(result_folder+image_name+'_jpeg90.jpg',img,encode_param)
        print(result_folder+image_name+'_jpeg90.jpg')
    
if __name__ == '__main__':
    main()
