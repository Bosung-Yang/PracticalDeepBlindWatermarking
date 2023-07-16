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
    mean = 0
    var = 10
    sigma = var ** 0.5
    
    for image in images:
        image_name = image.split('/')[4][:12]
        mixed = cv.imread(image)
        gaussian = np.random.normal(mean, sigma, (mixed.shape))
        mixed = mixed + gaussian
        
        cv.imwrite(result_folder+image_name+'gn.png',mixed)

    
if __name__ == '__main__':
    main()
