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
    var = 1
    sigma = var
    
    for image in images:
        image_name = image.split('/')[4][:12]
        mixed = cv.imread(image)
        row,col,ch = mixed.shape
        s_vs_p = 0.5
        amount = 0.05
        out = np.copy(mixed)
        # Salt mode
        num_salt = np.ceil(amount * mixed.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in mixed.shape]
        out[coords] = 1
        num_pepper = np.ceil(amount* mixed.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in mixed.shape]
        out[coords] = 0
        cv.imwrite(result_folder+image_name+'sp.png',out)

    
if __name__ == '__main__':
    main()
