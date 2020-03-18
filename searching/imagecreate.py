# import itertools
# import matplotlib
# import matplotlib.pyplot as plt
from copy import deepcopy
from random import randint
import numpy as np
import cv2
import os
import sys
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img', default='/home/baoge/output/output/places2_01.png', type=str,
                    help='The input img for single image ')

parser.add_argument('--input_dirimg', default='/home/baoge/imagemask/input', type=str,
                    help='The input folder path for multi-images')
parser.add_argument('--output_dirmask', default='/home/baoge/imagemask/mask', type=str,
                    help='The output file path of mask.')
parser.add_argument('--output_dirmasked', default='/home/baoge/imagemask/imgmasked', type=str,
                    help='The output file path of masked.')
parser.add_argument('--MAX_MASK_NUMS', default='100', type=int,
                    help='max numbers of masks')

parser.add_argument('--MAX_DELTA_HEIGHT', default='32', type=int,
                    help='max height of delta')
parser.add_argument('--MAX_DELTA_WIDTH', default='32', type=int,
                    help='max width of delta')

parser.add_argument('--HEIGHT', default='128', type=int,
                    help='max height of delta')
parser.add_argument('--WIDTH', default='128', type=int,
                    help='max width of delta')

parser.add_argument('--IMG_SHAPES', type=eval, default=(256,256, 3))


# 随机生成不规则掩膜
def random_mask(img_path,height, width,config,channels=3,bsave=True):
    """Generates a random irregular mask with lines, circles and elipses"""
    img = np.zeros((height, width, channels), np.uint8)
    img_data = cv2.imread(img_path)

    # Set size scale
    size = int((width + height) * 0.02)
    if width < 64 or height < 64:
        raise Exception("Width and Height of mask must be at least 64!")

    # Draw random shortlines
    for _ in range(randint(5, 10)):
    #    x1, x2 = randint(1, width), randint(1, width)
    #    y1, y2 = randint(1, height), randint(1, height)
        x1 = randint(1, width)
        y1 = randint(1, height)
        for _ in range(randint(5, 10)):
            while(True):
                x2 = randint(x1-40, x1+40)
                if x2<=0 or x2>=255:
                    continue
                else:
                    break
            while(True):
                y2 = randint(y1-40, y1+40)
                if y2<=0 or y2>=255:
                    continue
                else:
                    break
            thickness = 3
            x = randint(x1-15,x1+15)
            y = randint(y1-15,y1+15)
            cv2.line(img, (x, y), (x2, y2), (255,255,255), thickness)

    # Draw random longlines
    for _ in range(randint(5, 10)):
        #    x1, x2 = randint(1, width), randint(1, width)
        #    y1, y2 = randint(1, height), randint(1, height)
        x1 = randint(1, width)
        y1 = randint(1, height)
        for _ in range(randint(1, 5)):
            while (True):
                x2 = randint(x1 - 10, x1 + 10)
                if x2 <= 0 or x2 >= 255:
                    continue
                else:
                    break
            while (True):
                y2 = randint(y1 - 100, y1 + 100)
                if y2 <= 0 or y2 >= 255:
                    continue
                else:
                    break
            thickness = randint(2,3)
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), thickness)

    # Draw random circles
    for _ in range(randint(30, 50)):
        x1, y1 = randint(1, width), randint(1, height)
        radius = randint(1, 5)
        cv2.circle(img, (x1, y1), radius, (255,255,255), -1)

    # Draw random ellipses
    #for _ in range(randint(10, 20)):
    #    x1, y1 = randint(1, width), randint(1, height)
    #    s1, s2 = randint(1, width), randint(1, height)
    #    a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
        #thickness = randint(1, 3)
    #    thickness = 5
    #    cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, (255,255,255), thickness)
    mask=img
    img_shape = config.IMG_SHAPES
    img_height = img_shape[0]
    img_width = img_shape[1]

    image = cv2.resize(img_data, (img_width, img_height))
    masked_img = deepcopy(image)
    masked_img[mask == 255] = 255

    #cv2.imwrite("1.png",img)
    #if bsave:
    #    save_name_mask = "mask1"
    #    cv2.imwrite(save_name_mask, img)
    if bsave:
        save_name_mask = os.path.join(config.output_dirmask, img_path.split('/')[-1])
        cv2.imwrite(save_name_mask, mask)

        save_name_masked = os.path.join(config.output_dirmasked, img_path.split('/')[-1])
        cv2.imwrite(save_name_masked, masked_img)

    return masked_img, mask


def get_path(config):
    if not os.path.exists(config.input_dirimg):
        os.mkdir(config.input_dirimg)
    if not os.path.exists(config.output_dirmask):
        os.mkdir(config.output_dirmask)
    if not os.path.exists(config.output_dirmasked):
        os.mkdir(config.output_dirmasked)

def img2maskedImg(dataset_dir):
    files = []
    image_list = os.listdir(dataset_dir)
    files = [os.path.join(dataset_dir, _) for _ in image_list]
    length = len(files)
    for index, jpg in enumerate(files):
        try:
            sys.stdout.write('\r>>Converting image %d/%d ' % (index, length))
            sys.stdout.flush()
            random_mask(jpg,256, 256,config,channels=3,bsave=True)
        except IOError as e:
            print('could not read:', jpg)
            print('error:', e)
            print('skip it\n')

    sys.stdout.write('Convert Over!\n')
    sys.stdout.flush()
# python3 generate_mask.py --img ./examples/celeba/000042.jpg --HEIGHT 64 --WIDTH 64

if __name__ == '__main__':
    config = parser.parse_args()
    get_path(config)
    #random_mask(1024, 1024, config)
    img2maskedImg(config.input_dirimg)

