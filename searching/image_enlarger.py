from copy import deepcopy
from random import randint
import numpy as np
import cv2
import os
import sys
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dirimg', default='/home/baoge/output/output', type=str,
                    help='The input folder path for multi-images')
parser.add_argument('--output_enlarged', default='/home/baoge/output/output', type=str,
                    help='The output file path of mask.')

def enlarger(img_path,bsave=True):
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    fx = 2
    fy = 2
    enlarge = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    if bsave:
        save_name_enlarge = os.path.join(config.output_enlarged, img_path.split('/')[-1])
        cv2.imwrite(save_name_enlarge, enlarge)
    #print("xxxxx")
    return enlarge

def img2enlargedImg(dataset_dir):
    files = []
    image_list = os.listdir(dataset_dir)
    files = [os.path.join(dataset_dir, _) for _ in image_list]
    length = len(files)
    for index, jpg in enumerate(files):
        try:
            sys.stdout.write('\r>>Converting image %d/%d ' % (index, length))
            sys.stdout.flush()
            enlarger(jpg)
        except IOError as e:
            print('could not read:', jpg)
            print('error:', e)
            print('skip it\n')

    sys.stdout.write('Convert Over!\n')
    sys.stdout.flush()

def get_path(config):
    if not os.path.exists(config.input_dirimg):
        os.mkdir(config.input_dirimg)
    if not os.path.exists(config.output_enlarged):
        os.mkdir(config.output_enlarged)


if __name__ == '__main__':
    config = parser.parse_args()
    get_path(config)

    img2enlargedImg(config.input_dirimg)
