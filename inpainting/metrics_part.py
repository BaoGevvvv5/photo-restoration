
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from glob import glob
from ntpath import basename
from imageio import imread
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from skimage.color import rgb2gray

def parse_args():
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--data-path', help='Path to ground truth data', type=str)
    parser.add_argument('--mask-path', help='Path to ground mask', type=str)
    parser.add_argument('--output-path', help='Path to output data', type=str)
    parser.add_argument('--debug', default=0, help='Debug', type=int)
    args = parser.parse_args()
    return args


def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)


args = parse_args()
for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg))
path_true = args.data_path
path_pred = args.output_path
path_mask = args.mask_path
psnr = []
ssim = []
mae = []
names = []
index = 1



files = list(glob(path_true + '/*.jpg')) + list(glob(path_true + '/*.png'))



for fn in sorted(files):
    name = basename(str(fn))
    names.append(name)

    img_gt = (imread(str(fn)) / 255.0).astype(np.float32)
    img_pred = (imread(path_pred + '/' + basename(str(fn))) / 255.0).astype(np.float32)
    #img_pred = imread(path_pred + '/' + basename(str(fn)))
    img_mask = imread(path_mask + '/' + basename(str(fn)))
    #print(path_mask)
    mask_size=img_mask.shape
    mask_height=mask_size[0]
    mask_width=mask_size[1]
    #print(mask_height)
    x=[]
    y=[]
    for i in range(mask_height):
        for j in range(mask_width):
            if all(img_mask[i,j] == (255,255,255)):
                x.append(i)
                y.append(j)
    #print(x)
    x=sorted(x)
    x0=x[0]
    x1=x[-1]
    y0=y[0]
    y1=y[-1]
    #print(x0)
    img_gt = img_gt[x0:x1,y0:y1]
    img_pred = img_pred[x0:x1,y0:y1]
    x=[]
    y=[]
    img_gt = rgb2gray(img_gt)
    img_pred = rgb2gray(img_pred)
    #print(img_gt.shape)
    #print(img_gt.shape)
    psnr.append(compare_psnr(img_gt, img_pred, data_range=1))
    ssim.append(compare_ssim(img_gt, img_pred, data_range=1, win_size=15,multichannel = True))
    mae.append(compare_mae(img_gt, img_pred))
    if np.mod(index, 100) == 0:
        print(
            str(index) + ' images processed',
            "PSNR: %.4f" % round(np.mean(psnr), 4),
            "SSIM: %.4f" % round(np.mean(ssim), 4),
            "MAE: %.4f" % round(np.mean(mae), 4),
        )
    index += 1

np.savez(args.output_path + '/metrics.npz', psnr=psnr, ssim=ssim, mae=mae, names=names)
print(
    "PSNR: %.4f" % round(np.mean(psnr), 4),
    "PSNR Variance: %.4f" % round(np.var(psnr), 4),
    "SSIM: %.4f" % round(np.mean(ssim), 4),
    "SSIM Variance: %.4f" % round(np.var(ssim), 4),
    "MAE: %.4f" % round(np.mean(mae), 4),
    "MAE Variance: %.4f" % round(np.var(mae), 4)
)



