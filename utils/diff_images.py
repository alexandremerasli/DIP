import numpy as np
import matplotlib.pyplot as plt

import argparse
from utils_func import *

parser = argparse.ArgumentParser(description='Display absolute difference between 2 images')
parser.add_argument('--img1', type=str, dest='img1', help='first image .img')
parser.add_argument('--img2', type=str, dest='img2', help='second image .img')

args = parser.parse_args()

root = 'data/Algo/'
PETImage_shape = (128,128)

img1_np = fijii_np(args.img1, shape=(PETImage_shape))
img2_np = fijii_np(args.img2, shape=(PETImage_shape))

plt.imshow(np.abs(img1_np - img2_np), cmap='gray_r')
plt.title('absolute difference between img1 and img2')
plt.colorbar()
plt.savefig(root+'diff_img.png')