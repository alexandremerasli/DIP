import numpy as np
import matplotlib.pyplot as plt

import argparse

def fijii_np(path,shape,type='<f'):
    """"Transforming raw data to numpy array"""
    file_path=(path)
    dtype = np.dtype(type)
    fid = open(file_path, 'rb')
    data = np.fromfile(fid,dtype)
    image = data.reshape(shape)
    return image

parser = argparse.ArgumentParser(description='Display absolute difference between 2 images')
parser.add_argument('--img1', type=str, dest='img1', help='first image .img')
parser.add_argument('--img2', type=str, dest='img2', help='second image .img')

args = parser.parse_args()

root = 'data/Algo/'
PETImage_shape = (112,112)

img1_np = fijii_np(args.img1, shape=(PETImage_shape))
img2_np = fijii_np(args.img2, shape=(PETImage_shape))

plt.figure()
plt.imshow(np.abs(img1_np - img2_np), cmap='gray_r')
plt.imshow(img1_np - img2_np, cmap='gray_r')
plt.title('absolute difference between img1 and img2')
plt.colorbar()
plt.savefig(root+'diff_img.png')

MSE_normed = np.linalg.norm(img1_np - img2_np) / (PETImage_shape[0]*PETImage_shape[1])
print("MSE : ",MSE_normed)
print("Numerical error below threshold : ",MSE_normed < 1e-5)

plt.figure()
plt.imshow(np.abs(img1_np), cmap='gray_r')
plt.imshow(img1_np, cmap='gray_r')
plt.title('img1')
plt.colorbar()
plt.savefig(root+'img1.png')

plt.figure()
plt.imshow(np.abs(img2_np), cmap='gray_r')
plt.imshow(img2_np, cmap='gray_r')
plt.title('img2')
plt.colorbar()
plt.savefig(root+'img2.png')