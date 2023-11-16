import numpy as np
import matplotlib.pyplot as plt

import argparse

def fijii_np(path,shape,type_im='<f'):
    """"Transforming raw data to numpy array"""
    file_path=(path)
    try:
        fid = open(file_path, 'rb')
        dtype = np.dtype('<f')
        data = np.fromfile(fid,dtype)
        image = data.reshape(shape)
    except:
        fid = open(file_path, 'rb')
        dtype = np.dtype('<d')
        data = np.fromfile(fid,dtype)
        image = data.reshape(shape)
    return image

parser = argparse.ArgumentParser(description='Display absolute difference between 2 images')
parser.add_argument('--img1', type=str, dest='img1', help='first image .img')
parser.add_argument('--img2', type=str, dest='img2', help='second image .img')

args = parser.parse_args()

root = 'data/Algo/'
PETImage_shape = (112,112,1)
same_scale_TMI = True # Save diff image with fixed scale to compare several difference images for TMI ReLU artifacts experiment

image = fijii_np("/home/meraslia/workspace_reco/nested_admm/data/Algo/Data/database_v2/image3_3/image3_3.raw",PETImage_shape)
img1_np = fijii_np(args.img1, shape=(PETImage_shape))
print("min 1 = " ,np.min(img1_np))
print("mean 1 = " ,np.mean(img1_np))
print("max 1 = " ,np.max(img1_np))
img2_np = fijii_np(args.img2, shape=(PETImage_shape))
print("")
print("min 2 = " ,np.min(img2_np))
print("mean 2 = " ,np.mean(img2_np))
print("max 2 = " ,np.max(img2_np))

plt.figure()
#plt.imshow(np.abs(img1_np - img2_np), cmap='gray_r')
if (same_scale_TMI):
    maxi=max(abs(np.min(img1_np - img2_np)),np.max(img1_np - img2_np))
    maxi=0.2
    mini=-maxi
    plt.imshow(img1_np - img2_np, cmap='bwr',vmin=mini,vmax=maxi)
else:
    plt.imshow(img1_np - img2_np, cmap='bwr')
plt.title('absolute difference between img1 and img2')
plt.ylim
plt.colorbar()
plt.savefig(root+'diff_img.png')

plt.figure()
plt.imshow(img1_np / img2_np, cmap='bwr')

'''
for i in range(img1_np.shape[0]):
    for j in range(img1_np.shape[1]):
        if (np.abs(img1_np[i,j] / img2_np[i,j] > 100)):
            print("img1_np = ",img1_np[i,j])
            print("img2_np = ",img2_np[i,j])
'''
plt.title('relative difference between img1 and img2')
plt.colorbar()
plt.savefig(root+'relative_diff_img.png')

MSE_normed = np.linalg.norm(img1_np - img2_np) / (PETImage_shape[0]*PETImage_shape[1]*PETImage_shape[2])
print("MSE : ",MSE_normed)
print("Numerical error below threshold : ",MSE_normed < 1e-5)

plt.figure()
#plt.imshow(np.abs(img1_np), cmap='gray_r')
plt.imshow(img1_np, cmap='gray_r',vmin=np.min(img1_np),vmax=np.max(img1_np))
plt.title('img1')
plt.colorbar()
plt.savefig(root+'img1.png')

plt.figure()
#plt.imshow(np.abs(img2_np), cmap='gray_r')
plt.imshow(img2_np, cmap='gray_r',vmin=np.min(img1_np),vmax=np.max(img1_np))
plt.title('img2')
plt.colorbar()
plt.savefig(root+'img2.png')