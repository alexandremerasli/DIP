

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def fijii_np(path,shape,type_im='<f'):
    file_path=(path)
    dtype = np.dtype(type_im)
    fid = open(file_path, 'rb')
    data = np.fromfile(fid,dtype)
    if (1 in shape): # 2D
        image = data.reshape(shape)
    else: # 3D
        image = data.reshape(shape[::-1])

    return image


def save_img(img,name):
    fp=open(name,'wb')
    img.tofile(fp)
    print('Succesfully save in:', name)

# Load MR image
shape = (112,112)
img = fijii_np("data/Algo/Data/database_v2/image40_1/image40_1_mr.raw",shape)
img = fijii_np("data/Algo/Data/database_v2/image50_0/image50_0_mr.raw",shape)
conv_img = np.copy(img)

np.random.seed(1)

# Choose task
task = "convolve_cascade"
task = "convolve_iteratively"
# task = "convolve_big_kernel"

if (task == "convolve_cascade"):
    # Convolve MR with several Gaussian kernels in cascade (a new one at each new iteration of the loop)
    for i in range(10):
        # Convolve i+1 kernels 
        random_kernel = (np.random.normal(0,1,9).reshape((3,3)))
        if (i>0):
            kernel = signal.convolve2d(np.copy(kernel),np.copy(random_kernel), mode='full')
        else:
            kernel = np.copy(random_kernel)
        # Convolve with MR image
        conv_img = signal.convolve2d(np.copy(img), np.copy(kernel), mode='valid')
        conv_img = 0.2*conv_img*(conv_img<0)+conv_img*(conv_img>=0)
        print(np.min(conv_img))
        plt.imshow(conv_img,cmap='gray')
        # plt.imshow(kernel,cmap='gray')
        # save_img(kernel.astype(np.float32),"data/Algo/Data/kernels_" + str(i) + ".img")
        # save_img(kernel.astype(np.float32),"data/Algo/Data/conv_img_" + str(i) + ".img")
        
        # Histogram of values in the kernel
        # histo = np.histogram(kernel,bins=(2*i+3)**2)
        # bins = 0.5*(histo[1][:-1] + histo[1][1:])
        # plt.bar(bins,histo[0]*len(bins))

        # Histogram of gaussian sample
        # histo_gaussian = np.histogram(np.random.normal(0,1,10000),bins=(2*i+3)**2)
        # bins = 0.5*(histo_gaussian[1][:-1] + histo_gaussian[1][1:])
        # plt.bar(bins,histo_gaussian[0]*len(bins))
        plt.show()

if (task == "convolve_iteratively"):
    # Convolve iteratively MR with new kernel (should be the same as above)
    for i in range(10):
        # kernel = np.random.normal(0,np.sqrt(1/16),9).reshape((3,3))
        kernel = np.random.uniform(-np.sqrt(1/(3*3*1)),np.sqrt(1/(3*3*1)),9).reshape((3,3))
        # conv_img = signal.convolve2d(np.copy(conv_img), np.copy(kernel), mode='valid', boundary='fill', fillvalue=0)
        conv_img = signal.convolve2d(np.copy(conv_img), np.copy(kernel), mode='same', boundary='fill', fillvalue=0)
        plt.imshow(conv_img,cmap='gray',vmin=np.min(conv_img),vmax=np.max(conv_img))
        plt.show()
        save_img(conv_img.astype(np.float32),"data/Algo/Data/conv_" + str(i) + ".img")

if (task == "convolve_big_kernel"):
    # Convolve MR with big gaussian kernel
    kernel_size = 3
    kernel = (np.random.normal(0,1,kernel_size*kernel_size).reshape((kernel_size,kernel_size)))
    conv_img = signal.convolve2d(np.copy(img), np.copy(kernel), mode='same')
    plt.imshow(conv_img,cmap='gray')
    plt.show()
    print("end")
