

import cv2
import numpy as np
import matplotlib.pyplot as plt


def fijii_np(path,shape,type_im='<f'):
    """"Transforming raw data to numpy array"""               
    file_path=(path)
    if (1 in shape):
        nb_dimensions = 2
    else:    
        nb_dimensions = 3
    dtype_np = np.dtype(type_im)
    with open(file_path, 'rb') as fid:
        data = np.fromfile(fid,dtype_np)
        if (nb_dimensions == 2): # 2D
            image = data.reshape(shape)
        else: # 3D
            image = data.reshape(shape[::-1])
    return image


def save_img(img,name):
    fp=open(name,'wb')
    img.tofile(fp)
    print('Succesfully save in:', name)

shape = (112,112,1)
subroot = "data/Algo/"
img = fijii_np(subroot + "/Data/initialization/ADMMReg_it100.img",shape)

#plt.imshow(img,cmap="gray")
#plt.show()

l = [5, 9, 13]
fig, axs = plt.subplots(1,3)
for i in range(len(l)):
    img_blur = cv2.GaussianBlur(img, (l[i], l[i]), 0)
    show = axs[i].imshow(np.max(img_blur) - img_blur,cmap="gray",vmin=0,vmax=np.max(img_blur))
    plt.colorbar(show, ax=axs[i])
plt.show()

save_img(img_blur,subroot + "/Data/initialization/ADMMReg_blurred_it10000.img")