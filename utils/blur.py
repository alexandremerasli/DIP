

import cv2
import numpy as np
import matplotlib.pyplot as plt


def fijii_np(path,shape,type_im='<d'):
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

shape = (112,112,1)
img = fijii_np("/home/meraslia/workspace_reco/nested_admm/data/Algo/Data/initialization/ADMMLim_it100.img",shape)

#plt.imshow(img,cmap="gray")
#plt.show()

l = [5, 9, 13]
fig, axs = plt.subplots(1,3)
for i in range(len(l)):
    img_blur = cv2.GaussianBlur(img, (l[i], l[i]), 0)
    show = axs[i].imshow(np.max(img_blur) - img_blur,cmap="gray",vmin=0,vmax=np.max(img_blur))
    plt.colorbar(show, ax=axs[i])
plt.show()

save_img(img_blur,"/home/meraslia/workspace_reco/nested_admm/data/Algo/Data/initialization/ADMMLim_blurred_it10000.img")