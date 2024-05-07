import numpy as np
import matplotlib.pyplot as plt

import argparse

def fijii_np(path,shape,type_im='>u2'):
    """"Transforming raw data to numpy array"""
    file_path=(path)
    # type_im='>u2'
    type_im='<f4'
    dtype = np.dtype(type_im)
    fid = open(file_path, 'rb')
    data = np.fromfile(fid,dtype)
    if (1 in shape): # 2D
        #shape = (shape[0],shape[1])
        image = data.reshape(shape)
    else: # 3D
        image = data.reshape(shape[::-1])
    return image



def save_img(img,name):
    fp=open(name,'wb')
    img.tofile(fp)
    print('Succesfully save in:', name)

root = '/disk/workspace_reco/nested_admm/data/Algo/image010_3D/mr_axial_resampled.raw'
root = '/disk/workspace_reco/nested_admm/data/Algo/image010_3D/mr_interpolated.img'
# root = '/disk/workspace_reco/nested_admm/data/Algo/image010_3D/pet.raw'
PETImage_shape = (230,150,127)
PETImage_shape = (230,127,150)

img1_np = fijii_np(root, shape=(PETImage_shape))

plt.imshow(img1_np[:,:,100],cmap="gray")
plt.show()
print("ok")

# save_img(np.transpose(img1_np,axes=(1,2,0)),'/disk/workspace_reco/nested_admm/data/Algo/image010_3D/mr_interpolated_resampled.raw')
