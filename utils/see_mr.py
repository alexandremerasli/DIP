import numpy as np
import matplotlib.pyplot as plt

import argparse

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

subroot = "data/Algo/"
subsubroot = 'image010_3D/mr_axial_resampled.raw'
subsubroot = 'image010_3D/mr_interpolated.img'
# subsubroot = 'image010_3D/pet.raw'
PETImage_shape = (230,150,127)
PETImage_shape = (230,127,150)

img1_np = fijii_np(subroot + subsubroot, shape=(PETImage_shape))

plt.imshow(img1_np[:,:,100],cmap="gray")
plt.show()
print("ok")

# save_img(np.transpose(img1_np,axes=(1,2,0)),subroot + 'image010_3D/mr_interpolated_resampled.raw')
