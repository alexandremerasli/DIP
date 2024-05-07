import numpy as np
import matplotlib.pyplot as plt

import argparse

def fijii_np(path,shape,type_im='>u2'):
    """"Transforming raw data to numpy array"""
    file_path=(path)
    type_im='>u2'
    # type_im='>f4'
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
root = '/disk/workspace_reco/nested_admm/data/Algo/image010_3D/crane_t1.raw'
# root = '/disk/workspace_reco/nested_admm/data/Algo/image010_3D/pet.raw'
PETImage_shape = (256,256,176)
# PETImage_shape = (126,169,245)
# PETImage_shape = (344,344,127)

img1_np = fijii_np(root, shape=(PETImage_shape))

# img_padded = np.zeros((258,359,359),dtype=">u2")
img_padded = np.zeros((359,258,359),dtype=">u2")
print(img1_np.shape)
img_padded[91:359-92,1:258-1,51:359-52] = img1_np

print(img_padded.shape)

plt.imshow(img1_np[:,60,:],cmap="gray")
plt.imshow(img_padded[:,140,:],cmap="gray")
plt.show()
print("ok")

save_img(np.transpose(img1_np,axes=(1,2,0)),'/disk/workspace_reco/nested_admm/data/Algo/image010_3D/crane_t1_axial.raw')
save_img(np.transpose(img_padded,axes=(1,2,0)),'/disk/workspace_reco/nested_admm/data/Algo/image010_3D/crane_t1_axial_padded.raw')