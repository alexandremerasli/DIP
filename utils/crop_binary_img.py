import os
import matplotlib.pyplot as plt
import numpy as np
from re import split

def fijii_np(path,shape,type_im='<f'):
    """"Transforming raw data to numpy array"""
    file_path=(path)
    dtype_np = np.dtype(type_im)
    with open(file_path, 'rb') as fid:
        data = np.fromfile(fid,dtype_np)
        # image = data.reshape(shape)
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


# Define image to reshape and old shape and new shape
PETImage_shape = (284,284,184)
img_path = "/home/MEDECINE/mera1140/sherbrooke_workspace/wakusuteshon/24_06_07_iecFirstTestAlgoOnUHR/Datasets/MLEM_UHR_sens_scaled/MLEM_UHR_sens_scaled_it20.img"
img_np = fijii_np(img_path,PETImage_shape)
PETImage_shape_new = (192,192,184)

# Scale image with good dimensions
reduced_img_np = img_np[:,int((PETImage_shape[1]-PETImage_shape_new[1])/2):PETImage_shape[1]-int((PETImage_shape[1]-PETImage_shape_new[1])/2),int((PETImage_shape[1]-PETImage_shape_new[1])/2):PETImage_shape[1]-int((PETImage_shape[1]-PETImage_shape_new[1])/2)]

# Save image
save_img(reduced_img_np, img_path[:-4] + "_" + str(PETImage_shape_new[0]) + ".img")
print("End")
