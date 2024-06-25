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
        image = data.reshape(shape)
                    
    return image

def save_img(img,name):
    fp=open(name,'wb')
    img.tofile(fp)
    print('Succesfully save in:', name)


# Define image to rescale and shape
PETImage_shape = (192,192,184)
img_path = "/home/MEDECINE/mera1140/sherbrooke_workspace/wakusuteshon/24_06_07_iecFirstTestAlgoOnUHR/Datasets/sensitivity_generation_thirdTest/sensitivity_generation_thirdTest_sensitivity.img"
img_np = fijii_np(img_path,PETImage_shape)

# Apply scaling factor
true_sensitivity_factor = 100 / 1.1
scaling_factor = np.max(img_np) * true_sensitivity_factor
img_np = img_np / scaling_factor

# Save image
save_img(img_np, img_path[:-4] + "_scaled.img")
print("End")
