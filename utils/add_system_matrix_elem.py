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

def atoi(text):
    return int(text) if text.isdigit() else text
    
def natural_keys(text):
    return [ atoi(c) for c in split(r'(\d+)', text) ] # APGMAP final curves + resume computation

PETImage_shape = (112,112)
sinogram_shape = (344,252)
sinogram_norm_path = "data/Algo/Data/database_v2/image10_1000/simu0_1/simu0_1_nm.s"
sinogram_norm_np = fijii_np(sinogram_norm_path,sinogram_shape)

root_syst_mat = "data/Algo/mat_syst_folder"

syst_mat_disorder_np = np.zeros((sinogram_shape[0]*sinogram_shape[1],PETImage_shape[0]*PETImage_shape[1]),dtype=np.float32)

# Loop over all files in the directory
root_filename = "matrice_systeme_line_"
img_list = os.listdir(root_syst_mat)
img_list.sort(key=natural_keys)
sinogram_bin_without_zeros = -1
true_sinogram_bin = -1
# for true_sinogram_bin in range(sinogram_shape[0]*sinogram_shape[1]):
for filename in reversed(img_list):
    sinogram_bin_without_zeros += 1
    true_sinogram_bin += 1
    while (np.squeeze(np.ravel(sinogram_norm_np))[true_sinogram_bin] == 0):
        true_sinogram_bin += 1
    # Check if the file is a .img file
    filename = root_filename + str(sinogram_bin_without_zeros) + ".img"
    if filename.endswith(".img"):
        # Check if the numpy array is not equal to zero
        if (np.squeeze(np.ravel(sinogram_norm_np))[true_sinogram_bin] != 0):
            print(true_sinogram_bin)
            syst_mat_disorder_np[true_sinogram_bin,:] = np.ravel(fijii_np(os.path.join(root_syst_mat, filename),PETImage_shape))


save_img(syst_mat_disorder_np, "data/Algo/final_syst_mat.img")
print("End")
