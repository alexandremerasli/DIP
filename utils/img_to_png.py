import os
import matplotlib.pyplot as plt
import numpy as np

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


# Convert .img images to .png
my_PETImage_shape = (112,112)
folder_list = "/home/meraslia/Documents/Thèse/Résultats à montrer/2024_soutenance"
for image in os.listdir(folder_list):
    if image.split(".")[-1] == "img":
        img = fijii_np(folder_list + "/" + image,my_PETImage_shape)
        plt.imshow(img,cmap='gray_r',vmin=0,vmax=50)
        plt.axis('off')
        plt.colorbar()
        plt.savefig(folder_list + "/" + image[:-4] + ".png")
        plt.close()