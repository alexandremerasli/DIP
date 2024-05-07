import os
import matplotlib.pyplot as plt
import numpy as np

def fijii_np(path,shape,type_im='<f'):
    """"Transforming raw data to numpy array"""
    file_path=(path)
    dtype = np.dtype(type_im)
    fid = open(file_path, 'rb')
    data = np.fromfile(fid,dtype)
    image = data.reshape(shape)
    return image



# Convert .img images to .png
my_PETImage_shape = (112,112)
folder_list = "/home/meraslia/Documents/Thèse/Résultats à montrer/2024_01_16/manuscrit hyperparam/lr/images for manuscript"
folder_list = "/home/meraslia/Documents/Thèse/Résultats à montrer/2024_01_16/manuscrit hyperparam/artifacts/black lines : brain 2D 200 DIP 10 internes rho 3/"
folder_list = "/home/meraslia/Documents/Thèse/Résultats à montrer/2024_01_16/manuscrit hyperparam/artifacts/"
folder_list = "/home/meraslia/Documents/Thèse/Résultats à montrer/2024_01_16/manuscrit hyperparam/it ext/4_1 MR 3"
folder_list = "/home/meraslia/Documents/Thèse/Résultats à montrer/2024_01_16/manuscrit hyperparam/DNA-APPGML/90Y"
folder_list = "/home/meraslia/Documents/Thèse/Résultats à montrer/2024_01_16/manuscrit hyperparam/nb it DIP/EMV inside DNA"
folder_list = "/home/meraslia/Documents/Thèse/Résultats à montrer/2024_01_16/manuscrit hyperparam/EMV"
folder_list = "/home/meraslia/Documents/Thèse/Résultats à montrer/2024_01_16/TMI_paper"
folder_list = "/home/meraslia/Documents/Thèse/Résultats à montrer/2024_01_16/fantomes"
folder_list = "/home/meraslia/Documents/Thèse/Résultats à montrer/2024_soutenance"
for image in os.listdir(folder_list):
    if image.split(".")[-1] == "img":
        img = fijii_np(folder_list + "/" + image,my_PETImage_shape)
        # plt.imshow(img,cmap='gray_r',vmin=0,vmax=500)
        # plt.imshow(img,cmap='gray_r',vmin=0,vmax=12.5)
        plt.imshow(img,cmap='gray_r',vmin=0.094,vmax=0.098)
        plt.imshow(img,cmap='gray_r',vmin=0,vmax=50)
        plt.axis('off')
        plt.colorbar()
        plt.savefig(folder_list + "/" + image[:-4] + ".png")
        plt.close()
        print(image)