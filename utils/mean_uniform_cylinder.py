import numpy as np
import pandas as pd
from pathlib import Path
from os.path import isfile
import matplotlib.pyplot as plt

def fijii_np(path,shape,type_im='<f'):
    """"Transforming raw data to numpy array"""               
    file_path=(path)
    nb_dimensions = len(shape)
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

folder_path = "data/Algo/image00_cylinder/"
subfolder = "ADMMReg_TOF_70_0.0002"
subfolder = "ADMMReg_sans_TOF_70_0.0002"
#subfolder = "ADMMReg_TOF_70"

subsubfolder = subfolder

finalOuterIter = 100
alpha_list = np.zeros((finalOuterIter,1))
relativePrimalResidual_list = np.zeros((finalOuterIter,1))
relativeDualResidual_list = np.zeros((finalOuterIter,1))

#'''
for outer_it in range(1,finalOuterIter+1):
    filename = folder_path + subfolder + "/" + subsubfolder + '_it' + format(outer_it)
    im = fijii_np(filename + ".img",(57,112,112))


fig, ax1 = plt.subplots()
plt.plot(np.arange(1,finalOuterIter+1),np.log10(relativePrimalResidual_list))
plt.plot(np.arange(1,finalOuterIter+1),np.log10(relativeDualResidual_list))
plt.legend(["relativePrimalResidual","relativeDualResidual"])
plt.title("relative residuals for " + subfolder)
plt.xlabel("it")
plt.ylabel("relativePrimalResidual (log scale)")
ax1.set_ylim(-2,3)
plt.savefig("relativePrimalResidual for " + subfolder + ".png")
#'''
