import numpy as np
import pandas as pd
from pathlib import Path
from os.path import isfile
import matplotlib.pyplot as plt

def fijii_np(path,shape,type_im='<f'):
    """"Transforming raw data to numpy array"""
    file_path=(path)
    dtype = np.dtype(type_im)
    fid = open(file_path, 'rb')
    data = np.fromfile(fid,dtype)
    image = data.reshape(shape)
    return image

def save_img(img,name):
    fp=open(name,'wb')
    img.tofile(fp)
    print('Succesfully save in:', name)

folder_path = "/home/meraslia/workspace_reco/nested_admm/data/Algo/image00_cylinder/"
subfolder = "ADMMLim_TOF_70_0.0002"
subfolder = "ADMMLim_sans_TOF_70_0.0002"
#subfolder = "ADMMLim_TOF_70"

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
