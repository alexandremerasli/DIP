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







PETImage_shape = (112,112)
sinogram_shape = (344,252)
sinogram_shape_transpose = (252,344)

norm = np.ravel(fijii_np("data/Algo/Data/database_v2/image10_1000/simu0_1/simu0_1_nm.s",sinogram_shape_transpose))
norm = np.where(norm==0,0,1/norm)
atn = np.ravel(fijii_np("data/Algo/Data/database_v2/image10_1000/simu0_1/simu0_1_at.s",sinogram_shape_transpose))
atn = np.where(atn==0,0,1/atn)
A = fijii_np("data/Algo/final_syst_mat.img",(sinogram_shape[0]*sinogram_shape[1],PETImage_shape[0]*PETImage_shape[1]))
x = np.ravel(fijii_np("data/Algo/Data/database_v2/image10_1000/image10_1000.img",PETImage_shape))
# x = np.ravel(fijii_np("data/Algo/Data/database_v2/image10_1000/image10_1000.img",PETImage_shape),order='F')
y = fijii_np("data/Algo/Data/database_v2/image10_1000/simu0_1/simu0_1_pt.s",sinogram_shape_transpose,type_im=np.dtype('int16'))
r = fijii_np("data/Algo/Data/database_v2/image10_1000/simu0_1/simu0_1_rd.s",sinogram_shape_transpose)
s = fijii_np("data/Algo/Data/database_v2/image10_1000/simu0_1/simu0_1_sc.s",sinogram_shape_transpose)

Ax_without_norm_atn = np.dot(A,x)
Ax_without_norm = np.multiply(atn,Ax_without_norm_atn)
Ax = np.multiply(norm,Ax_without_norm)

sinogram_shape_transpose = (252,344)
Ax_reshaped = np.reshape(Ax,sinogram_shape_transpose)
Ax_copy = np.copy(Ax_reshaped)
Ax_reshaped[int(sinogram_shape_transpose[0]/2):,:] = Ax_copy[:int(sinogram_shape_transpose[0]/2),:]
Ax_reshaped[:int(sinogram_shape_transpose[0]/2),:] = Ax_copy[int(sinogram_shape_transpose[0]/2):,:]


forward_model = Ax_reshaped+r+s
plt.figure()
plt.title("Ax+r+s")
plt.imshow(forward_model,cmap="gray_r")
plt.colorbar()
plt.figure()
plt.title("y prompt")
plt.imshow(y,cmap="gray_r")
plt.colorbar()

plt.figure()
plt.title("Ax+r+s-y")
plt.imshow(Ax_reshaped+r+s-y,cmap="gray_r")
plt.colorbar()
plt.show()

save_img(np.reshape(forward_model,sinogram_shape_transpose),"data/Algo/Data/database_v2/image10_1000/simu0_1/test_forward_model.s")


print("End")