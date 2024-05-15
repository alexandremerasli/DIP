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





phantom = "image10_1000"
phantom = "image40_1"
phantom = "image50_1"
phantom = "image50_20"

PETImage_shape = (112,112)
sinogram_shape = (344,252)
sinogram_shape_transpose = (252,344)

norm = np.ravel(fijii_np("data/Algo/Data/database_v2/" + phantom + "/simu0_1/simu0_1_nm.s",sinogram_shape_transpose))
norm = np.where(norm==0,0,1/norm)
atn = np.ravel(fijii_np("data/Algo/Data/database_v2/" + phantom + "/simu0_1/simu0_1_at.s",sinogram_shape_transpose))
atn = np.where(atn==0,0,1/atn)

if ("5" in phantom):
    A = fijii_np("data/Algo/final_syst_mat_vox_2mm.img",(sinogram_shape[0]*sinogram_shape[1],PETImage_shape[0]*PETImage_shape[1]))
    if ("50_20" in phantom):
        calibration_factor = 0.712591
    elif ("50_1" in phantom):
        calibration_factor = 0.359494
    else:
        calibration_factor = 1
elif ("40_1" in phantom):
    A = fijii_np("data/Algo/final_syst_mat_vox_4mm.img",(sinogram_shape[0]*sinogram_shape[1],PETImage_shape[0]*PETImage_shape[1]))
    calibration_factor = 130.529
elif ("10_1000" in phantom):
    A = fijii_np("data/Algo/final_syst_mat_vox_4mm.img",(sinogram_shape[0]*sinogram_shape[1],PETImage_shape[0]*PETImage_shape[1]))
    calibration_factor = 0.741026
else:
    A = fijii_np("data/Algo/final_syst_mat_vox_4mm.img",(sinogram_shape[0]*sinogram_shape[1],PETImage_shape[0]*PETImage_shape[1]))
    calibration_factor = 1

x = np.ravel(fijii_np("data/Algo/Data/database_v2/" + phantom + "/" + phantom + ".img",PETImage_shape))
# x = np.ravel(fijii_np("data/Algo/Data/database_v2/" + "image40_1" + "/" + "image40_1" + ".img",PETImage_shape))
# x = np.ravel(fijii_np("data/Algo/Data/database_v2/" + phantom + "/" + phantom + ".img",PETImage_shape),order='F')
y = fijii_np("data/Algo/Data/database_v2/" + phantom + "/simu0_1/simu0_1_pt.s",sinogram_shape_transpose,type_im=np.dtype('int16'))
r = fijii_np("data/Algo/Data/database_v2/" + phantom + "/simu0_1/simu0_1_rd.s",sinogram_shape_transpose)
s = fijii_np("data/Algo/Data/database_v2/" + phantom + "/simu0_1/simu0_1_sc.s",sinogram_shape_transpose)

Ax_castor_50_1 = fijii_np("data/Algo/image50_1_Ax_it0_good_size.s",sinogram_shape_transpose)

sinogram_shape_transpose = (252,344)
Ax_without_norm_atn = 1 / calibration_factor * np.dot(A,x)
Ax_without_norm_atn_reshaped = np.reshape(Ax_without_norm_atn,sinogram_shape_transpose)
Ax_without_norm = np.multiply(atn,Ax_without_norm_atn)
Ax = np.multiply(norm,Ax_without_norm)

Ax_reshaped = np.reshape(Ax,sinogram_shape_transpose)
Ax_copy = np.copy(Ax_reshaped)
Ax_reshaped[int(sinogram_shape_transpose[0]/2):,:] = Ax_copy[:int(sinogram_shape_transpose[0]/2),:]
Ax_reshaped[:int(sinogram_shape_transpose[0]/2),:] = Ax_copy[int(sinogram_shape_transpose[0]/2):,:]


forward_model = Ax_reshaped + r + s
# plt.figure()
# plt.title("Ax_without_norm_atn")
# plt.imshow(Ax_without_norm_atn_reshaped, cmap="gray_r")
# plt.colorbar()
# plt.figure()
# plt.title("Ax_castor_50_1")
# plt.imshow(Ax_castor_50_1, cmap="gray_r")
# plt.colorbar()
# plt.figure()
# plt.title("Ax")
# plt.imshow(Ax_reshaped,cmap="gray_r")
# plt.colorbar()
# plt.figure()
# plt.title("r")
# plt.imshow(r,cmap="gray_r")
# plt.colorbar()
# plt.figure()
# plt.title("s")
# plt.imshow(s,cmap="gray_r")
# plt.colorbar()
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

save_img(np.reshape(forward_model,sinogram_shape_transpose),"data/Algo/Data/database_v2/" + phantom + "/simu0_1/test_forward_model.s")


print("End")