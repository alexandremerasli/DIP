import numpy as np
import matplotlib.pyplot as plt

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



def save_img(img,name):
    fp=open(name,'wb')
    img.tofile(fp)
    print('Succesfully save in:', name)

subroot = "data/Algo/"
subsubroot = 'image010_3D/mr_axial_resampled.raw'
subsubroot = 'image010_3D/crane_t1.raw'
# subsubroot = 'image010_3D/pet.raw'
PETImage_shape = (256,256,176)
# PETImage_shape = (126,169,245)
# PETImage_shape = (344,344,127)

img1_np = fijii_np(subroot + subsubroot, shape=(PETImage_shape))

# img_padded = np.zeros((258,359,359),dtype=">u2")
img_padded = np.zeros((359,258,359),dtype=">u2")
print(img1_np.shape)
img_padded[91:359-92,1:258-1,51:359-52] = img1_np

print(img_padded.shape)

plt.imshow(img1_np[:,60,:],cmap="gray")
plt.imshow(img_padded[:,140,:],cmap="gray")
plt.show()
print("ok")

save_img(np.transpose(img1_np,axes=(1,2,0)),subroot + 'image010_3D/crane_t1_axial.raw')
save_img(np.transpose(img_padded,axes=(1,2,0)),subroot + 'image010_3D/crane_t1_axial_padded.raw')