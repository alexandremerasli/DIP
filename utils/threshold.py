import numpy as np
import matplotlib.pyplot as plt

def fijii_np(path,shape,type_im='>u2'):
    """"Transforming raw data to numpy array"""
    file_path=(path)
    # type_im='>u2'
    type_im='<f4'
    dtype = np.dtype(type_im)
    fid = open(file_path, 'rb')
    data = np.fromfile(fid,dtype)
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

# root = '/disk/workspace_reco/nested_admm/data/Algo/Data/database_v2/image010_3D/BSREM_it30.img'
root = 'data/Algo/Data/database_v2/image010_3D/image010_3D.img'
root = 'data/Algo/Data/database_v2/image50_0/image50_0_mr.raw'

PETImage_shape = (232,152,127)
# PETImage_shape = (230,150,127)
# PETImage_shape = (172,172,127)
PETImage_shape = (112,112,1)

img1_np = fijii_np(root, shape=(PETImage_shape))
# Threshold
# img1_np = img1_np > 2000
img1_np = img1_np > 200
img1_np = img1_np.astype(np.float32)
# Reshape
# img1_np = img1_np.reshape(PETImage_shape[::-1])

# plt.imshow(img1_np[50,:,:],cmap="gray")
plt.imshow(img1_np,cmap="gray")
plt.show()
print("ok")

# save_img(img1_np,'/disk/workspace_reco/nested_admm/data/Algo/Data/database_v2/image010_3D/phantom_mask010_3D.raw')
save_img(img1_np,'data/Algo/Data/database_v2/image50_0/phantom_mask50_0.raw')






# ###### resize (crop and pad BSREM)
# to_crop = fijii_np("data/Algo/Data/initialization/image010_3D/BSREM_30it/replicate_1/BSREM_it30.img", shape=(PETImage_shape))
# print(to_crop.shape)

# cropped = np.zeros(((127,172,172)))
# cropped[:,10:cropped.shape[1]-10,:] = to_crop[:,:,int((232-172)/2):to_crop.shape[2]-int((232-172)/2)]

# print(cropped.shape)
# save_img(cropped,"data/Algo/Data/initialization/image010_3D/BSREM_30it/replicate_1/BSREM_it30_172_172.img")