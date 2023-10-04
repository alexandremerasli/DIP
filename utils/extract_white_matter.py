import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology as morph

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

root = 'data/Algo/Data/database_v2/image50_1/image50_1.raw'

PETImage_shape = (112,112)

# Read image and MR tumors
img1_np = fijii_np(root, shape=(PETImage_shape))
MR_img = fijii_np("data/Algo/Data/database_v2/image50_1/image50_1_mr.raw", shape=(PETImage_shape))
# Threshold
img1_np = np.where(img1_np == 2,1,0)
img1_np = img1_np.astype(np.float32)
# Threshold MR to get tumor ROIs
MR_img = np.where(MR_img == 1400,1,0)
MR_img = MR_img.astype(np.float32)
# Remove tumors from thresholded PET image
img1_np = np.where(MR_img == 1,0,img1_np)


# Show before erosion
plt.imshow(img1_np,cmap="gray")

# Erosion
# declare an structuring elment
selem = morph.disk(2)
# apply a scipy morphological operation
eroded_im = morph.erosion(img1_np, selem)

# Show result
plt.figure()
plt.imshow(eroded_im,cmap="gray")
plt.show()

# save_img(img1_np,'/disk/workspace_reco/nested_admm/data/Algo/Data/database_v2/image010_3D/phantom_mask010_3D.raw')
save_img(eroded_im,"data/Algo/Data/database_v2/image50_1/background_mask50_1.raw")






# ###### resize (crop and pad BSREM)
# to_crop = fijii_np("data/Algo/Data/initialization/image010_3D/BSREM_30it/replicate_1/BSREM_it30.img", shape=(PETImage_shape))
# print(to_crop.shape)

# cropped = np.zeros(((127,172,172)))
# cropped[:,10:cropped.shape[1]-10,:] = to_crop[:,:,int((232-172)/2):to_crop.shape[2]-int((232-172)/2)]

# print(cropped.shape)
# save_img(cropped,"data/Algo/Data/initialization/image010_3D/BSREM_30it/replicate_1/BSREM_it30_172_172.img")