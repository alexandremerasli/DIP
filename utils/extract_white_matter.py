import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk, erosion

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

# phantom = "image50_1"
phantom = "image50_2"
subroot = "data/Algo/"
subsubroot = '/Data/database_v2/' + phantom + '/' + phantom + '.raw'

PETImage_shape = (112,112)

# Read image and MR tumors
img1_np = fijii_np(subroot + subsubroot, shape=(PETImage_shape))
MR_img = fijii_np(subroot + "Data/database_v2/" + str(phantom) + "/" + str(phantom) + "_mr.raw", shape=(PETImage_shape))
# Threshold
img1_np = np.where(img1_np == 2,1,0)
img1_np = img1_np.astype(np.float32)

if (phantom == "image50_1"):
    # Threshold MR to get tumor ROIs
    MR_img = np.where(MR_img == 1400,1,0)
    MR_img = MR_img.astype(np.float32)
    # Remove tumors from thresholded PET image
    img1_np = np.where(MR_img == 1,0,img1_np)


# Show before erosion
plt.imshow(img1_np,cmap="gray")

# Erosion
# declare an structuring elment
selem = disk(2)
# apply a scipy morphological operation
eroded_im = erosion(img1_np, selem)

# Show result
plt.figure()
plt.imshow(eroded_im,cmap="gray")
plt.show()

# save_img(img1_np,subroot + "Data/database_v2/image010_3D/phantom_mask010_3D.raw")
save_img(eroded_im,subroot + "Data/database_v2/" + phantom + "/background_mask" + phantom[5:] + ".raw")
save_img(eroded_im,subroot + "Data/database_v2/" + phantom + "/white_matter_" + phantom[5:] + ".raw")






# ###### resize (crop and pad BSREM)
# to_crop = fijii_np(subroot + "Data/initialization/image010_3D/BSREM_30it/replicate_1/BSREM_it30.img", shape=(PETImage_shape))
# print(to_crop.shape)

# cropped = np.zeros(((127,172,172)))
# cropped[:,10:cropped.shape[1]-10,:] = to_crop[:,:,int((232-172)/2):to_crop.shape[2]-int((232-172)/2)]

# print(cropped.shape)
# save_img(cropped,subroot + "Data/initialization/image010_3D/BSREM_30it/replicate_1/BSREM_it30_172_172.img")