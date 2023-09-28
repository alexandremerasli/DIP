import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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


def points_in_circle(center_y,center_x,radius,PETImage_shape,inner_circle=True): # x and y are inverted in an array compared to coordinates
    liste = [] 

    center_x += int(PETImage_shape[0]/2)
    center_y += int(PETImage_shape[1]/2)
    for x in range(0,PETImage_shape[0]):
        for y in range(0,PETImage_shape[1]):
            if (x+0.5-center_x)**2 + (y+0.5-center_y)**2 <= radius**2:
                liste.append((x,y))

    return liste

# Read PET and MR brain images (without tumor)
path_PET = 'data/Algo/Data/database_v2/image50_0/image50_0.raw'
path_MR = 'data/Algo/Data/database_v2/image50_0/image50_0_mr.raw'
PETImage_shape = (112,112,1)
img_PET = fijii_np(path_PET, shape=(PETImage_shape))
img_MR = fijii_np(path_MR, shape=(PETImage_shape))

# Show MR
plt.imshow(img_MR,cmap="gray")
# plt.show()

# Add tumors
tumor_1a_ROI = points_in_circle(15,-25,4,PETImage_shape)

# tumor_1b_ROI = points_in_circle(0,25,4,PETImage_shape)
xx,yy = np.meshgrid(np.arange(66,72),np.arange(52,60))
tumor_1b_ROI = list(map(tuple, np.dstack([xx.ravel(), yy.ravel()])[0]))

tumor_2_MR_ROI = points_in_circle(-25,0,8,PETImage_shape)
tumor_2_PET_ROI = points_in_circle(-27,0,4,PETImage_shape)
tumor_3a_ROI = points_in_circle(13,25,4,PETImage_shape)


ROI_MR_list = [tumor_1a_ROI,tumor_2_MR_ROI,tumor_3a_ROI]
ROI_PET_list = [tumor_1a_ROI,tumor_2_PET_ROI]
tumor_MR_values_list = [1400,1400,1400]
tumor_PET_values_list = [10,10,10]
# mask_list = [cold_mask, tumor_mask, phantom_mask, bkg_mask]
for i in range(len(ROI_MR_list)):
    ROI_MR = ROI_MR_list[i]
    # mask = mask_list[i]
    for couple in ROI_MR:
        #mask[int(couple[0] - PETImage_shape[0]/2)][int(couple[1] - PETImage_shape[1]/2)] = 1
        print(couple)
        img_MR[couple] = tumor_MR_values_list[i]

for i in range(len(ROI_PET_list)):
    ROI_PET = ROI_PET_list[i]
    
    for couple in ROI_PET:
        #mask[int(couple[0] - PETImage_shape[0]/2)][int(couple[1] - PETImage_shape[1]/2)] = 1
        print(couple)
        img_PET[couple] = tumor_PET_values_list[i]

# Show MR
plt.figure()
plt.imshow(img_MR,cmap="gray")
plt.figure()
plt.imshow(img_PET,cmap="gray_r")
plt.show()
Path("data/Algo/Data/database_v2/image50_1").mkdir(parents=True, exist_ok=True)
save_img(img_PET,'data/Algo/Data/database_v2/image50_1/image50_1.raw')
save_img(img_MR,'data/Algo/Data/database_v2/image50_1/image50_1_mr.raw')