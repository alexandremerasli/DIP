import numpy as np
import matplotlib.pyplot as plt

def save_img(img,name):
    fp=open(name,'wb')
    img.tofile(fp)
    print('Succesfully save in:', name)

def points_in_circle_edge(self,center_y,center_x,radius,PETImage_shape,inner_circle=True): # x and y are inverted in an array compared to coordinates
    liste = [] 

    center_x += int(PETImage_shape[0]/2)
    center_y += int(PETImage_shape[1]/2)
    for x in range(0,PETImage_shape[0]):
        for y in range(0,PETImage_shape[1]):
            if (x+0.5-center_x)**2 + (y+0.5-center_y)**2 <= radius**2 and (x+0.5-center_x)**2 + (y+0.5-center_y)**2 > (radius - 2)**2:
                liste.append((x,y))

    return liste

PETImage_shape = (32,32)
# PETImage_shape = (112,112)

PET_phantom = np.zeros(PETImage_shape,dtype='<f')
MR_phantom = np.zeros(PETImage_shape,dtype='<f')
cold_hot_ROI_phantom = np.zeros(PETImage_shape,dtype='<f')
inside_ROI_phantom = np.zeros(PETImage_shape,dtype='<f')

# rectangle phantom with 3 regions
case = 10

# PET_phantom = np.array([[100,100,100,100,100],[100,400,400,400,100],[100,400,10,400,100],[100,400,400,400,100],[100,100,100,100,100]])
# MR_phantom[1:-1,1:-1] = 0.096*np.ones(PETImage_shape)[1:-1,1:-1]
# inside_ROI_phantom = np.ones(PETImage_shape)


# plt.imshow(PET_phantom,cmap='gray_r')
# plt.colorbar()
# plt.show()
# plt.imshow(MR_phantom,cmap='gray')
# plt.colorbar()
# plt.show()
# print("end")


# Define ROIs

inside_ROI_phantom = np.ones(PETImage_shape)
center_x = int(PETImage_shape[0]/2)
center_y = int(PETImage_shape[1]/2)
radius = 5-1
for x in range(0,PETImage_shape[0]):
    for y in range(0,PETImage_shape[1]):
        if (x+0.5-center_x)**2 + (y+0.5-center_y)**2 <= radius**2 and (x+0.5-center_x)**2 + (y+0.5-center_y)**2 > (radius - 2)**2:
            cold_hot_ROI_phantom[x,y] = 1

# Save phantoms
from pathlib import Path
Path("data/Algo/Data/database_v2/image3_" + str(case)).mkdir(parents=True, exist_ok=True)
# save_img(PET_phantom,"data/Algo/Data/database_v2/image3_" + str(case) + "/image3_" + str(case) + ".raw")
# save_img(PET_phantom,"data/Algo/Data/database_v2/image3_" + str(case) + "/image3_" + str(case) + ".img")
# save_img(MR_phantom,"data/Algo/Data/database_v2/image3_" + str(case) + "/image3_" + str(case) + "_atn.raw")
# save_img(MR_phantom,"data/Algo/Data/database_v2/image3_" + str(case) + "/image3_" + str(case) + "_atn.img")


save_img(inside_ROI_phantom,"data/Algo/Data/database_v2/image3_" + str(case) + "/background_mask3_" + str(case) + ".raw")
save_img(inside_ROI_phantom,"data/Algo/Data/database_v2/image3_" + str(case) + "/phantom_mask3_" + str(case) + ".raw")
save_img(cold_hot_ROI_phantom,"data/Algo/Data/database_v2/image3_" + str(case) + "/cold_mask3_" + str(case) + ".raw")
# save_img(cold_hot_ROI_phantom,"data/Algo/Data/database_v2/image3_" + str(case) + "/tumor_perfect_match_ROI_mask3_" + str(case) + ".raw")
# save_img(cold_hot_ROI_phantom,"data/Algo/Data/database_v2/image3_" + str(case) + "/tumor_TEP_mask3_" + str(case) + ".raw")
# save_img(cold_hot_ROI_phantom,"data/Algo/Data/database_v2/image3_" + str(case) + "/tumor_TEP_match_square_ROI_mask3_" + str(case) + ".raw")
save_img(cold_hot_ROI_phantom,"data/Algo/Data/database_v2/image3_" + str(case) + "/tumor_mask3_" + str(case) + ".raw")

# cold edge and cold inside not well defined
save_img(cold_hot_ROI_phantom,"data/Algo/Data/database_v2/image3_" + str(case) + "/cold_inside_mask3_" + str(case) + ".raw")
save_img(cold_hot_ROI_phantom,"data/Algo/Data/database_v2/image3_" + str(case) + "/cold_edge_mask3_" + str(case) + ".raw")
