import numpy as np
import matplotlib.pyplot as plt

def save_img(img,name):
    fp=open(name,'wb')
    img.tofile(fp)
    print('Succesfully save in:', name)

PETImage_shape = (112,112)

PET_phantom = np.zeros(PETImage_shape)
MR_phantom = np.zeros(PETImage_shape)
cold_hot_ROI_phantom = np.zeros(PETImage_shape)
inside_ROI_phantom = np.zeros(PETImage_shape)

# phantom with 3 regions
# for i in range(PETImage_shape[0]):
#     for j in range(PETImage_shape[1]):
#         if i > j:
#             phantom[i,j] = 0.5
#         elif i <= j:
#             if i + j < PETImage_shape[0]:
#                 phantom[i,j] = 0.1
#             else:
#                 phantom[i,j] = 0.2

# cylindrical phantom with 2 regions, 4 cases

insert = "cold"
# insert = "hot"

MRI = "keep_colors"
MRI = "reverse_colors"

if (insert == "cold" and MRI == "keep_colors"):
    case = 1
elif (insert == "cold" and MRI == "reverse_colors"):
    case = 2
elif (insert == "hot" and MRI == "keep_colors"):
    case = 3
elif (insert == "hot" and MRI == "reverse_colors"):
    case = 4

center_x_big = int(PETImage_shape[0]/2)
center_y_big = int(PETImage_shape[1]/2)
radius_big = int(PETImage_shape[1]*3/8)

center_x_small = int(PETImage_shape[0]*2/5)
center_y_small = int(PETImage_shape[1]*2/5)
radius_small = int(PETImage_shape[1]*1/10)

for x in range(0,PETImage_shape[0]):
    for y in range(0,PETImage_shape[1]):
        if (x+0.5-center_x_big)**2 + (y+0.5-center_y_big)**2 <= radius_big**2:
            PET_phantom[x,y] = 0.25
            MR_phantom[x,y] = 0.25
            inside_ROI_phantom[x,y] = 1
        if (x+0.5-center_x_small)**2 + (y+0.5-center_y_small)**2 <= radius_small**2:
            PET_phantom[x,y] = 1 * (insert=="hot") + 0.025 * (insert=="cold")
            MR_phantom[x,y] = PET_phantom[x,y] * (MRI == "keep_colors") + (1 * (insert=="cold") + 0.025 * (insert=="hot")) * (MRI == "reverse_colors")
            cold_hot_ROI_phantom[x,y] = 1


plt.imshow(PET_phantom,cmap='gray_r')
# plt.colorbar()
# plt.show()
plt.imshow(MR_phantom,cmap='gray')
plt.colorbar()
plt.show()
print("end")


# Save phantoms
from pathlib import Path
Path("data/Algo/Data/database_v2/image3_" + str(case)).mkdir(parents=True, exist_ok=True)
save_img(PET_phantom,"data/Algo/Data/database_v2/image3_" + str(case) + "/image3_" + str(case) + ".raw")
save_img(PET_phantom,"data/Algo/Data/database_v2/image3_" + str(case) + "/image3_" + str(case) + ".img")
save_img(MR_phantom,"data/Algo/Data/database_v2/image3_" + str(case) + "/image3_" + str(case) + "_atn.raw")
save_img(MR_phantom,"data/Algo/Data/database_v2/image3_" + str(case) + "/image3_" + str(case) + "_atn.img")


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
