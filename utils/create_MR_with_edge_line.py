import matplotlib.pyplot as plt
import numpy as np


def fijii_np(path,shape,type_im=None):
    """"Transforming raw data to numpy array"""
    attempts = 0

    while attempts < 1000:
        attempts += 1
        try:
            type_im = ('<f')*(type_im=='<f') + ('<d')*(type_im=='<d')
            file_path=(path)
            dtype_np = np.dtype(type_im)
            with open(file_path, 'rb') as fid:
                data = np.fromfile(fid,dtype_np)
                if (1 in shape): # 2D
                    #shape = (shape[0],shape[1])
                    image = data.reshape(shape)
                else: # 3D
                    image = data.reshape(shape[::-1])
            attempts = 1000
            break
        except:
            # fid.close()
            type_im = ('<f')*(type_im=='<d') + ('<d')*(type_im=='<f')
            file_path=(path)
            dtype_np = np.dtype(type_im)
            with open(file_path, 'rb') as fid:
                data = np.fromfile(fid,dtype_np)
                if (1 in shape): # 2D
                    #shape = (shape[0],shape[1])
                    try:
                        image = data.reshape(shape)
                    except Exception as e:
                        pass
                else: # 3D
                    image = data.reshape(shape[::-1])
            
            fid.close()

    return image


def points_in_circle(center_y,center_x,radius,PETImage_shape,inner_circle=True): # x and y are inverted in an array compared to coordinates
    liste = [] 

    center_x += int(PETImage_shape[0]/2)
    center_y += int(PETImage_shape[1]/2)
    for x in range(0,PETImage_shape[0]):
        for y in range(0,PETImage_shape[1]):
            if (x+0.5-center_x)**2 + (y+0.5-center_y)**2 <= radius**2 and (x+0.5-center_x)**2 + (y+0.5-center_y)**2 > (radius - 2)**2:
                liste.append((x,y))

    return liste

def save_img(img,name):
    fp=open(name,'wb')
    img.tofile(fp)
    #print('Succesfully save in:', name)

PETImage_shape = (112,112)

# image = fijii_np("/home/meraslia/workspace_reco/nested_admm/data/Algo/Data/database_v2/image40_0/image40_0.raw",(112,112),type_im='<f')
image = fijii_np("/home/meraslia/workspace_reco/nested_admm/data/Algo/Data/database_v2/image40_0/image40_0_mr.raw",PETImage_shape,type_im='<f')
phantom_ROI = points_in_circle(0/4,0/4,150/4,PETImage_shape)
for couple in phantom_ROI:
    image[couple] = 130

# fig = plt.figure()
# ax = plt.gca()
# nb_crop = 10
# # im = ax.imshow(image[nb_crop:len(image) - nb_crop,nb_crop:len(image) - nb_crop], cmap='gray_r',vmin=0,vmax=500) # Showing all images with same contrast and white is zero (gray_r)
# im = ax.imshow(image[nb_crop:len(image) - nb_crop,nb_crop:len(image) - nb_crop], cmap='gray',vmin=0,vmax=60) # Showing all images with same contrast and white is zero (gray_r)

# # ax.colorbar()
# ax.axis('off')

# fig.tight_layout()
# plt.rcParams['figure.figsize'] = 10, 10

# from mpl_toolkits.axes_grid1 import make_axes_locatable
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
   
# cb = fig.colorbar(im, cax=cax)
# cb.set_label("(AU)",size="large")

# fig.savefig("/home/meraslia/workspace_reco/nested_admm/data/image_mr.png")

save_img(image,"/home/meraslia/workspace_reco/nested_admm/data/image_mr_white_line.img")