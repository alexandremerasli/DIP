import numpy as np
import matplotlib.pyplot as plt

import argparse

def fijii_np(path,shape,type_im='<f'):
    """"Transforming raw data to numpy array"""
    file_path=(path)
    try:
        dtype = np.dtype(type_im)
        fid = open(file_path, 'rb')
        data = np.fromfile(fid,dtype)
        if (1 in shape): # 2D
            #shape = (shape[0],shape[1])
            image = data.reshape(shape)
        else: # 3D
            image = data.reshape(shape[::-1])
    except:
        dtype = np.dtype('<d')
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

root = '/disk/workspace_reco/nested_admm/data/Algo/image010_3D/replicate_1/nested_TMI/'
PETImage_shape = (172,172,127)


image_name_BSREM = "BSREM_it30"
image_name_BSREM_Bowsher = "BSREM_Bowsher_it30"
image_name_MLEM = "MLEM_it60"
image_name_ES = "out_DIP-1_FINAL"
image_name_DNA = "out_DIP99_FINAL"
image_name_DIPRecon = "DIPRecon_out_DIP25_FINAL"
image_name_DIPRecon = "ES_DIPRecon3D"
image_name_noisy_BSREM = "BSREM_it30_noisy"
image_name_MR = "image010_3D_atn"
num_slice = 63

image_name_list = [image_name_DNA]
image_name_list = [image_name_BSREM, image_name_BSREM_Bowsher, image_name_MLEM, image_name_ES, image_name_DNA, image_name_DIPRecon, image_name_noisy_BSREM, image_name_MR]
image_name_list = [image_name_DIPRecon]

for image_name in image_name_list:
    print(image_name)
    img1_np = fijii_np(root + image_name + ".img", shape=(PETImage_shape))

    plt.figure()
    if (image_name != image_name_MR):
        plt.imshow(img1_np[num_slice-1,:,:],cmap="gray_r",vmin=0,vmax=17000)
    else:
        plt.imshow(img1_np[num_slice-1,:,:],cmap="gray",vmin=np.min(img1_np),vmax=np.max(img1_np)/2)
    plt.colorbar()
    plt.axis("off")
    # plt.show()

    plt.savefig(root + image_name + "_slice_" + str(num_slice) + ".png")