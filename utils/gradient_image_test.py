import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

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

def sobel_x():
    kernel = np.zeros((3,3))
    kernel[0,0]=1
    kernel[1,0]=2
    kernel[2,0]=1
    kernel[0,1]=kernel[1,1]=kernel[2,1]=0
    kernel[0,2]=-1
    kernel[1,2]=-2
    kernel[2,2]=-1
    return kernel 
    
def sobel_y():
    kernel = np.zeros((3,3))
    kernel[0,0]=1
    kernel[0,1]=2
    kernel[0,2]=1
    kernel[1,0]=kernel[1,1]=kernel[1,2]=0
    kernel[2,0]=-1
    kernel[2,1]=-2
    kernel[2,2]=-1
    return kernel 

def sobel_y_sep():
    kernel = np.zeros((1,3))
    kernel[0]=1
    kernel[1]=2
    kernel[2]=1


def laplacian():
    kernel = np.ones((3,3))
    kernel[1,1]=-8
    return kernel 

it = 99
root_MR = '/home/meraslia/workspace_reco/nested_admm/data/Algo/image40_1/replicate_1/nested/Block2/config_recoI=APGMAP_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=10_tau_D=2_lr=0.01_sub_i=100_opti_=Adam_skip_=3_scali=positive_normalization_input=CT_nb_ou=10_mlem_=False_A_AML=-100/out_cnn/24/out_DIP' + str(it) + '_FINAL.img'
root_intermediate = '/home/meraslia/workspace_reco/nested_admm/data/Algo/image40_1/replicate_1/nested/Block2/config_recoI=APGMAP_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=10_tau_D=2_lr=0.01_sub_i=100_opti_=Adam_skip_=3_scali=positive_normalization_input=random_nb_ou=10_mlem_=False_A_AML=-100/out_cnn/24/out_DIP' + str(it) + '_FINAL.img'
root_random = '/home/meraslia/workspace_reco/nested_admm/data/Algo/image40_1/replicate_1/nested/Block2/config_recoI=APGMAP_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=200_tau_D=2_lr=0.01_sub_i=100_opti_=Adam_skip_=3_scali=positive_normalization_input=random_nb_ou=10_mlem_=False_A_AML=-100/out_cnn/24/out_DIP' + str(it) + '_FINAL.img'

PETImage_shape = (112,112)

# Read images
img_MR = fijii_np(root_MR, shape=(PETImage_shape))
img_intermediate = fijii_np(root_intermediate, shape=(PETImage_shape))
img_random = fijii_np(root_random, shape=(PETImage_shape))

kernel = laplacian()
gradient_img_MR = ndimage.convolve(img_MR,kernel)
gradient_img_intermediate = ndimage.convolve(img_intermediate,kernel)
gradient_img_random = ndimage.convolve(img_random,kernel)

img_list = [img_MR, img_intermediate, img_random]
gradient_img_list = [gradient_img_MR, gradient_img_intermediate, gradient_img_random]
title_list = ["MR","intermediate","random"]

for i in range(len(gradient_img_list)):

    # Show images
    # plt.figure()
    # plt.imshow(img_list[i],cmap="gray_r")
    # plt.title(title_list[i])
    # plt.figure()
    # plt.imshow(gradient_img_list[i],cmap="gray_r")
    # plt.title(title_list[i])

    # Plot profile
    line = img_list[i][30,58:78]
    plt.subplot(3,2,2*i+1)
    plt.plot(line)
    plt.title(title_list[i])
    plt.ylim([0,200])
    line = gradient_img_list[i][30,58:78]
    plt.subplot(3,2,2*i+2)
    plt.plot(line)
    plt.title("gradient " + title_list[i])
    plt.ylim([-200,200])
    
plt.show()