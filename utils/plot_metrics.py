import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

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

nb_it = 2000
# im_stacked = np.zeros((nb_it,112,112),dtype='<f')

subroot = "data/Algo/"
folder = subroot + "/image50_1/replicate_1/DNA/Block2/post_reco config_recoI=APPGML_image=BSREM_it30_rho=3_adapt=nothing_mu_DI=100.1_tau_D=200_lr=0.01_opti_=Adam_skip_=3_overr=True_scali=standardization_input=CT_nb_ou=10_mlem_=False_A_AML=-10/out_cnn/24/"

global_it = 0
subfolder_list = ["out_DIP-100_epoch="]
it_list = np.arange(0,nb_it)
MSE = np.zeros(len(it_list))
SSIM = np.zeros(len(it_list))

image_gt = fijii_np(subroot + "/Data/database_v2/image50_1/image50_1.img",(112,112))
phantom_ROI = fijii_np(subroot + "/Data/database_v2/image50_1/phantom_mask50_1.raw",(112,112))
image_gt_cropped = image_gt * phantom_ROI

for subfolder in subfolder_list:
    for it in it_list:
        filename = folder + subfolder + "_it" + str(it)
        filename = folder + subfolder + str(it)      

        print(it)
        
        im_it = fijii_np(filename + ".img",(112,112))
        im_it_cropped = im_it * phantom_ROI
        MSE[it] = np.mean((image_gt - im_it)**2)
        SSIM[it] = structural_similarity(np.squeeze(image_gt_cropped), np.squeeze(im_it_cropped), data_range=(im_it_cropped).max() - (im_it_cropped).min())


        root_save = folder

# plt.title("MSE with GT")
plt.plot(it_list[50:],MSE[50:])
plt.xlabel("Epochs")
plt.ylabel("MSE with GT")
plt.savefig(subroot + "/image50_1/replicate_1/DNA/Block2/post_reco config_recoI=APPGML_image=BSREM_it30_rho=3_adapt=nothing_mu_DI=100.1_tau_D=200_lr=0.01_opti_=Adam_skip_=3_overr=True_scali=standardization_input=CT_nb_ou=10_mlem_=False_A_AML=-10/out_cnn/" + "MSE_GT.png")

plt.figure()
# plt.title("SSIM with GT")
plt.plot(it_list[50:],SSIM[50:])
plt.xlabel("Epochs")
plt.ylabel("SSIM with GT")
plt.savefig(subroot + "/image50_1/replicate_1/DNA/Block2/post_reco config_recoI=APPGML_image=BSREM_it30_rho=3_adapt=nothing_mu_DI=100.1_tau_D=200_lr=0.01_opti_=Adam_skip_=3_overr=True_scali=standardization_input=CT_nb_ou=10_mlem_=False_A_AML=-10/out_cnn/" + "SSIM_GT.png")
