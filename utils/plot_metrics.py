import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

def fijii_np(path,shape,type_im='<f'):
    file_path=(path)
    dtype = np.dtype(type_im)
    fid = open(file_path, 'rb')
    data = np.fromfile(fid,dtype)
    image = data.reshape(shape)
    return image

def save_img(img,name):
    fp=open(name,'wb')
    img.tofile(fp)
    print('Succesfully save in:', name)

nb_it = 2000
# im_stacked = np.zeros((nb_it,112,112),dtype='<f')

root = "data/Algo/image50_1/replicate_1/nested/Block2/post_reco config_recoI=APGMAP_image=BSREM_it30_rho=3_adapt=nothing_mu_DI=100.1_tau_D=200_lr=0.01_opti_=Adam_skip_=3_overr=True_scali=standardization_input=CT_nb_ou=10_mlem_=False_A_AML=-10/out_cnn/24/"

global_it = 0
subfolder_list = ["out_DIP-100_epoch="]
it_list = np.arange(0,nb_it)
MSE = np.zeros(len(it_list))
SSIM = np.zeros(len(it_list))

image_gt = fijii_np("data/Algo/Data/database_v2/image50_1/image50_1.raw",(112,112))
phantom_ROI = fijii_np("data/Algo/Data/database_v2/image50_1/phantom_mask50_1.raw",(112,112))
image_gt_cropped = image_gt * phantom_ROI

for subfolder in subfolder_list:
    for it in it_list:
        filename = root + subfolder + "_it" + str(it)
        filename = root + subfolder + str(it)      

        print(it)
        
        im_it = fijii_np(filename + ".img",(112,112))
        im_it_cropped = im_it * phantom_ROI
        MSE[it] = np.mean((image_gt - im_it)**2)
        SSIM[it] = structural_similarity(np.squeeze(image_gt_cropped), np.squeeze(im_it_cropped), data_range=(im_it_cropped).max() - (im_it_cropped).min())


        root_save = root

# plt.title("MSE with GT")
plt.plot(it_list[50:],MSE[50:])
plt.xlabel("Epochs")
plt.ylabel("MSE with GT")
plt.savefig("data/Algo/image50_1/replicate_1/nested/Block2/post_reco config_recoI=APGMAP_image=BSREM_it30_rho=3_adapt=nothing_mu_DI=100.1_tau_D=200_lr=0.01_opti_=Adam_skip_=3_overr=True_scali=standardization_input=CT_nb_ou=10_mlem_=False_A_AML=-10/out_cnn/" + "MSE_GT.png")

plt.figure()
# plt.title("SSIM with GT")
plt.plot(it_list[50:],SSIM[50:])
plt.xlabel("Epochs")
plt.ylabel("SSIM with GT")
plt.savefig("data/Algo/image50_1/replicate_1/nested/Block2/post_reco config_recoI=APGMAP_image=BSREM_it30_rho=3_adapt=nothing_mu_DI=100.1_tau_D=200_lr=0.01_opti_=Adam_skip_=3_overr=True_scali=standardization_input=CT_nb_ou=10_mlem_=False_A_AML=-10/out_cnn/" + "SSIM_GT.png")
