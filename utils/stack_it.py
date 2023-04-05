import numpy as np
from pathlib import Path

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

nb_it = 300
im_stacked = np.zeros((nb_it,112,112),dtype='<f')
root = "/home/meraslia/workspace_reco/nested_admm/data/Algo/image40_0/replicate_9/nested/Block1/config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=10_tau_D=2_lr=0.01_sub_i=100_opti_=Adam_skip_=3_scali=standardization_input=random_nb_ou=10_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_stopp=0_saveS=0_mlem_=False/during_eq22/"
#root = "/home/meraslia/workspace_reco/nested_admm/data/Algo/image40_0/replicate_9/nested/Block2/config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=10_tau_D=2_lr=0.01_sub_i=300_opti_=Adam_skip_=3_scali=standardization_input=random_nb_ou=10_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_stopp=0_saveS=0_mlem_=False/out_cnn/24/"

vox = np.zeros(nb_it)

for it in range(1,nb_it+1):
    subfolder = "15"
    #subfolder = "out_DIP14_epoch="
    filename = root + subfolder + "_it" + str(it)
    #filename = root + subfolder + str(it)
    im_it = fijii_np(filename + ".img",(112,112))
    vox[it-1] = im_it[50,50]
    im_stacked[it-1,:,:] = im_it

save_img(im_stacked,root + "0" + subfolder + "it_" + str(it) + "_stacked.img")

import matplotlib.pyplot as plt
plt.plot(vox)
plt.show()