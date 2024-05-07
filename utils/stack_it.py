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

task = "DNA"
# task = "denoising"
# task = "denoising_in_DNA"
# task = "likelihood_in_DNA"

im_3D = False
nb_it = 2000-10
nb_it = 995-10
nb_it = 150
it_start_denoising_in_DNA = 475
it_start_denoising_in_DNA = 0
# it_start_denoising_in_DNA = 995
# im_stacked = np.zeros((nb_it,152,232),dtype='<f')
if (task == "denoising"):
    if (im_3D):
        im_stacked = np.zeros((nb_it,172,172),dtype='<f')
    else:
        im_stacked = np.zeros((nb_it,112,112),dtype='<f')
elif (task == "DNA" or task == "denoising_in_DNA" or task == "likelihood_in_DNA"):
    if (im_3D):
        im_stacked = np.zeros((nb_it+1,172,172),dtype='<f')
    else:
        im_stacked = np.zeros((nb_it+1,112,112),dtype='<f')

# root = "/home/meraslia/workspace_reco/nested_admm/data/Algo/image40_0/replicate_9/nested/Block1/config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=10_tau_D=2_lr=0.01_sub_i=100_opti_=Adam_skip_=3_scali=standardization_input=random_nb_ou=10_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_stopp=0_saveS=0_mlem_=False/during_eq22/"
# #root = "/home/meraslia/workspace_reco/nested_admm/data/Algo/image40_0/replicate_9/nested/Block2/config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=10_tau_D=2_lr=0.01_sub_i=300_opti_=Adam_skip_=3_scali=standardization_input=random_nb_ou=10_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_stopp=0_saveS=0_mlem_=False/out_cnn/24/"
# root = "/disk/workspace_reco/nested_admm/data/Algo/image40_0/replicate_1/Gong/Block2/post_reco config_image=BSREM_it30_rho=0_adapt=nothing_"+\
#         "mu_DI=4_tau_D=2_lr=0.01_opti_=Adam_skip_=0_scali=positive_normalization_input=random_nb_ou=3_mlem_=False/out_cnn/24/"

# rho = 0.003
# root = "/home/meraslia/workspace_reco/nested_admm/data/Algo/image40_0/replicate_1/nested/Block2/config_image=BSREM_it30_rho=" + str(rho) + "_adapt=nothing_"+\
#         "mu_DI=10_tau_D=2_lr=0.01_sub_i=100_opti_=Adam_skip_=3_scali=standardization_input=random_nb_ou=10_alpha=1_adapt=both_mu_ad=2_"+\
#         "tau=100_tau_m=100_stopp=0_saveS=0_mlem_=False/out_cnn/24/"

root = "/disk/workspace_reco/nested_admm/data/Algo/image010_3D/replicate_1/nested/Block2/post_reco config_image=BSREM_it30_rho=3e-06_adapt=nothing_mu_DI=10_tau_D=2_lr=0.01_opti_=Adam_skip_=3_scali=standardization_input=random_nb_ou=10_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_stopp=0_saveS=0_mlem_=False/out_cnn/24/"
root = "data/Algo/image40_1/replicate_1/nested/Block2/post_reco config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=222_tau_D=2_lr=0.01_opti_=Adam_skip_=3_scali=positive_normalization_scali=False_input=CT_nb_ou=10_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_stopp=0_saveS=0_mlem_=False/out_cnn/24/"

# root = "data/Algo/image40_1/replicate_1/nested/Block2/post_reco config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=333_tau_D=2_lr=0.01_opti_=Adam_skip_=3_scali=positive_normalization_scali=False_input=CT_nb_ou=10_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_stopp=0_saveS=0_mlem_=False/out_cnn/24/"
root = "data/Algo/image50_0/replicate_1/nested/Block2/post_reco config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=333_tau_D=2_lr=0.01_opti_=Adam_skip_=3_scali=positive_normalization_scali=False_input=CT_nb_ou=10_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_stopp=0_saveS=0_mlem_=False/out_cnn/24/"
root = "data/Algo/image50_0/replicate_1/nested/Block2/config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=100_tau_D=2_monit=True_lr=0.01_sub_i=100_opti_=Adam_skip_=3_scali=positive_normalization_input=random_nb_ou=10_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_stopp=0_saveS=0_mlem_=False/out_cnn/24/"
root = "data/Algo/image50_0/replicate_1/nested/Block2/post_reco config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=333_tau_D=2_lr=0.01_opti_=Adam_skip_=3_scali=standardization_input=CT_nb_ou=10_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_stopp=0_saveS=0_mlem_=False/out_cnn/24/"
root = "data/Algo/image50_0/replicate_1/nested/Block2/post_reco config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=333_tau_D=2_monit=True_lr=1_opti_=Adam_skip_=3_scali=standardization_input=CT_nb_ou=10_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_stopp=0_saveS=0_mlem_=False/out_cnn/24/"
root = "data/Algo/image50_0/replicate_1/nested/Block2/post_reco config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=333_tau_D=2_lr=0.001_opti_=Adam_skip_=3_scali=standardization_input=CT_nb_ou=10_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_stopp=0_saveS=0_mlem_=False/out_cnn/24/"
root = "data/Algo/image50_0/replicate_1/nested/Block2/post_reco config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=333_tau_D=2_lr=0.01_opti_=Adam_skip_=3_scali=positive_normalization_input=CT_nb_ou=10_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_stopp=0_saveS=0_mlem_=False/out_cnn/24/"
root = "data/Algo/image010_3D/replicate_1/nested/Block2/post_reco config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=333_tau_D=2_lr=0.01_opti_=Adam_skip_=3_scali=standardization_scali=False_input=CT_nb_ou=100_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_stopp=0_saveS=0_mlem_=False/out_cnn/24/"
root = "data/Algo/image010_3D/replicate_1/nested/Block2/config_image=BSREM_it30_rho=3e-05_adapt=nothing_mu_DI=400_tau_D=2_lr=0.01_sub_i=300_opti_=Adam_skip_=3_scali=standardization_scali=False_input=CT_nb_ou=30_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_stopp=0_saveS=0_mlem_=False/out_cnn/24/"

root = "data/Algo/image010_3D/replicate_1/nested/Block2/config_image=BSREM_it30_rho=3e-05_adapt=nothing_mu_DI=600_tau_D=2_lr=0.01_sub_i=300_opti_=Adam_skip_=3_scali=standardization_scali=False_input=CT_nb_ou=30_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_stopp=0_saveS=0_mlem_=False/out_cnn/24/"
# root = "data/Algo/image010_3D/replicate_1/nested/Block2/config_image=BSREM_it30_rho=3e-05_adapt=nothing_mu_DI=600_tau_D=2_lr=0.01_sub_i=300_opti_=Adam_skip_=3_scali=standardization_scali=False_input=CT_nb_ou=1_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_stopp=0_saveS=0_mlem_=False/out_cnn/24/"

root = "data/Algo/image40_1/replicate_3/nested/Block2/config_image=BSREM_it30_rho=0.03_adapt=nothing_mu_DI=100_tau_D=2_lr=0.01_sub_i=100_opti_=Adam_skip_=3_scali=positive_normalization_input=random_nb_ou=10_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_stopp=0_saveS=0_mlem_=False/out_cnn/24/"
root = "data/Algo/image50_1/replicate_2/nested/Block2/config_recoI=APGMAP_image=BSREM_it30_rho=3_adapt=nothing_mu_DI=10_tau_D=2_lr=0.01_sub_i=30_opti_=Adam_skip_=2_scali=positive_normalization_input=CT_nb_ou=2_mlem_=False_A_AML=-10/out_cnn/24/"
# sub_iter_DIP = 100
# nb_outer_it = 10
# root = "data/Algo/image50_1/replicate_1/nested/Block2/config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=100_tau_D=2_lr=0.01_sub_i=" + str(sub_iter_DIP) + "_opti_=Adam_skip_=3_scali=standardization_input=CT_nb_ou=" + str(nb_outer_it) + "_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_stopp=0_saveS=0_mlem_=False/out_cnn/24/"
# root = "data/Algo/image40_0/replicate_1/nested/Block2/post_reco config_image=BSREM_it30_rho=0_adapt=nothing_mu_DI=50_tau_D=50_lr=0.01_overr=True_opti_=SGD_skip_=3_scali=standardization_input=CT_nb_ou=1000_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_stopp=0_saveS=0_mlem_=False/out_cnn/24/"
# root = "data/Algo/image40_0/replicate_1/nested/Block2/post_reco config_image=BSREM_it30_rho=0_adapt=nothing_mu_DI=3000_tau_D=50_lr=0.01_overr=True_opti_=SGD_skip_=3_scali=standardization_input=CT_nb_ou=1000_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_stopp=0_saveS=0_mlem_=False/out_cnn/24/"


root = "/home/meraslia/workspace_reco/nested_admm/data/Algo/image010_3D/replicate_1/nested/Block2/config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=100.123_tau_D=2_lr=0.01_sub_i=100_opti_=Adam_skip_=3_scali=standardization_input=CT_nb_ou=30_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_stopp=0_saveS=0_mlem_=False/out_cnn/24/"
root = "/home/meraslia/workspace_reco/nested_admm/data/Algo/image010_3D/replicate_1/nested/Block2/config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=100_tau_D=2_lr=0.01_sub_i=2000_opti_=Adam_skip_=3_scali=standardization_input=CT_nb_ou=100_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_stopp=0_saveS=0_mlem_=False/out_cnn/24/"
root = "data/Algo/image010_3D/replicate_1/nested/Block1/100config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=100.1_tau_D=2_lr=0.01_sub_i=100_opti_=Adam_skip_=3_scali=standardization_input=CT_nb_ou=30_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_stopp=0_saveS=0_mlem_=False/during_eq22/"

root = "data/Algo/image50_2/replicate_1/nested/Block2/config_image=BSREM_it30_rho=3_adapt=nothing_mu_DI=10.1_tau_D=2_lr=0.01_sub_i=30_opti_=Adam_skip_=3_overr=True_scali=positive_normalization_input=CT_nb_ou=3_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_mlem_=False/out_cnn/24/"
root = "/home/meraslia/workspace_reco/nested_admm/data/Algo/image50_2/replicate_1/nested/Block2/config_image=BSREM_it30_rho=0.3_adapt=nothing_mu_DI=10.1_tau_D=2_lr=0.01_sub_i=30_opti_=Adam_skip_=3_overr=True_scali=positive_normalization_input=CT_nb_ou=100_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_mlem_=False/out_cnn/24/"

root = "data/Algo/image010_3D/replicate_1/Gong/Block2/config_image=BSREM_it30_rho=3e-08_adapt=nothing_mu_DI=100.1_tau_D=2_lr=0.01_sub_i=600_opti_=Adam_skip_=3_scali=positive_normalization_input=CT_nb_ou=2_mlem_=False/out_cnn/24/"
root = "data/Algo/image4_1/replicate_1/Gong/Block2/config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=20_tau_D=2_lr=0.01_sub_i=100_opti_=Adam_skip_=3_scali=positive_normalization_input=CT_nb_ou=2_mlem_=False/out_cnn/24/"

root = "data/Algo/image4_1/replicate_1/nested/Block2/GPU_config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=100_tau_D=2_lr=0.01_sub_i=1000_opti_=Adam_skip_=3_scali=positive_normalization_input=CT_nb_ou=10_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_mlem_=False/out_cnn/24/"
root = "data/Algo/image4_1/replicate_1/nested/Block2/config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=100_tau_D=2_lr=0.01_sub_i=1000_opti_=Adam_skip_=3_scali=positive_normalization_input=CT_nb_ou=10_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_mlem_=False/out_cnn/24/"

vox = np.zeros(nb_it+1)
vox_2 = np.zeros(nb_it+1)
if (im_3D):
    num_slice = 54

# subfolder_list = ["out_DIP-100_epoch=","beforeReLU_DIP-100_epoch="]
global_it = 0
if (task == "denoising"):
    subfolder_list = ["out_DIP-100_epoch="]
    if (im_3D):
        it_list = np.arange(1,nb_it+1)
    else:
        it_list = np.arange(0,nb_it)
elif (task == "DNA"):
    subfolder_list = ["out_DIP-1_epoch="]
    it_list = np.arange(0,nb_it+1)
elif (task == "denoising_in_DNA"):
    global_it = 0
    # global_it = -1
    subfolder_list = ["out_DIP" + str(global_it) + "_epoch="]
    it_list = np.arange(0,nb_it)
elif (task == "likelihood_in_DNA"):
    global_it = 0
    subfolder_list = [str(global_it) + "_it"]
    it_list = np.arange(1,nb_it+1)
for subfolder in subfolder_list:
    for it in it_list:
        print(it)
        if (task == "denoising"):
            it_list = np.arange(1,nb_it+1)
            filename = root + subfolder + "_it" + str(it)
            filename = root + subfolder + str(it)
        elif (task == "DNA"):
            it_list = np.arange(0,nb_it+2)
            if (im_3D):
                filename = root + "out_DIP" + str(it-1) + "_FINAL" # DNA
            else:
                filename = root + "out_DIP" + str(it-1) + "_FINAL" # DNA
        if (task == "denoising_in_DNA"):
            it_list = np.arange(1,nb_it+1) 
            filename = root + subfolder + "" + str(it + it_start_denoising_in_DNA)
        if (task == "likelihood_in_DNA"):
            it_list = np.arange(1,nb_it+1) 
            filename = root + subfolder + "" + str(it)
        # subfolder = "15"
        # filename = root + subfolder + str(it-1)
        

        if (not im_3D):
            im_it = fijii_np(filename + ".img",(112,112))
            vox[it] = im_it[10,10]
            vox_2[it] = im_it[10,11]
            im_stacked[it,:,:] = im_it
        if (im_3D):
            # im_it = fijii_np(filename + ".img",(127,152,232))
            im_it = fijii_np(filename + ".img",(127,172,172))
            vox[it-1] = im_it[num_slice,45,45]
            vox_2[it-1] = im_it[num_slice,45,46]
            im_stacked[it,:,:] = im_it[num_slice,:,:]


        root_save = root

    # root = "data/Algo/image4_1/replicate_1/nested/Block2/GPU_config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=100_tau_D=2_lr=0.01_sub_i=1000_opti_=Adam_skip_=3_scali=positive_normalization_input=CT_nb_ou=10_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_mlem_=False/out_cnn/"
    root = "data/Algo/image4_1/replicate_1/nested/Block2/config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=100_tau_D=2_lr=0.01_sub_i=1000_opti_=Adam_skip_=3_scali=positive_normalization_input=CT_nb_ou=10_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_mlem_=False/out_cnn/"


    if (im_3D):
        save_img(im_stacked,root + "0" + subfolder + "it_" + str(it) + "num_slice_" + str(num_slice) + "_stacked_" + str(task) + ".img")
    else:
        save_img(im_stacked,root + "0" + subfolder + "it_" + str(it) + "_stacked_" + str(task) + ".img")
    # save_img(im_stacked,root_save + str(rho) + subfolder + "it_" + str(it) + "_stacked.img")

    # import matplotlib.pyplot as plt
    # plt.plot(vox[30:])
    # plt.plot(vox_2[30:])
    # plt.legend(["vox","vox_2"])
    # plt.show()