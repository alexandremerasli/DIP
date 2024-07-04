import numpy as np
from pathlib import Path

def fijii_np(path,shape,type_im='<f'):
    """"Transforming raw data to numpy array"""               
    file_path=(path)
    nb_dimensions = len(shape)
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

subroot = "data/Algo/"
folder = subroot + "image4_1/replicate_1/DNA/Block2/config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=100_tau_D=2_lr=0.01_sub_i=1000_opti_=Adam_skip_=3_scali=positive_normalization_input=CT_nb_ou=10_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_mlem_=False/out_cnn/24/"

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
            filename = folder + subfolder + "_it" + str(it)
            filename = folder + subfolder + str(it)
        elif (task == "DNA"):
            it_list = np.arange(0,nb_it+2)
            if (im_3D):
                filename = folder + "out_DIP" + str(it-1) + "_FINAL" # DNA
            else:
                filename = folder + "out_DIP" + str(it-1) + "_FINAL" # DNA
        if (task == "denoising_in_DNA"):
            it_list = np.arange(1,nb_it+1) 
            filename = folder + subfolder + "" + str(it + it_start_denoising_in_DNA)
        if (task == "likelihood_in_DNA"):
            it_list = np.arange(1,nb_it+1) 
            filename = folder + subfolder + "" + str(it)
        # subfolder = "15"
        # filename = folder + subfolder + str(it-1)
        

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


        folder_save = folder

    # folder = subroot + "/image4_1/replicate_1/DNA/Block2/GPU_config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=100_tau_D=2_lr=0.01_sub_i=1000_opti_=Adam_skip_=3_scali=positive_normalization_input=CT_nb_ou=10_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_mlem_=False/out_cnn/"
    folder = subroot + "/image4_1/replicate_1/DNA/Block2/config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=100_tau_D=2_lr=0.01_sub_i=1000_opti_=Adam_skip_=3_scali=positive_normalization_input=CT_nb_ou=10_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_mlem_=False/out_cnn/"


    if (im_3D):
        save_img(im_stacked,folder + "0" + subfolder + "it_" + str(it) + "num_slice_" + str(num_slice) + "_stacked_" + str(task) + ".img")
    else:
        save_img(im_stacked,folder + "0" + subfolder + "it_" + str(it) + "_stacked_" + str(task) + ".img")
    # save_img(im_stacked,folder_save + str(rho) + subfolder + "it_" + str(it) + "_stacked.img")

    # import matplotlib.pyplot as plt
    # plt.plot(vox[30:])
    # plt.plot(vox_2[30:])
    # plt.legend(["vox","vox_2"])
    # plt.show()