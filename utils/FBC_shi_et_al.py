import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import imageio
from multiprocessing import Pool
from multiprocessing import Process
from functools import partial

def fijii_np(path,shape,type_im='<f'):
    """"Transforming raw data to numpy array"""
    file_path=(path)
    dtype = np.dtype(type_im)
    fid = open(file_path, 'rb')
    data = np.fromfile(fid,dtype)
    image = data.reshape(shape)
    return image

def get_circular_statastic(img_it, img_gt, size=0.2):
    assert(size>0 and size<1)

    ftimage_it = np.fft.fft2(img_it)
    ftimage_it = abs(np.fft.fftshift(ftimage_it))

    ftimage_gt = np.fft.fft2(img_gt)
    ftimage_gt = abs(np.fft.fftshift(ftimage_gt))

    m_data = ftimage_it/(ftimage_gt+1e-8)
    # m_data = np.clip(m_data, 0, 1)
    m_data = np.abs(m_data)

    h,w = m_data.shape

    center = (int(w/2), int(h/2))
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    avg_mask_list = []
    pre_mask = np.zeros((h,w))
    for sz in np.linspace(size, 1, int(1/size)):

        radius = center[0]*sz#pow(center[0]**2+center[1]**2,0.5)
        mask = dist_from_center <= radius
        mask = mask.astype(np.int32)

        mask_sz = (mask-pre_mask).astype(np.int32) # Frequency band in the Fourier image (5 different masks by default like in Shi et al. paper)
        pre_mask = mask

        # avg_mask_list.append(np.sum(mask_sz*m_data)/np.sum(mask_sz))
        ftimage_it_masked = mask_sz*ftimage_it
        ftimage_gt_masked = mask_sz*ftimage_gt
        cross_correlation = np.sum((ftimage_it_masked * ftimage_gt_masked)) / np.sqrt(np.sum(ftimage_it_masked**2) * np.sum(ftimage_gt_masked**2))
        avg_mask_list.append(cross_correlation)

    return avg_mask_list

def write_fbc_in_csv(iter_,iters_path,PETImage_shape,gt,sub_iter_DIP,train_res_list):
    iteration = iter_.split("=")[1].split('.')[0]
    if (int(iteration) < sub_iter_DIP):
        image_path = os.path.join(iters_path, iter_)
        image_np = fijii_np(image_path,PETImage_shape)
        res_FBC = get_circular_statastic(image_np,gt)
        # res_FBC[int(iteration)] = l[0][int(iteration)]

        row = [iteration,*res_FBC]
        # row = [*res_FBC]
        # iter_res = pd.DataFrame([row], columns=['iteration','lowest','low','medium','high','highest'])
        # iter_res = pd.DataFrame([row])
        return row
        train_res_list[int(iteration)] = iter_res
        train_res = pd.concat(train_res_list,axis=0)
        
    else:
        return []

def fbc_to_csv(iters_path,target_filename,gt,PETImage_shape,sub_iter_DIP):
    iters_path_filtered = list()
    for x in os.listdir(iters_path):
        if x.split(".")[0][-1] != 'd':
            iters_path_filtered.append(x)
    iters = sorted(iters_path_filtered, key=lambda x: int(x.split("=")[1].split('.')[0]))

    train_res_list = np.zeros(len(iters))
    res_FBC = np.zeros(len(iters))
    # for it in iters:
        
    write_fbc_in_csv_partial=partial(write_fbc_in_csv,iters_path=iters_path,PETImage_shape=PETImage_shape,gt=gt,sub_iter_DIP=sub_iter_DIP,train_res_list=train_res_list)
    write_fbc_in_csv_partial(iters[0])
    with Pool(35) as p:
        train_res = p.map(write_fbc_in_csv_partial,iters)
    # train_res = pd.DataFrame([train_res], columns=['iteration','lowest','low','medium','high','highest'])
    train_res = pd.DataFrame(train_res)
    train_res.columns = ['iteration','lowest','low','medium','high','highest']
    train_res.to_csv(target_filename)

    # processes = []
    #     p = Process(target=write_fbc_in_csv, args=(iteration,iters_path,PETImage_shape,gt))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()

# Compute FBC from folder of images
my_PETImage_shape = (112,112)
my_gt = fijii_np("data/Algo/Data/database_v2/image40_1/image40_1.img",my_PETImage_shape)
my_corrupt = fijii_np("data/Algo/Data/initialization/image40_1/BSREM_30it/replicate_13/BSREM_it30.img",my_PETImage_shape)
lr_list = [0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5]
lr_list = [1]
sub_iter_DIP = 5000
ref_image_list = [my_gt,my_corrupt]
ref_image_names_list = ["gt","corrupt"]
for i in range(len(ref_image_list)):
    ref_image = ref_image_list[i]
    ref_image_name = ref_image_names_list[i]
    FBC_video = []
    FBC_video_same_scale = []
# for ref_image in [my_corrupt]:
#     ref_image_name = "corrupt"
# for ref_image in [my_gt]:
#     ref_image_name = "gt"
    for p in range(len(lr_list)):
        print("lr =",lr_list[p])
        short_subfolder = "data/Algo/image40_1/replicate_13/nested/Block2"
        subfolder = short_subfolder + "/post_reco config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=10_tau_D=2.5_monit=False_lr=" + str(lr_list[p]) + "_opti_=Adam_skip_=3_scali=normalization_input=CT_nb_ou=2_alpha=1_adapt=both_mu_ad=2_tau=2_tau_m=100_stopp=0.001_saveS=1_mlem_=False/out_cnn/"
        subfolder = short_subfolder + "/post_reco config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=10_tau_D=2.55_monit=True_lr=" + str(lr_list[p]) + "_opti_=Adam_skip_=3_scali=normalization_input=CT_nb_ou=2_alpha=1_adapt=both_mu_ad=2_tau=2_tau_m=100_stopp=0.001_saveS=1_mlem_=False/out_cnn/"
        my_iters_path = subfolder + "24/"
        my_FBC_output_path = subfolder + "FBC/"
        Path(my_FBC_output_path).mkdir(parents=True, exist_ok=True)
        my_output_csv_path = my_FBC_output_path + "FBC_" + str(ref_image_name) + "_" + str(sub_iter_DIP) + "it.csv"
        fbc_to_csv(my_iters_path,my_output_csv_path,ref_image,my_PETImage_shape,sub_iter_DIP)

        # Plot
        df = pd.read_csv(my_output_csv_path)
        iterations_csv = df["iteration"]
        frequency_band_list = ["lowest","low","medium","high","highest"]
        freq_csv = 5*[0]
        fig, ax = plt.subplots()
        for i in range(len(frequency_band_list)):
            freq = frequency_band_list[i]
            freq_csv[i] = df[freq]
            ax.plot(iterations_csv,freq_csv[i],label=freq)
        ax.legend()
        ax.set_title("lr = " + str(lr_list[p]))
        fig.savefig(my_FBC_output_path + "FBC_plot_" + str(ref_image_name) + "_" + str(sub_iter_DIP) + "it.png")
        # plt.show()
        ax.set_ylim([0,1])
        fig.savefig(my_FBC_output_path + "FBC_plot_" + str(ref_image_name) + "_" + str(sub_iter_DIP) + "it_same_scale.png")
        print("end")

        # Concatenate PNG figures
        FBC_video.append(imageio.imread(my_FBC_output_path + "FBC_plot_" + str(ref_image_name) + "_" + str(sub_iter_DIP) + "it.png"))
        FBC_video_same_scale.append(imageio.imread(my_FBC_output_path + "FBC_plot_" + str(ref_image_name) + "_" + str(sub_iter_DIP) + "it_same_scale.png"))

    # PNG to video
    imageio.mimsave(short_subfolder + "FBC_plot_" + str(ref_image_name) + "_" + str(sub_iter_DIP) + "it.avi", FBC_video,fps=1)
    imageio.mimsave(short_subfolder + "FBC_plot_" + str(ref_image_name) + "_" + str(sub_iter_DIP) + "it_same_scale.avi", FBC_video_same_scale,fps=1)