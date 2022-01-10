#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script showing previously computed results instead of running again iterations
"""

## Python libraries

# Pytorch
from torch.utils.tensorboard import SummaryWriter

# Useful
import os

# Math
import numpy as np

# Local files to import
from utils.utils_func import *

# Configuration dictionnary for hyperparameters to tune
config = {
    "lr" : 0.001,
    "sub_iter_DIP" : 100,
    "rho" : 0.003,
    "opti_DIP" : 'Adam',
    "mlem_sequence" : False,
    "d_DD" : 6,
    "k_DD" : 32,
    "skip_connections" : False,
    "image" : "image0"
}

# For VS Code (without command line)
net = 'DIP' # Network architecture
max_iter = 20 # Outer iterations
test = 24

# Useful variables
suffix =  suffix_func(config) # suffix to make difference between raytune runs (different hyperparameters)
root=os.getcwd() # We do not use raytune, so it is local directory
subroot = root + '/data/Algo/'  # Directory root
writer = SummaryWriter()

# Define PET input dimensions according to input data dimensions
PETImage_shape_str = read_input_dim(subroot+'Data/database_v2/' + config["image"] + '/' + config["image"] + '.hdr')
PETImage_shape = input_dim_str_to_list(PETImage_shape_str)

# Metrics arrays
PSNR_recon = np.zeros(max_iter)
PSNR_norm_recon = np.zeros(max_iter)
MSE_recon = np.zeros(max_iter)
MA_cold_recon = np.zeros(max_iter)
CRC_hot_recon = np.zeros(max_iter)
CRC_bkg_recon = np.zeros(max_iter)
IR_bkg_recon = np.zeros(max_iter)

#Loading Ground Truth image to compute metrics
image_gt = fijii_np(subroot+'Data/database_v2/' + config["image"] + '/' + config["image"] + '.raw',shape=(PETImage_shape))

for i in range(max_iter):
    f = fijii_np(subroot+'Block2/out_cnn/'+ format(test)+'/out_' + net + '' + format(i) + suffix + '.img',shape=(PETImage_shape)) # loading DIP output

    # Metrics for NN output
    compute_metrics(PETImage_shape,f,image_gt,i,PSNR_recon,PSNR_norm_recon,MSE_recon,MA_cold_recon,CRC_hot_recon,CRC_bkg_recon,IR_bkg_recon,config["image"],writer=writer,write_tensorboard=True)

    # Display images in tensorboard
    #write_image_tensorboard(writer,image_init,"initialization of DIP output",suffix,image_gt) # DIP input in tensorboard
    #write_image_tensorboard(writer,image_net_input,"DIP input",suffix,image_gt) # Initialization of DIP output in tensorboard
    write_image_tensorboard(writer,image_gt,"Ground Truth",suffix,image_gt) # Ground truth image in tensorboard

    # Write image over ADMM iterations
    if ((max_iter>=10) and (i%(max_iter // 10) == 0)):

        write_image_tensorboard(writer,f,"Image over ADMM iterations (" + net + "output)",suffix,image_gt,i) # Showing all images with same contrast to compare them together
        write_image_tensorboard(writer,f,"Image over ADMM iterations (" + net + "output, FULL CONTRAST)",suffix,image_gt,i,full_contrast=True) # Showing each image with contrast = 1
    

writer.close()