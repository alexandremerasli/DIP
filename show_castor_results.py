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
import argparse

# Math
import numpy as np
import matplotlib.pyplot as plt

# Local files to import
from utils_func import *

# For VS Code (without command line)
opti = 'BSREM' # CASToR optimizer
max_iter = 30 # Optimizer number of iterations
test = 24

# Useful variables
root=os.getcwd() # We do not use raytune, so it is local directory
subroot = root + '/data/Algo/'  # Directory root
writer = SummaryWriter()

# Define PET input dimensions according to input data dimensions
PETImage_shape_str = read_input_dim()
PETImage_shape = input_dim_str_to_list(PETImage_shape_str)

# Metrics arrays
PSNR_recon = np.zeros(max_iter)
PSNR_norm_recon = np.zeros(max_iter)
MSE_recon = np.zeros(max_iter)
MA_cold_recon = np.zeros(max_iter)
CRC_hot_recon = np.zeros(max_iter)
CRC_bkg_recon = np.zeros(max_iter)
IR_bkg_recon = np.zeros(max_iter)
bias_cold_recon = np.zeros(max_iter)
bias_hot_recon = np.zeros(max_iter)

#Loading Ground Truth image to compute metrics
image_gt = fijii_np(subroot+'Block2/data/phantom_act.img',shape=(PETImage_shape))

for i in range(1,max_iter):
    f = fijii_np(subroot+'Comparaison/' + opti + '/' + opti + '_it' + format(i) + '.img',shape=(PETImage_shape)) # loading optimizer output

    # Metrics for NN output
    compute_metrics(f,image_gt,i,max_iter,PSNR_recon,PSNR_norm_recon,MSE_recon,MA_cold_recon,CRC_hot_recon,CRC_bkg_recon,IR_bkg_recon,bias_cold_recon,bias_hot_recon,writer=writer,write_tensorboard=True)

    # Display images in tensorboard
    #write_image_tensorboard(writer,image_init,"initialization of DIP output") # DIP input in tensorboard
    #write_image_tensorboard(writer,image_net_input,"DIP input") # Initialization of DIP output in tensorboard
    write_image_tensorboard(writer,image_gt,"Ground Truth") # Ground truth image in tensorboard

    # Write image over ADMM iterations
    if ((i%(max_iter // 10) == 0)):
        write_image_tensorboard(writer,f,"Image over " + opti + " iterations",i) # Showing all images with same contrast to compare them together
        write_image_tensorboard(writer,f,"Image over " + opti + " iterations (FULL CONTRAST)",i,full_contrast=True) # Showing each image with contrast = 1
    

writer.close()