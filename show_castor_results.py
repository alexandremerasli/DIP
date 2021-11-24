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
import sys

# Math
import numpy as np
import matplotlib.pyplot as plt

# Local files to import
from utils_func import *

print(len(sys.argv))
print(sys.argv)
if (len(sys.argv) - 1 == 3):
    opti = sys.argv[1] # CASToR optimizer
    max_iter = int(sys.argv[2]) # Number of outer iterations
    test = int(sys.argv[3]) # Label of the experiment
else:
    ## Arguments for linux command to launch script
    # Creating arguments
    parser = argparse.ArgumentParser(description='DIP + ADMM computation')
    parser.add_argument('--opti', type=str, dest='opti', help='CASToR optimizer')
    parser.add_argument('--max_iter', type=int, dest='max_iter', help='number of outer iterations')
    parser.add_argument('--test', type=str, dest='test', help='Label of experiment')

    # Retrieving arguments in this python script
    args = parser.parse_args()
    if (args.opti is not None): # Must check if all args are None    
        opti = args.opti
        max_iter = int(args.max_iter)
        test = int(args.test)
    else: # For VS Code (without command line)
        opti = 'BSREM' # CASToR optimizer
        max_iter = 10 # Optimizer number of iterations
        test = 24

# Useful variables
root=os.getcwd() # We do not use raytune, so it is local directory
subroot = root + '/data/Algo/'  # Directory root
writer = SummaryWriter()

# Define PET input dimensions according to input data dimensions
PETImage_shape_str = read_input_dim(subroot+'Data/castor_output_it60.hdr')
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
image_gt = fijii_np(subroot+'Data/phantom/phantom_act.img',shape=(PETImage_shape))

for i in range(1,max_iter):
    print(i)
    f = fijii_np(subroot+'Comparison/' + opti + '/' + opti + '_it' + format(i) + '.img',shape=(PETImage_shape)) # loading optimizer output

    # Metrics for NN output
    compute_metrics(PETImage_shape,f,image_gt,i,PSNR_recon,PSNR_norm_recon,MSE_recon,MA_cold_recon,CRC_hot_recon,CRC_bkg_recon,IR_bkg_recon,bias_cold_recon,bias_hot_recon,writer=writer,write_tensorboard=True)

    # Display images in tensorboard
    #write_image_tensorboard(writer,image_init,"initialization of DIP output") # DIP input in tensorboard
    #write_image_tensorboard(writer,image_net_input,"DIP input") # Initialization of DIP output in tensorboard
    write_image_tensorboard(writer,image_gt,"Ground Truth") # Ground truth image in tensorboard

    # Write image over ADMM iterations
    if ((max_iter>=10) and (i%(max_iter // 10) == 0)):

        write_image_tensorboard(writer,f,"Image over " + opti + " iterations",i) # Showing all images with same contrast to compare them together
        write_image_tensorboard(writer,f,"Image over " + opti + " iterations (FULL CONTRAST)",i,full_contrast=True) # Showing each image with contrast = 1
    

writer.close()