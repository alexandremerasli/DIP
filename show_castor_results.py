#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script showing previously computed results instead of running again iterations
"""

## Python libraries

# Pytorch
from torch.utils import tensorboard
from torch.utils.tensorboard import SummaryWriter

# Useful
import os
import argparse
import sys

# Math
import numpy as np

# Local files to import
from utils.utils_func import *

if (len(sys.argv) - 1 == 6):
    opti = sys.argv[1] # CASToR optimizer
    max_iter = int(sys.argv[2]) # Number of outer iterations
    test = int(sys.argv[3]) # Label of the experiment
    suffix = sys.argv[4] # Suffix containing hyperparameters configuration (for ADMMLim)
    image = sys.argv[5] # Image (phantom) to choose
    beta = eval(sys.argv[6]) # Penalty strength beta
else:
    ## Arguments for linux command to launch script
    # Creating arguments
    parser = argparse.ArgumentParser(description='DIP + ADMM computation')
    parser.add_argument('--opti', type=str, dest='opti', help='CASToR optimizer')
    parser.add_argument('--max_iter', type=int, dest='max_iter', help='number of outer iterations')
    parser.add_argument('--test', type=str, dest='test', help='Label of experiment')
    parser.add_argument('--beta', type=str, dest='beta', help='penalty strength (beta)', nargs='+')
    parser.add_argument('--image', type=str, dest='image', help='phantom image from database')

    # Retrieving arguments in this python script
    args = parser.parse_args()
    if (args.opti is not None): # Must check if all args are None    
        opti = args.opti
        max_iter = int(args.max_iter)
        test = int(args.test)
        image = args.image # phantom image from database
        beta = list(args.beta)
        beta = [float(i) for i in beta]
    else: # For VS Code (without command line)
        opti = 'MLEM' # CASToR optimizer
        max_iter = 10 # Optimizer number of iterations
        test = 24
        image = "image0"
        beta = [0.01]

if (opti != 'ADMMLim'):
    suffix = ""

# Useful variables
root=os.getcwd() # We do not use raytune, so it is local directory
subroot = root + '/data/Algo/'  # Directory root

# Define PET input dimensions according to input data dimensions
PETImage_shape_str = read_input_dim(subroot+'Data/database_v2/' + image + '/' + image + '.hdr')
PETImage_shape = input_dim_str_to_list(PETImage_shape_str)

#Loading Ground Truth image to compute metrics
image_gt = fijii_np(subroot+'Data/database_v2/' + image + '/' + image + '.raw',shape=(PETImage_shape))

writer = SummaryWriter()

for p in range(len(beta)):
    # Metrics arrays
    PSNR_recon = np.zeros(max_iter)
    PSNR_norm_recon = np.zeros(max_iter)
    MSE_recon = np.zeros(max_iter)
    MA_cold_recon = np.zeros(max_iter)
    CRC_hot_recon = np.zeros(max_iter)
    CRC_bkg_recon = np.zeros(max_iter)
    IR_bkg_recon = np.zeros(max_iter)
    for i in range(1,max_iter):
        print(i)
        if (opti == 'ADMMLim'):
            f = fijii_np(subroot+'Comparison/' + opti + '/' + suffix + '/ADMM/0_' + format(i) + '_it1'  + '.img',shape=(PETImage_shape)) # loading optimizer output
        else:
            f = fijii_np(subroot+'Comparison/' + opti + '_beta_' + str(beta[p]) + '/' +  opti + '_beta_' + str(beta[p]) + '_it' + format(i) + '.img',shape=(PETImage_shape)) # loading optimizer output

        # Metrics for NN output
        compute_metrics(PETImage_shape,f,image_gt,i,PSNR_recon,PSNR_norm_recon,MSE_recon,MA_cold_recon,CRC_hot_recon,CRC_bkg_recon,IR_bkg_recon,image,writer=writer,write_tensorboard=True)

        # Display images in tensorboard
        #write_image_tensorboard(writer,image_init,"initialization of DIP output",suffix,image_gt) # DIP input in tensorboard
        #write_image_tensorboard(writer,image_net_input,"DIP input",suffix,image_gt) # Initialization of DIP output in tensorboard
        write_image_tensorboard(writer,image_gt,"Ground Truth",suffix,image_gt) # Ground truth image in tensorboard

        # Write image over ADMM iterations
        if (((max_iter>=10) and (i%(max_iter // 10) == 0)) or (max_iter<10)):
            beta_string = ''
            if (len(beta) > 1):
                beta_string = ', beta = ' + str(beta[p])
            write_image_tensorboard(writer,f,"Image over " + opti + " iterations" + beta_string,suffix,image_gt,i) # Showing all images with same contrast to compare them together
            write_image_tensorboard(writer,f,"Image over " + opti + " iterations (FULL CONTRAST)" + beta_string,suffix,image_gt,i,full_contrast=True) # Showing each image with contrast = 1
    

writer.close()