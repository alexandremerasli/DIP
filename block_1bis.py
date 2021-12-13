#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Apr  3 17:19:59 2020

@author: hernan (2020) then alexandre (2021)

Variables :
image_sens : sensitivity image
image_gt : ground truth image
x_out : DIP output
x : image x at iteration i
image_init : initialization of image x (x_0)
f_init : initialization of DIP output (f_0), BUT NOT theta_0
"""

## Python libraries

# Pytorch
import torch
from torch.utils.tensorboard import SummaryWriter

# Useful
import os
from pathlib import Path
import argparse
import time
import subprocess
from functools import partial
from ray import tune

# Math
import numpy as np
import matplotlib.pyplot as plt

# Local files to import
from utils_func import *

def admm_loop(config, args, root):

    # Retrieving arguments in this function
    processing_unit = args.proc
    max_iter = args.max_iter # Outer iterations 
    finetuning = args.finetuning # Finetuning or not for the DIP optimizations (block 2)

    test = 24  # Label of the experiment
    
    '''
    Creation of directory 
    '''
    suffix =  suffix_func(config) # suffix to make difference between raytune runs (different hyperparameters)
    subroot = root + '/data/Algo/'  # Directory root

    Path(subroot+'Block1/' + suffix + '/before_eq22').mkdir(parents=True, exist_ok=True) # CASToR path
    Path(subroot+'Block1/' + suffix + '/during_eq22').mkdir(parents=True, exist_ok=True) # CASToR path
    Path(subroot+'Block1/' + suffix + '/out_eq22').mkdir(parents=True, exist_ok=True) # CASToR path

    Path(subroot+'Images/out_final/'+format(test)+'/').mkdir(parents=True, exist_ok=True) # Output of the framework (Last output of the DIP)

    Path(subroot+'Block2/checkpoint/'+format(test)+'/').mkdir(parents=True, exist_ok=True)
    Path(subroot+'Block2/out_cnn/'+ format(test)+'/').mkdir(parents=True, exist_ok=True) # Output of the DIP block every outer iteration
    Path(subroot+'Block2/out_cnn/vae').mkdir(parents=True, exist_ok=True) # Output of the DIP block every outer iteration
    Path(subroot+'Block2/out_cnn/cnn_metrics/'+ format(test)+'/').mkdir(parents=True, exist_ok=True) # DIP block metrics
    Path(subroot+'Block2/out_cnn/'+ format(test)+'/like/').mkdir(parents=True, exist_ok=True) # folder for Likelihood calculation (using CASTOR)
    Path(subroot+'Block2/x_label/'+format(test) + '/').mkdir(parents=True, exist_ok=True) # x corrupted - folder

    Path(subroot+'Block2/checkpoint/'+format(test)+'/').mkdir(parents=True, exist_ok=True)
    Path(subroot+'Block2/out_cnn/'+ format(test)+'/').mkdir(parents=True, exist_ok=True)
    Path(subroot+'Block2/mu/'+ format(test)+'/').mkdir(parents=True, exist_ok=True)
    Path(subroot+'Block2/out_cnn/cnn_metrics/'+ format(test)+'/').mkdir(parents=True, exist_ok=True)

    Path(subroot+'Comparison/MLEM/').mkdir(parents=True, exist_ok=True) # CASTor path
    Path(subroot+'Comparison/BSREM/').mkdir(parents=True, exist_ok=True) # CASTor path

    Path(subroot+'Config/').mkdir(parents=True, exist_ok=True) # CASTor path

    Path(subroot+'Data/initialization').mkdir(parents=True, exist_ok=True)

    # Define PET input dimensions according to input data dimensions
    PETImage_shape_str = read_input_dim(subroot+'Data/castor_output_it60.hdr')
    PETImage_shape = input_dim_str_to_list(PETImage_shape_str)

    """
    Initialization : command line for ML-EM using CASTOR framework
    """

    # Config dictionnary for hyperparameters
    rho = config["rho"]
    alpha = config["alpha"]
    sub_iter_MAP = config["sub_iter_MAP"]
    net = config["net"]
    if config["input"] == 'uniform': # Do not standardize or normalize if uniform, otherwise NaNs
        config["scaling"] = "nothing"
    scaling_input = config["scaling"]
    np.save(subroot + 'Config/config' + suffix + '.npy', config) # Save this configuration of hyperparameters, and reload it at the beginning of block 2 thanks to suffix (passed in subprocess call argumentsZ)

    """
    Initialization : variables
    """

    # ADMM variables
    mu = 0* np.ones((PETImage_shape[0], PETImage_shape[1]), dtype='<f')
    save_img(mu,subroot+'Block2/mu/'+ format(test)+'/mu_' + format(-1) + suffix + '.img') # saving mu

    x_out = np.zeros((PETImage_shape[0], PETImage_shape[1]), dtype='<f') # output of DIP

    writer = SummaryWriter()

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

    """
    CASTOR framework
    """

    ## Loading images (NN input, first DIP output, GT)
    # Loading DIP input
    # Creating random image input for DIP while we do not have CT, but need to be removed after
    create_input(net,PETImage_shape,config) # to be removed when CT will be used instead of random input. DO NOT PUT IT IN BLOCK 2 !!!
    # Loading DIP input (we do not have CT-map, so random image created in block 1)
    image_net_input = load_input(net,PETImage_shape,config) # Scaling of DIP input. DO NOT CREATE RANDOM INPUT IN BLOCK 2 !!! ONLY AT THE BEGINNING, IN BLOCK 1    
    image_net_input_scale = rescale_imag(image_net_input,scaling_input)[0] # Rescale of DIP input
    # DIP input image, numpy --> torch
    image_net_input_torch = torch.Tensor(image_net_input_scale)
    # Adding dimensions to fit network architecture
    if (net == 'DIP' or net == 'DIP_VAE'):
        image_net_input_torch = image_net_input_torch.view(1,1,PETImage_shape[0],PETImage_shape[1]) # For DIP
    else:
        if (net == 'DD'):
            input_size_DD = int(PETImage_shape[0] / (2**config["d_DD"])) # if original Deep Decoder (i.e. only with decoder part)
            image_net_input_torch = image_net_input_torch.view(1,config["k_DD"],input_size_DD,input_size_DD) # For Deep Decoder, if original Deep Decoder (i.e. only with decoder part)
        elif (net == 'DD_AE'):
            input_size_DD = PETImage_shape[0] # if auto encoder based on Deep Decoder
            image_net_input_torch = image_net_input_torch.view(1,1,input_size_DD,input_size_DD) # For Deep Decoder, if auto encoder based on Deep Decoder
    torch.save(image_net_input_torch,subroot + 'Data/initialization/image_net_input_torch.pt')

    # Ininitializing DIP output and first image x with f_init and image_init
    image_init_path_without_extension = ''
    #image_init = np.ones((PETImage_shape[0],PETImage_shape[1])) # initializing CASToR MAP reconstruction with uniform image with ones.
    image_init_path_without_extension = '1_im_value_cropped'
    #image_init_path_without_extension = 'BSREM_it30_REF_cropped'
    
    #f_init = fijii_np(subroot + 'Data/initialization/' + 'BSREM_it30_REF_cropped.img',shape=(PETImage_shape))
    #f_init = fijii_np(subroot+'Comparison/MLEM/MLEM_converge_avec_post_filtre.img',shape=(PETImage_shape))
    f_init = np.ones((PETImage_shape[0],PETImage_shape[1]), dtype='<f')
    save_img(f_init,subroot+'Block2/out_cnn/'+ format(test)+'/out_' + net + '' + format(-1) + suffix + '.img') # saving DIP output

    #Loading Ground Truth image to compute metrics
    image_gt = fijii_np(subroot+'Data/phantom/phantom_act.img',shape=(PETImage_shape))

    f = f_init  # Initializing DIP output with f_init

    for i in range(max_iter):
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Outer iteration !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', i)
        start_time_outer_iter = time.time()
        
        # Define command line to run ADMM with CASToR
        if (i==0): # For first iteration, put rho to zero
            castor_command_line_x = castor_admm_command_line(PETImage_shape_str, alpha, 0)
        else:
            castor_command_line_x = castor_admm_command_line(PETImage_shape_str, alpha, rho)
        
        # Reconstruction with CASToR (first equation of ADMM)
        x_label = castor_reconstruction(writer, i, castor_command_line_x, subroot, sub_iter_MAP, test, config, suffix, f, mu, PETImage_shape, image_init_path_without_extension) # without ADMMLim file

        # Write image over ADMM iterations
        if ((max_iter>=10) and (i%(max_iter // 10) == 0) or True):

            write_image_tensorboard(writer,x_label,"Corrupted image (x_label) over ADMM iterations",suffix,i) # Showing all corrupted images with same contrast to compare them together
            write_image_tensorboard(writer,x_label,"Corrupted image (x_label) over ADMM iterations (FULL CONTRAST)",suffix,i,full_contrast=True) # Showing each corrupted image with contrast = 1
        
        # Block 2 - CNN - 10 iterations
        start_time_block2= time.time()
        successful_process = subprocess.call(["python3", root+"/block_2_bis_lightning.py", str(i), str(test), net, processing_unit, finetuning, PETImage_shape_str, root, suffix]) #Calling block 2 algorithm and passing variables (current iter-number of epochs and test number, chosen net, processing unit, way to do finetuning, and image dimensions)
        if successful_process != 0: # if there is an error in block2, then stop the run
            raise ValueError('An error occured in block2 computation. Stopping overall iterations.')
        print("--- %s seconds - DIP block ---" % (time.time() - start_time_block2))
        f = fijii_np(subroot+'Block2/out_cnn/'+ format(test)+'/out_' + net + '' + format(i) + suffix + '.img',shape=(PETImage_shape)) # loading DIP output

        # Metrics for NN output
        compute_metrics(PETImage_shape,f,image_gt,i,PSNR_recon,PSNR_norm_recon,MSE_recon,MA_cold_recon,CRC_hot_recon,CRC_bkg_recon,IR_bkg_recon,bias_cold_recon,bias_hot_recon,writer=writer,write_tensorboard=True)

        # Block 3 - equation 15 - mu
        mu = x_label- f
        save_img(mu,subroot+'Block2/mu/'+ format(test)+'/mu_' + format(i) + suffix + '.img') # saving mu

        write_image_tensorboard(writer,mu,"mu(FULL CONTRAST)",suffix,i,full_contrast=True) # Showing all corrupted images with same contrast to compare them together
        print("--- %s seconds - outer_iteration ---" % (time.time() - start_time_outer_iter))

        # Write image over ADMM iterations
        if ((max_iter>=10) and (i%(max_iter // 10) == 0) or (max_iter<10)):
            write_image_tensorboard(writer,f,"Image over ADMM iterations (" + net + "output)",suffix,i) # Showing all images with same contrast to compare them together
            write_image_tensorboard(writer,f,"Image over ADMM iterations (" + net + "output, FULL CONTRAST)",suffix,i,full_contrast=True) # Showing each image with contrast = 1
        
        # Display CRC vs STD curve in tensorboard
        if (i>max_iter - min(max_iter,10)):
            # Creating matplotlib figure
            plt.plot(IR_bkg_recon,CRC_hot_recon,linestyle='None',marker='x')
            plt.xlabel('IR')
            plt.ylabel('CRC')
            # Adding this figure to tensorboard
            writer.flush()
            writer.add_figure('CRC in hot region vs IR in background', plt.gcf(),global_step=i,close=True)
            writer.close()


    """
    Output framework
    """

    # Output of the framework
    x_out = f

    # Saving final image output
    save_img(x_out, subroot+'Images/out_final/final_out' + suffix + '.img')
    '''
    #Plot and save output of the framework
    plt.figure()
    plt.plot(STD_recon,TCR_Recon,'--ro', label='Recon Algorithm')
    plt.title('STD vs CRC')
    plt.legend(loc='lower right')
    plt.xlabel('STD')
    plt.ylabel('CRC')
    plt.savefig(subroot+'Images/metrics/'+format(test)+'/' + format(i) +'-wo-pre.png')
    

    # Display and saved.
    plt.figure()
    plt.imshow(x_out, cmap='gray_r')
    plt.colorbar()
    plt.title('Reconstructed image: %d ' % (i))
    plt.savefig(subroot+'Images/out_final/'+format(test)+'/' + format(test) +'.png')
    '''
    ## Averaging for VAE
    if (net == 'DIP_VAE'):
        print('Computing average over VAE ouputs')
        ## Initialize variables
        # Number of posterior samples to use in mean and variance computation
        n_posterior_samples = min(10,max_iter)
        print("Number of posterior samples :",n_posterior_samples, '(over', max_iter, 'overall iterations)')
        # Averaged and uncertainty images
        x_avg = np.zeros_like(x_out) # averaged image
        x_var = np.zeros_like(x_out) # uncertainty image
        list_samples = [] # list to store averaged and uncertainty images

        # Loading DIP input (we do not have CT-map, so random image created in block 1)
        image_net_input_torch = torch.load(subroot + 'Data/initialization/image_net_input_torch.pt') # DO NOT CREATE RANDOM INPUT IN BLOCK 2 !!! ONLY AT THE BEGINNING, IN BLOCK 1. JUST LOAD IT FROM BLOCK 1

        for i in range(n_posterior_samples):
            # Generate one descaled NN output
            out_descale = generate_nn_output(net, config, image_net_input_torch, PETImage_shape, finetuning, max_iter, test, suffix, subroot)
            list_samples.append(np.squeeze(out_descale))
            
        for i in range(n_posterior_samples):
            x_avg += list_samples[i] / n_posterior_samples
            x_var = (list_samples[i] - x_avg)**2 / n_posterior_samples
        
        # Computing metrics to compare averaging vae outputs with single output
        compute_metrics(x_avg,image_gt,i,PSNR_recon,PSNR_norm_recon,MSE_recon,MA_cold_recon,CRC_hot_recon,CRC_bkg_recon,IR_bkg_recon,bias_cold_recon,bias_hot_recon,write_tensorboard=False)

    # Display images in tensorboard
    write_image_tensorboard(writer,f_init,"initialization of DIP output (f_0)",suffix) # initialization of DIP output in tensorboard
    #write_image_tensorboard(writer,image_init,"initialization of CASToR MAP reconstruction (x_0)") # initialization of CASToR MAP reconstruction in tensorboard
    write_image_tensorboard(writer,image_net_input,"DIP input",suffix) # DIP input in tensorboard
    write_image_tensorboard(writer,image_gt,"Ground Truth",suffix) # Ground truth image in tensorboard

    if (net == 'DIP_VAE'):
        write_image_tensorboard(writer,x_avg,"Final averaged image (average over DIP outputs)",suffix) # Final averaged image in tensorboard
        write_image_tensorboard(writer,x_var,"Uncertainty image (VARIANCE over DIP outputs)",suffix) # Uncertainty image in tensorboard

# Configuration dictionnary for hyperparameters to tune
config = {
    #"lr" : tune.grid_search([0.0001,0.001,0.01]),
    "lr" : tune.grid_search([0.001]),
    #"sub_iter_DIP" : tune.grid_search([10,30,50]),
    #"sub_iter_DIP" : tune.grid_search([10,50,100,200]),
    #"sub_iter_DIP" : tune.grid_search([5,10,20]),
    "sub_iter_DIP" : tune.grid_search([100]),
    #"sub_iter_DIP" : tune.grid_search([50,100,200,500]),
    "sub_iter_MAP" : tune.grid_search([10,20]), # Block 1 iterations (Sub-problem 1 - MAP) if mlem_sequence is False
    "nb_iter_second_admm": tune.grid_search([10]), # Number of ADMM iterations (ADMM before NN)
    "net" : tune.grid_search(['DD','DIP']), # Network to use (DIP,DD,DIP_VAE)
    #"rho" : tune.grid_search([5e-4,3e-3,6e-2,1e-2]),
    "rho" : tune.grid_search([3e-4]),
    #"rho" : tune.grid_search([1e-6]), # Trying to reproduce MLEM result as rho close to 0
    "alpha" : tune.grid_search([0.005,0.05,0.5]),
    "opti_DIP" : tune.grid_search(['Adam']),
    #"opti_DIP" : tune.grid_search(['LBFGS']),
    "mlem_sequence" : tune.grid_search([False]),
    "d_DD" : tune.grid_search([4]), # not below 6, otherwise 128 is too little as output size
    "k_DD" : tune.grid_search([32]),
    "skip_connections" : tune.grid_search([False]),
    "scaling" : tune.grid_search(['standardization','normalization','nothing']),
    "input" : tune.grid_search(['random'])
}
#'''
config = {
    "lr" : tune.grid_search([0.01]), # 0.01 for DIP, 0.001 for DD
    "sub_iter_DIP" : tune.grid_search([100]), # 10 for DIP, 100 for DD
    "sub_iter_MAP" : tune.grid_search([10]), # Block 1 iterations (Sub-problem 1 - MAP) if mlem_sequence is False
    "nb_iter_second_admm": tune.grid_search([10]), # Number of ADMM iterations (ADMM before NN)
    "net" : tune.grid_search(['DIP']), # Network to use (DIP,DD,DIP_VAE)
    "rho" : tune.grid_search([0.0003]),
    "alpha" : tune.grid_search([0.005]),
    "opti_DIP" : tune.grid_search(['Adam']),
    "mlem_sequence" : tune.grid_search([False]),
    "d_DD" : tune.grid_search([4]), # not above 4, otherwise 112 is too little as output size / not above 6, otherwise 128 is too little as output size
    "k_DD" : tune.grid_search([32]),
    "skip_connections" : tune.grid_search([0,1,2,3]),
    "scaling" : tune.grid_search(['standardization']),
    "input" : tune.grid_search(['random'])
}
#'''

## Arguments for linux command to launch script
# Creating arguments
parser = argparse.ArgumentParser(description='DIP + ADMM computation')
parser.add_argument('--proc', type=str, dest='proc', help='processing unit (CPU, GPU or both)')
parser.add_argument('--max_iter', type=int, dest='max_iter', help='number of outer iterations')
parser.add_argument('--finetuning', type=str, dest='finetuning', help='finetuning or not for the DIP optimizations', nargs='?', const='False')

# Retrieving arguments in this python script
args = parser.parse_args()

# For VS Code (without command line)
if (args.proc is None): # Must check if all args are None
    args.proc = 'CPU'
    args.max_iter = 10 # Outer iterations
    args.finetuning = 'last' # Finetuning or not for the DIP optimizations (block 2)
    
config_combination = 1
for i in range(len(config)):
    config_combination *= len(list(list(config.values())[i].values())[0])

if args.proc == 'CPU':
    resources_per_trial = {"cpu": 1, "gpu": 0}
elif args.proc == 'GPU':
    resources_per_trial = {"cpu": 0, "gpu": 0.1} # "gpu": 1 / config_combination
    #resources_per_trial = {"cpu": 0, "gpu": 1} # "gpu": 1 / config_combination
elif args.proc == 'both':
    resources_per_trial = {"cpu": 10, "gpu": 1} # not efficient

#reporter = CLIReporter(
#    parameter_columns=['lr'],
#    metric_columns=['mse'])

# Start tuning of hyperparameters = start each admm computation in parallel
#try: # resume previous run (if it exists)
#    anaysis_raytune = tune.run(partial(admm_loop,args=args,root=os.getcwd()), config=config, local_dir = os.getcwd() + '/runs', name=suffix_func(config) + str(args.max_iter), resources_per_trial = resources_per_trial, resume = "ERRORED_ONLY")#, progress_reporter = reporter)
#except: # do not resume previous run because there is no previous one
#   anaysis_raytune = tune.run(partial(admm_loop,args=args,root=os.getcwd()), config=config, local_dir = os.getcwd() + '/runs', name=suffix_func(config) + "_max_iter=" + str(args.max_iter), resources_per_trial = resources_per_trial)#, progress_reporter = reporter)


anaysis_raytune = tune.run(partial(admm_loop,args=args,root=os.getcwd()), config=config, local_dir = os.getcwd() + '/runs', resources_per_trial = resources_per_trial)#, progress_reporter = reporter)