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
image_init : initialization of image x = initialization of DIP output
"""

## Python libraries

# Pytorch
import numpy
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
    net = args.net # Network architecture
    processing_unit = args.proc
    max_iter = args.max_iter # Outer iterations
    if (args.sub_iter_MAP is None): # Block 1 iterations (Sub-problem 1 - MAP)
        sub_iter_MAP = 2 # no argument given in command line
    else:
        sub_iter_MAP = args.sub_iter_MAP  
    sub_iter_DIP = args.sub_iter_DIP # Block 2 iterations (Sub-problem 2 - DIP)
    finetuning = args.finetuning # Finetuning or not for the DIP optimizations (block 2)

    # For VS Code (without command line)
    if (args.net is None): # Must check if all args is None
        net = 'DIP_VAE' # Network architecture
        processing_unit = 'CPU'
        max_iter = 30 # Outer iterations
        sub_iter_MAP = 2 # Block 1 iterations (Sub-problem 1 - MAP)
        sub_iter_DIP = 50 # Block 2 iterations (Sub-problem 2 - DIP)
        finetuning = 'last' # Finetuning or not for the DIP optimizations (block 2)

    test = 24  # Label of the experiment
    
    '''
    Creation of directory 
    '''
    suffix =  suffix_func(config) # suffix to make difference between raytune runs (different hyperparameters)
    subroot = root + '/data/Algo/'  # Directory root

    Path(subroot+'Block1/Test_block1/' + suffix + '/out_eq22').mkdir(parents=True, exist_ok=True) # CASTor path

    Path(subroot+'Images/eq_22/'+format(test)+'/').mkdir(parents=True, exist_ok=True) # First subproblem - folder
    Path(subroot+'Images/image_EM/'+format(test)+'/').mkdir(parents=True, exist_ok=True) # MLEM image - folder
    Path(subroot+'Images/metrics/'+format(test)+'/').mkdir(parents=True, exist_ok=True) # Block 1 metrics - folder
    Path(subroot+'Images/iter_in_DIP/'+format(test)+'/').mkdir(parents=True, exist_ok=True) # Images of every iteration in DIP block - folder
    Path(subroot+'Images/out_cnn/'+format(test)+'/').mkdir(parents=True, exist_ok=True) # Output of DIP every n epochs - folder
    Path(subroot+'Images/out_final/'+format(test)+'/').mkdir(parents=True, exist_ok=True) # Output of the framework (Last output of the DIP)
    Path(subroot+'Images/x_label/'+ format(test)+'/').mkdir(parents=True, exist_ok=True) # Folder for every updated corrupted image

    Path(subroot+'Block2/checkpoint/'+format(test)+'/').mkdir(parents=True, exist_ok=True)
    Path(subroot+'Block2/out_cnn/'+ format(test)+'/').mkdir(parents=True, exist_ok=True) # Output of the DIP block every outer iteration
    Path(subroot+'Block2/out_cnn/vae').mkdir(parents=True, exist_ok=True) # Output of the DIP block every outer iteration
    Path(subroot+'Block2/out_cnn/cnn_metrics/'+ format(test)+'/').mkdir(parents=True, exist_ok=True) # DIP block metrics
    Path(subroot+'Block2/out_cnn/'+ format(test)+'/like/').mkdir(parents=True, exist_ok=True) # folder for Likelihood calculation (using CASTOR)
    Path(subroot+'Images/mu_diff/' + format(test) + '/').mkdir(parents=True, exist_ok=True) # Here are saved the difference between x corrupt (x_label) and recontructed x every outer iterations
    Path(subroot+'Block2/x_label/'+format(test) + '/').mkdir(parents=True, exist_ok=True) # x corrupted - folder
    Path(subroot+'Block2/data/'+format(test) + '/').mkdir(parents=True, exist_ok=True) # ground truth

    Path(subroot+'Images/uncertainty/'+format(test)+'/').mkdir(parents=True, exist_ok=True) # Directory where all the samples are saved
    Path(subroot+'Block2/checkpoint/'+format(test)+'/').mkdir(parents=True, exist_ok=True)
    Path(subroot+'Block2/out_cnn/'+ format(test)+'/').mkdir(parents=True, exist_ok=True)
    Path(subroot+'Block2/out_cnn/cnn_metrics/'+ format(test)+'/').mkdir(parents=True, exist_ok=True)
    Path(subroot+'Block2/data/x_label_DIP/'+format(test) + '/').mkdir(parents=True, exist_ok=True) # Directory where all labels for DIP stage are saved

    Path(subroot+'Comparaison/MLEM/').mkdir(parents=True, exist_ok=True) # CASTor path
    Path(subroot+'Comparaison/BSREM/').mkdir(parents=True, exist_ok=True) # CASTor path

    Path(subroot+'Config/').mkdir(parents=True, exist_ok=True) # CASTor path

    # Define PET input dimensions according to input data dimensions
    PETImage_shape_str = read_input_dim()
    PETImage_shape = input_dim_str_to_list(PETImage_shape_str)

    """
    Initialization : comand line for ML-EM using CASTOR framework
    """

    # Config dictionnary for hyperparameters
    rho = config["rho"]
    np.save(subroot + 'Config/config' + suffix + '.npy', config) # Save this configuration of hyperparameters, and reload it at the beginning of block 2 thanks to suffix (passed in subprocess call argumentsZ)

    # castor-recon command line
    header_file = ' -df ' + subroot + 'Data/data_eff10/data_eff10.cdh' # PET data path

    executable = 'castor-recon'
    dim = ' -dim ' + PETImage_shape_str
    vox = ' -vox 4,4,4'
    vb = ' -vb 3'
    th = ' -th 0'
    proj = ' -proj incrementalSiddon'
    opti = ' -opti MLEM'
    subroot_output_path = ' -dout ' + subroot + 'Block1/Test_block1/' + suffix + '/' # Output path for CASTOR framework
    input_path = ' -img ' + subroot + 'Block1/Test_block1/' + suffix + '/out_eq22/' # Input path for CASTOR framework

    # Command line for calculating the Likelihood
    vb_like = ' -vb 0'
    opti_like = ' -opti-fom'

    castor_command_line = executable + dim + vox + header_file + vb + th + proj + opti + opti_like

    """
    Initialization : variables
    """

    # ADMM variables
    mu = 0* np.ones((PETImage_shape[0], PETImage_shape[1]), dtype='<f')
    # rho = rho * np.ones((PETImage_shape[0], PETImage_shape[1]), dtype='<f')

    x = np.zeros((PETImage_shape[0], PETImage_shape[1]), dtype='<f') # image x at iteration i
    x_out = np.zeros((PETImage_shape[0], PETImage_shape[1]), dtype='<f') # output of DIP

    #writer = SummaryWriter(comment='-%s-%s-maxIter_%s' % (net,finetuning, max_iter ))
    writer = SummaryWriter()

    """
    MLEM - CASTOR framework
    """

    ## Loading images (sensitivity, NN input, first DIP output, GT)
    # Load sensitivity image (A_ij)
    image_sens = fijii_np(subroot+'Data/castor_output_it60_ss1_sensitivity.img', shape=(PETImage_shape))

    # Loading DIP input
    # Creating random image input for DIP while we do not have CT, but need to be removed after
    create_random_input(net,PETImage_shape) # to be removed when CT will be used instead of random input. DO NOT PUT IT IN BLOCK 2 !!!
    # Loading DIP input (we do not have CT-map, so random image created in block 1)
    image_net_input = load_input(net,PETImage_shape) # Normalization of DIP input. DO NOT CREATE RANDOM INPUT IN BLOCK 2 !!! ONLY AT THE BEGINNING, IN BLOCK 1
    # image_net_input_norm, maxe_input = norm_imag(image_net_input) # Normalization of DIP input
    image_net_input_norm,mean_im,std_im = stand_imag(image_net_input) # Standardization of DIP input
    # DIP input image, numpy --> torch
    image_net_input_torch = torch.Tensor(image_net_input_norm)
    # Adding dimensions to fit network architecture
    if (net == 'DIP' or net == 'DIP_VAE'):
        image_net_input_torch = image_net_input_torch.view(1,1,PETImage_shape[0],PETImage_shape[1]) # For DIP
    else:
        image_net_input_torch = image_net_input_torch.view(1,32,16,16) # For Deep Decoder
    torch.save(image_net_input_torch,subroot + 'Data/image_net_input_torch.pt')

    # Ininitializeing DIP output and first image x with image_init = DIP output after fitting a MLEM image
    image_init = np.ones((PETImage_shape[0],PETImage_shape[1])) # initializing DIP output with uniform image with ones.

    '''
    # Save config_init with different sub_iter_DIP just for this call
    config_init = config
    config_init["sub_iter_DIP"] = 100
    suffix = '_init' + suffix_func(config_init)
    np.save(root + '/config' + suffix + '.npy', config_init) # Save this configuration of hyperparameters, and reload it at the beginning of block 2 thanks to suffix (passed in subprocess call argumentsZ)

    # Reading and saving (= copying) DIP x_label (corrupted image) as MLEM image
    MLEM_label = fijii_np(subroot+'Data/castor_output_it60.img',shape=(PETImage_shape))
    i = -1 # before true iterations
    save_img(MLEM_label, subroot+'Block2/x_label/' + format(test)+'/'+ format(i) +'_x_label' + suffix + '.img')
    # DIP fitting to obtain first image x = first DIP output
    subprocess.call(["python3", root+"/block_2_bis_lightning.py", str(i), str(test), 'DIP', processing_unit, 'False', PETImage_shape_str, root, suffix]) #Calling block 2 algorithm and passing variables (current iter-number of epochs and test number, chosen net, processing unit, way to do finetuning, and image dimensions)
    image_init = fijii_np(subroot+'Block2/out_cnn/'+ format(test)+'/out_DIP' + format(i) + suffix + '.img',shape=(PETImage_shape)) # loading DIP output
    
    import sys
    sys.exit()
    # renommer pour la suite initialimage
    #'''

    #Loading Ground Truth image to compute metrics
    image_gt = fijii_np(subroot+'Block2/data/phantom_act.img',shape=(PETImage_shape))
    # Normalization of GT
    # image_gt_norm_np, maxe_gt_input_norm = norm_imag(image_gt) # Normalization of GT
    image_gt_norm,mean_gt,std_gt= stand_imag(image_gt) # Standardization of GT

    f = image_init  # Initializing DIP output with image_init

    for i in range(max_iter):
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Outer iteration !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', i)
        start_time_outer_iter = time.time()
        
        # Reconstruction with CASToR (first equation of ADMM)
        x_label = castor_reconstruction(i, castor_command_line, subroot, sub_iter_MAP, test, subroot_output_path, input_path, config, suffix, image_sens, rho, f, mu, PETImage_shape)

        # Block 2 - CNN - 10 iterations
        start_time_block2= time.time()
        subprocess.call(["python3", root+"/block_2_bis_lightning.py", str(i), str(test), net, processing_unit, finetuning, PETImage_shape_str, root, suffix]) #Calling block 2 algorithm and passing variables (current iter-number of epochs and test number, chosen net, processing unit, way to do finetuning, and image dimensions)
        print("--- %s seconds - DIP block ---" % (time.time() - start_time_block2))
        f = fijii_np(subroot+'Block2/out_cnn/'+ format(test)+'/out_DIP' + format(i) + suffix + '.img',shape=(PETImage_shape)) # loading DIP output
        f[f<0] = 0
        # Metrics for NN output
        compute_metrics(f,image_gt,i,max_iter,writer=writer,write_tensorboard=True)

        # Block 3 - equation 15 - mu
        
        mu = x_label- f

        """
        plt.figure()
        plt.imshow(mu, cmap='gray_r')
        plt.colorbar()
        plt.title(r'mu difference ')
        plt.savefig(subroot+'Images/mu_diff/' + format(test) + '/' + format(i) + '.png')
        """

        print("--- %s seconds - outer_iteration ---" % (time.time() - start_time_outer_iter))

        '''
        # Loading DIP input (we do not have CT-map, so random image created in block 1)
        image_net_input_torch = torch.load(subroot + 'Data/image_net_input_torch.pt') # DO NOT CREATE RANDOM INPUT IN BLOCK 2 !!! ONLY AT THE BEGINNING, IN BLOCK 1. JUST LOAD IT FROM BLOCK 1
        model, model_class = choose_net(net, config)
        writer.flush()
        writer.add_graph(model, image_net_input_torch)
        writer.close()
        '''

    """
    Output framework
    """

    # Output of the framework
    x_out = f

    # Saving final image output
    save_img(x_out, subroot+'Images/out_final/final_out' + suffix + '.img')
    '''
    #Plot and save output of the famework
    plt.figure()
    plt.plot(STD_recon,TCR_Recon,'--ro', label='Recon Algorithm')
    plt.title('STD vs CRC')
    plt.legend(loc='lower right')
    plt.xlabel('STD')
    plt.ylabel('CRC')
    plt.savefig(subroot+'Images/metrics/'+format(test)+'/' + format(i) +'-wo-pre.png')
    '''

    # Display and saved.
    plt.figure()
    plt.imshow(x_out, cmap='gray_r')
    plt.colorbar()
    plt.title('Reconstructed image: %d ' % (i))
    plt.savefig(subroot+'Images/out_final/'+format(test)+'/' + format(test) +'.png')

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
        image_net_input_torch = torch.load(subroot + 'Data/image_net_input_torch.pt') # DO NOT CREATE RANDOM INPUT IN BLOCK 2 !!! ONLY AT THE BEGINNING, IN BLOCK 1. JUST LOAD IT FROM BLOCK 1

        for i in range(n_posterior_samples):
            # Generate one destandardized NN output
            out_destand = generate_nn_output(net, config, image_net_input_torch, PETImage_shape, finetuning, max_iter, test, suffix)
            list_samples.append(np.squeeze(out_destand))
            
        for i in range(n_posterior_samples):
            x_avg += list_samples[i] / n_posterior_samples
            x_var = (list_samples[i] - x_avg)**2 / n_posterior_samples
        
        # Computing metrics to compare averaging vae outputs with single output
        compute_metrics(x_avg,image_gt,i,max_iter,write_tensorboard=False)

    # Display images in tensorboard
    write_image_tensorboard(writer,image_init,"initialization of DIP output") # DIP input in tensorboard
    write_image_tensorboard(writer,image_net_input,"DIP input") # Initialization of DIP output in tensorboard
    write_image_tensorboard(writer,image_gt,"Ground Truth") # Ground truth image in tensorboard
    write_image_tensorboard(writer,x_out,"Final image (DIP output)") # DIP output in tensorboard

    if (net == 'DIP_VAE'):
        write_image_tensorboard(writer,x_avg,"Final averaged image (average over DIP outputs)") # Final averaged image in tensorboard
        write_image_tensorboard(writer,x_var,"Uncertainty image (VARIANCE over DIP outputs)") # Uncertainty image in tensorboard


## Arguments for linux command to launch script
# Creating arguments
parser = argparse.ArgumentParser(description='DIP + ADMM computation')
parser.add_argument('--net', type=str, dest='net', help='network to use (DIP,DD,DIP_VAE)')
parser.add_argument('--proc', type=str, dest='proc', help='processing unit (CPU or GPU)')
parser.add_argument('--max_iter', type=int, dest='max_iter', help='number of outer iterations')
parser.add_argument('--sub_iter_DIP', type=int, dest='sub_iter_DIP', help='number of block 2 iterations (Sub-problem 2 - DIP)')
parser.add_argument('--sub_iter_MAP', type=int, dest='sub_iter_MAP', help='number of block 1 iterations (Sub-problem 1 - MAP)', nargs='?', const=1)
parser.add_argument('--finetuning', type=str, dest='finetuning', help='finetuning or not for the DIP optimizations', nargs='?', const='False')

# Retrieving arguments in this python script
args = parser.parse_args()

# Configuration dictionnary for hyperparameters to tune
config = {
    "lr" : tune.grid_search([0.0001,0.001,0.01]),
    "sub_iter_DIP" : tune.grid_search([10,30,50]),
    "rho" : tune.grid_search([5e-4,3e-3,6e-2,1e-2]),
    "opti_DIP" : tune.grid_search(['Adam']),
    #"opti_DIP" : tune.grid_search(['LBFGS']),
    "mlem_subsets" : tune.grid_search([False,True])
}
'''
config = {
    "lr" : tune.grid_search([0.0001]),
    "sub_iter_DIP" : tune.grid_search([10]),
    "rho" : tune.grid_search([3e-3]),
    "opti_DIP" : tune.grid_search(['Adam']),
    "mlem_subsets" : tune.grid_search([False])
}
'''

#reporter = CLIReporter(
#    parameter_columns=['lr'],
#    metric_columns=['mse'])

# Start tuning of hyperparameters = start each admm computation in parallel
try: # resume previous run (if it exists)
    anaysis_raytune = tune.run(partial(admm_loop,args=args,root=os.getcwd()), config=config, local_dir = os.getcwd() + '/runs', resume = "ERRORED_ONLY")#, resources_per_trial = {'gpu' : 1}, progress_reporter = reporter)
except: # do not resume previous run because there is no previous one
    anaysis_raytune = tune.run(partial(admm_loop,args=args,root=os.getcwd()), config=config, local_dir = os.getcwd() + '/runs')#, resources_per_trial = {'gpu' : 1}, progress_reporter = reporter)