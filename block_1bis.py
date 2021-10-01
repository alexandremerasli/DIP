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
    finetuning = args.finetuning # Finetuning or not for the DIP optimizations (block 2)

    test = 24  # Label of the experiment
    
    '''
    Creation of directory 
    '''
    suffix =  suffix_func(config) # suffix to make difference between raytune runs (different hyperparameters)
    subroot = root + '/data/Algo/'  # Directory root

    Path(subroot+'Block1/Test_block1/' + suffix + '/out_eq22').mkdir(parents=True, exist_ok=True) # CASTor path
    Path(subroot+'Block1/Test_block1/' + suffix + '/before_eq22').mkdir(parents=True, exist_ok=True) # CASTor path

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
    vb = ' -vb 1'
    th = ' -th 1'
    proj = ' -proj incrementalSiddon'

    opti = ' -opti OPTITR'
    pnlt = ' -pnlt DIP_ADMM'
    pnlt_beta = ' -pnlt-beta ' + str(rho)

    subroot_output_path = ' -dout ' + subroot + 'Block1/Test_block1/' + suffix + '/' # Output path for CASTOR framework
    input_path = ' -img ' + subroot + 'Block1/Test_block1/' + suffix + '/out_eq22/' # Input path for CASTOR framework

    # Command line for calculating the Likelihood
    opti_like = ' -opti-fom'
    opti_like = ''

    castor_command_line = executable + dim + vox + header_file + vb + th + proj + opti + opti_like + pnlt + pnlt_beta

    """
    Initialization : variables
    """

    # ADMM variables
    mu = 0* np.ones((PETImage_shape[0], PETImage_shape[1]), dtype='<f')

    x = np.zeros((PETImage_shape[0], PETImage_shape[1]), dtype='<f') # image x at iteration i
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
    MLEM - CASTOR framework
    """

    ## Loading images (sensitivity, NN input, first DIP output, GT)
    # Load sensitivity image (A_ij)
    image_sens = fijii_np(subroot+'Data/castor_output_it60_ss1_sensitivity.img', shape=(PETImage_shape))

    # Loading DIP input
    # Creating random image input for DIP while we do not have CT, but need to be removed after
    create_random_input(net,PETImage_shape,config) # to be removed when CT will be used instead of random input. DO NOT PUT IT IN BLOCK 2 !!!
    # Loading DIP input (we do not have CT-map, so random image created in block 1)
    image_net_input = load_input(net,PETImage_shape,config) # Normalization of DIP input. DO NOT CREATE RANDOM INPUT IN BLOCK 2 !!! ONLY AT THE BEGINNING, IN BLOCK 1
    # image_net_input_norm, maxe_input = norm_imag(image_net_input) # Normalization of DIP input
    image_net_input_norm,mean_im,std_im = stand_imag(image_net_input) # Standardization of DIP input
    # DIP input image, numpy --> torch
    image_net_input_torch = torch.Tensor(image_net_input_norm)
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
    torch.save(image_net_input_torch,subroot + 'Data/image_net_input_torch.pt')

    # Ininitializeing DIP output and first image x with f_init and image_init
    #image_init_path_without_extension = ''
    #image_init = np.ones((PETImage_shape[0],PETImage_shape[1])) # initializing CASToR MAP reconstruction with uniform image with ones.
    #image_init_path_without_extension = 'Comparaison/BSREM/BSREM_30it_REF'
    image_init_path_without_extension = 'Comparaison/MLEM/MLEM_converge'
    image_init = fijii_np(subroot + image_init_path_without_extension + '.img',shape=(PETImage_shape)) # initializing CASToR MAP reconstruction with BSREM precomputed reference
    #f_init = fijii_np(subroot+'Comparaison/BSREM/BSREM_it30_REF.img',shape=(PETImage_shape))
    f_init = fijii_np(subroot+'Comparaison/MLEM/MLEM_converge.img',shape=(PETImage_shape))

    #Loading Ground Truth image to compute metrics
    image_gt = fijii_np(subroot+'Block2/data/phantom_act.img',shape=(PETImage_shape))

    f = f_init  # Initializing DIP output with f_init

    for i in range(max_iter):
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Outer iteration !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', i)
        start_time_outer_iter = time.time()
        
        # Reconstruction with CASToR (first equation of ADMM)
        x_label = castor_reconstruction(i, castor_command_line, subroot, sub_iter_MAP, test, subroot_output_path, input_path, config, suffix, f, mu, PETImage_shape, image_init_path_without_extension)
        
        # Write image over ADMM iterations
        if ((max_iter>=10) and (i%(max_iter // 10) == 0) or True):

            write_image_tensorboard(writer,x_label,"Corrupted image (x_label) over ADMM iterations",i) # Showing all corrupted images with same contrast to compare them together
            write_image_tensorboard(writer,x_label,"Corrupted image (x_label) over ADMM iterations (FULL CONTRAST)",i,full_contrast=True) # Showing each corrupted image with contrast = 1
        
        # Block 2 - CNN - 10 iterations
        start_time_block2= time.time()
        successful_process = subprocess.call(["python3", root+"/block_2_bis_lightning.py", str(i), str(test), net, processing_unit, finetuning, PETImage_shape_str, root, suffix]) #Calling block 2 algorithm and passing variables (current iter-number of epochs and test number, chosen net, processing unit, way to do finetuning, and image dimensions)
        if successful_process != 0: # if there is an error in block2, then stop the run
            raise ValueError('An error occured in block2 computation. Stopping overall iterations.')
        print("--- %s seconds - DIP block ---" % (time.time() - start_time_block2))
        f = fijii_np(subroot+'Block2/out_cnn/'+ format(test)+'/out_' + net + '' + format(i) + suffix + '.img',shape=(PETImage_shape)) # loading DIP output

        # Metrics for NN output
        compute_metrics(f,image_gt,i,PSNR_recon,PSNR_norm_recon,MSE_recon,MA_cold_recon,CRC_hot_recon,CRC_bkg_recon,IR_bkg_recon,bias_cold_recon,bias_hot_recon,writer=writer,write_tensorboard=True)

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

        '''
        # Saving DIP output for each ADMM iteration as video in tensorboard
        import imageio
        frames = [fijii_np(subroot+'Block2/out_cnn/'+ format(test)+'/out_' + net + '' + format(p) + suffix + '.img',shape=(PETImage_shape)) for p in range(i+1)] # loading DIP outputs
        imageio.mimsave(subroot+'output.gif', frames, fps=(max_iter / 1.0))
        '''

        # Write image over ADMM iterations
        if ((max_iter>=10) and (i%(max_iter // 10) == 0)):
            write_image_tensorboard(writer,f,"Image over ADMM iterations (" + net + "output)",i) # Showing all images with same contrast to compare them together
            write_image_tensorboard(writer,f,"Image over ADMM iterations (" + net + "output, FULL CONTRAST)",i,full_contrast=True) # Showing each image with contrast = 1
        
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
    #Plot and save output of the famework
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
        image_net_input_torch = torch.load(subroot + 'Data/image_net_input_torch.pt') # DO NOT CREATE RANDOM INPUT IN BLOCK 2 !!! ONLY AT THE BEGINNING, IN BLOCK 1. JUST LOAD IT FROM BLOCK 1

        for i in range(n_posterior_samples):
            # Generate one destandardized NN output
            out_destand = generate_nn_output(net, config, image_net_input_torch, PETImage_shape, finetuning, max_iter, test, suffix, subroot)
            list_samples.append(np.squeeze(out_destand))
            
        for i in range(n_posterior_samples):
            x_avg += list_samples[i] / n_posterior_samples
            x_var = (list_samples[i] - x_avg)**2 / n_posterior_samples
        
        # Computing metrics to compare averaging vae outputs with single output
        compute_metrics(x_avg,image_gt,i,PSNR_recon,PSNR_norm_recon,MSE_recon,MA_cold_recon,CRC_hot_recon,CRC_bkg_recon,IR_bkg_recon,bias_cold_recon,bias_hot_recon,write_tensorboard=False)

    # Display images in tensorboard
    write_image_tensorboard(writer,f_init,"initialization of DIP output (f_0)") # initialization of DIP output in tensorboard
    write_image_tensorboard(writer,image_init,"initialization of CASToR MAP reconstruction (x_0)") # initialization of CASToR MAP reconstruction in tensorboard
    write_image_tensorboard(writer,image_net_input,"DIP input") # DIP input in tensorboard
    write_image_tensorboard(writer,image_gt,"Ground Truth") # Ground truth image in tensorboard

    if (net == 'DIP_VAE'):
        write_image_tensorboard(writer,x_avg,"Final averaged image (average over DIP outputs)") # Final averaged image in tensorboard
        write_image_tensorboard(writer,x_var,"Uncertainty image (VARIANCE over DIP outputs)") # Uncertainty image in tensorboard

# Configuration dictionnary for hyperparameters to tune
config = {
    "lr" : tune.grid_search([0.0001,0.001,0.01]),
    #"sub_iter_DIP" : tune.grid_search([10,30,50]),
    "sub_iter_DIP" : tune.grid_search([10,50,100,200]),
    #"sub_iter_DIP" : tune.grid_search([50,100,200,500]),
    #"rho" : tune.grid_search([5e-4,3e-3,6e-2,1e-2]),
    "rho" : tune.grid_search([3e-3]),
    #"rho" : tune.grid_search([1e-6]), # Trying to reproduce MLEM result as rho close to 0
    "opti_DIP" : tune.grid_search(['Adam']),
    #"opti_DIP" : tune.grid_search(['LBFGS']),
    "mlem_sequence" : tune.grid_search([False]),
    "d_DD" : tune.grid_search([6]), # not below 6, otherwise 128 is too little as output size
    "k_DD" : tune.grid_search([32]),
    "skip_connections" : tune.grid_search([False])
}
#'''
config = {
    "lr" : tune.grid_search([0.001]),
    "sub_iter_DIP" : tune.grid_search([100]),
    "rho" : tune.grid_search([0.003]),
    "opti_DIP" : tune.grid_search(['Adam']),
    "mlem_sequence" : tune.grid_search([False]),
    "d_DD" : tune.grid_search([6]), # not below 6, otherwise 128 is too little as output size
    "k_DD" : tune.grid_search([32]),
    "skip_connections" : tune.grid_search([True])
}
#'''

## Arguments for linux command to launch script
# Creating arguments
parser = argparse.ArgumentParser(description='DIP + ADMM computation')
parser.add_argument('--net', type=str, dest='net', help='network to use (DIP,DD,DIP_VAE)')
parser.add_argument('--proc', type=str, dest='proc', help='processing unit (CPU, GPU or both)')
parser.add_argument('--max_iter', type=int, dest='max_iter', help='number of outer iterations')
parser.add_argument('--sub_iter_MAP', type=int, dest='sub_iter_MAP', help='number of block 1 iterations (Sub-problem 1 - MAP)', nargs='?', const=1)
parser.add_argument('--finetuning', type=str, dest='finetuning', help='finetuning or not for the DIP optimizations', nargs='?', const='False')

# Retrieving arguments in this python script
args = parser.parse_args()


# For VS Code (without command line)
if (args.net is None): # Must check if all args are None
    args.net = 'DD' # Network architecture
    args.proc = 'CPU'
    args.max_iter = 10 # Outer iterations
    args.sub_iter_MAP = 2 # Block 1 iterations (Sub-problem 1 - MAP) if mlem_sequence is False
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
