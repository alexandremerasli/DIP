"""
Import libraries
"""

"""
Variables :
image_gt : ground truth image
image_net_input : DIP input
image_corrupt : DIP label
x : image x at iteration i
x_init : initial image for MLEM reconstruction
variable_norm : normalized (or standardized) variable
variable_torch : torch representation of variable

"""

## Python libraries

# Useful
import os
import sys
import warnings
from datetime import datetime
from functools import partial
from ray import tune

# Math
import numpy as np

# Pytorch
import torch
from torchsummary import summary

# Local files to import
from utils_func import *

def post_reconstruction(config,root):
    admm_it = 1 # Set it to 1, 0 is for ADMM reconstruction with hard coded values
    test = 24 # Label of the experiment
    finetuning = 'False' # Finetuning (with best model or last saved model for checkpoint) or not for the DIP optimizations
    # PETImage_shape_str = sys.argv[6] # PET input dimensions (string, tF)
    # suffix =  'TEST_post_reconstruction_' + suffix_func(config) # suffix to make difference between raytune runs (different hyperparameters)

    subroot = root+'/data/Algo/'

    # np.save(subroot + 'Config/config' + suffix + '.npy', config) # Save this configuration of hyperparameters, and reload it at the beginning of block 2 thanks to suffix (passed in subprocess call argumentsZ)

    # config = np.load(subroot + 'Config/config' + suffix + '.npy',allow_pickle='TRUE').item()
    # Config dictionnary for hyperparameters
    # lr = config["lr"]
    sub_iter_DIP = config["sub_iter_DIP"]
    # rho = config["rho"]
    # opti_DIP = config["opti_DIP"]

    # Define PET input dimensions according to input data dimensions
    PETImage_shape_str = read_input_dim()
    PETImage_shape = input_dim_str_to_list(PETImage_shape_str)

    # Metrics arrays
    PSNR_recon = np.zeros(sub_iter_DIP)
    PSNR_norm_recon = np.zeros(sub_iter_DIP)
    MSE_recon = np.zeros(sub_iter_DIP)
    MA_cold_recon = np.zeros(sub_iter_DIP)
    CRC_hot_recon = np.zeros(sub_iter_DIP)
    CRC_bkg_recon = np.zeros(sub_iter_DIP)
    IR_bkg_recon = np.zeros(sub_iter_DIP)
    bias_cold_recon = np.zeros(sub_iter_DIP)
    bias_hot_recon = np.zeros(sub_iter_DIP)

    #Loading Ground Truth image to compute metrics
    image_gt = fijii_np(subroot+'Block2/data/phantom_act.img',shape=(PETImage_shape))

    ## Loading RAW stack of images
    # Loading DIP input (we do not have CT-map, so random image created in block 1)
    # Creating random image input for DIP while we do not have CT, but need to be removed after
    create_random_input(net,PETImage_shape,config) # to be removed when CT will be used instead of random input. Here we CAN create random input as this script is only run once
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

    image_net_input_torch = torch.load(subroot + 'Data/image_net_input_torch.pt')# Here we CAN create random input as this script is only run once

    # Loading DIP x_label (corrupted image) from block1
    image_corrupt = fijii_np(subroot+'Comparison/MLEM/MLEM_converge_avec_post_filtre.img',shape=(PETImage_shape))
    #image_corrupt = fijii_np(subroot+'Comparison/MLEM/MLEM_converge_sans_post_filtre.img',shape=(PETImage_shape)) # Does not "denoise" so well
    #image_corrupt = fijii_np(subroot+'Comparison/BSREM/BSREM_it30_REF.img',shape=(PETImage_shape))
    #image_corrupt = fijii_np(subroot+'Comparison/MLEM/MLEM_it2.img',shape=(PETImage_shape))

    # Normalization of x_label image
    # image_corrupt_norm_scale, maxe = norm_imag(image_corrupt) # Normalization of x_label image
    image_corrupt_norm,mean_label,std_label= stand_imag(image_corrupt) # Standardization of x_label image

    ## Transforming numpy variables to torch tensors


    # Corrupted image x_label, numpy --> torch
    image_corrupt_torch = torch.Tensor(image_corrupt_norm)
    # Adding dimensions to fit network architecture
    image_corrupt_torch = image_corrupt_torch.view(1,1,PETImage_shape[0],PETImage_shape[1])

    def train_process(config, finetuning, processing_unit, sub_iter_DIP, admm_it, image_net_input_torch, image_corrupt_torch):
        # Implements Dataset
        train_dataset = torch.utils.data.TensorDataset(image_net_input_torch, image_corrupt_torch)
        # train_dataset = Dataset(image_net_input_torch, image_corrupt_torch)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1) # num_workers is 0 by default, which means the training process will work sequentially inside the main process

        # Choose network architecture as model
        model, model_class = choose_net(net, config)

        checkpoint_simple_path = 'runs/' # To log loss in tensorboard thanks to Logger
        checkpoint_simple_path_exp = '' # We do not need its value in this script
        
        # Start training
        print('Starting optimization, iteration',admm_it)
        trainer = create_pl_trainer(finetuning, processing_unit, sub_iter_DIP, admm_it, net, checkpoint_simple_path, test, checkpoint_simple_path_exp,name=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        trainer.fit(model, train_dataloader)

        return model

    model = train_process(config, finetuning, processing_unit, sub_iter_DIP, admm_it, image_net_input_torch, image_corrupt_torch)

    """
    Saving variables and model
    """

    writer = model.logger.experiment # Assess to new variable, otherwise error : weakly-referenced object ...
    write_image_tensorboard(writer,image_corrupt,"Corrupted image to fit") # Showing corrupted image
    write_image_tensorboard(writer,image_corrupt,"Corrupted image to fit (FULL CONTRAST)",full_contrast=True) # Showing corrupted image with contrast = 1
    for epoch in range(0,sub_iter_DIP,sub_iter_DIP//10):
        # Load saved (STANDARDIZED) images
        net_outputs_path = subroot+'Block2/out_cnn/' + format(test) + '/out_' + net + '_post_reco_epoch=' + format(epoch) + suffix_func(config) + '.img'
        out = fijii_np(net_outputs_path,shape=(PETImage_shape))
        # Destandardize like at the beginning
        out_destand = destand_numpy_imag(out, mean_label, std_label)
        # Metrics for NN output
        compute_metrics(out_destand,image_gt,epoch,PSNR_recon,PSNR_norm_recon,MSE_recon,MA_cold_recon,CRC_hot_recon,CRC_bkg_recon,IR_bkg_recon,bias_cold_recon,bias_hot_recon,writer=writer,write_tensorboard=True)
        # Saving (now DESTANDARDIZED) image output
        save_img(out_destand, net_outputs_path)
        # Write images over epochs
        write_image_tensorboard(writer,out_destand,"Image over epochs (" + net + "output)",epoch) # Showing all images with same contrast to compare them together
        write_image_tensorboard(writer,out_destand,"Image over epochs (" + net + "output, FULL CONTRAST)",epoch,full_contrast=True) # Showing each image with contrast = 1
        writer.close()

    print('Finish')

"""
Receiving variables from block 1 part and initializing variables
"""

net = 'DD' # Network architecture
processing_unit = 'GPU' # Processing unit (CPU or GPU)

if (net=='DIP'):
    # Configuration dictionnary for hyperparameters to tune
    config = {
        "lr" : 0.01,
        "sub_iter_DIP" : 2000,
        "rho" : 0.003,
        "opti_DIP" : 'Adam',
        "mlem_sequence" : None, # None means we are in post reconstruction mode
        "d_DD" : 6,
        "k_DD" : 32,
        "skip_connections" : False
    }
elif (net=='DD'):
    # Configuration dictionnary for hyperparameters to tune
    config = {
        "lr" : 0.001,
        "sub_iter_DIP" : 500,
        "rho" : 0.003,
        "opti_DIP" : 'Adam',
        "mlem_sequence" : None, # None means we are in post reconstruction mode
        "d_DD" : 6,
        "k_DD" : 32,
        "skip_connections" : False
    }
elif (net=='DD_AE'):
# Configuration dictionnary for hyperparameters to tune
    config = {
        "lr" : 0.001,
        "sub_iter_DIP" : 3000,
        "rho" : 0.003,
        "opti_DIP" : 'Adam',
        "mlem_sequence" : None, # None means we are in post reconstruction mode
        "d_DD" : 6,
        "k_DD" : 32,
        "skip_connections" : False
    }

# Configuration dictionnary for hyperparameters to tune
config = {
    "lr" : tune.grid_search([0.0001,0.001,0.01]),
    #"sub_iter_DIP" : tune.grid_search([10,30,50]),
    "sub_iter_DIP" : tune.grid_search([2000]),
    #"rho" : tune.grid_search([5e-4,3e-3,6e-2,1e-2]),
    "rho" : tune.grid_search([3e-3]),
    #"rho" : tune.grid_search([1e-6]), # Trying to reproduce MLEM result as rho close to 0
    "opti_DIP" : tune.grid_search(['Adam']),
    #"opti_DIP" : tune.grid_search(['LBFGS']),
    "mlem_sequence" : tune.grid_search([None]), # None means post reconstruction mode
    "d_DD" : tune.grid_search([6]), # not below 6, otherwise 128 is too little as output size
    "k_DD" : tune.grid_search([32]),
    "skip_connections" : tune.grid_search([True])
}
'''
config = {
    "lr" : tune.grid_search([0.001]),
    "sub_iter_DIP" : tune.grid_search([200]),
    "rho" : tune.grid_search([0.003]),
    "opti_DIP" : tune.grid_search(['Adam']),
    "mlem_sequence" : tune.grid_search([None]), # None means we are in post reconstruction mode
    "d_DD" : tune.grid_search([6]), # not below 6, otherwise 128 is too little as output size
    "k_DD" : tune.grid_search([32]),
    "skip_connections" : tune.grid_search([False])
}
'''

if processing_unit == 'CPU':
    resources_per_trial = {"cpu": 1, "gpu": 0}
elif processing_unit == 'GPU':
    resources_per_trial = {"cpu": 0, "gpu": 0.1} # "gpu": 1 / config_combination
    #resources_per_trial = {"cpu": 0, "gpu": 1} # "gpu": 1 / config_combination
elif processing_unit == 'both':
    resources_per_trial = {"cpu": 10, "gpu": 1} # not efficient

anaysis_raytune = tune.run(partial(post_reconstruction,root=os.getcwd()), config=config, local_dir = os.getcwd() + '/runs', resources_per_trial = resources_per_trial)#, progress_reporter = reporter)