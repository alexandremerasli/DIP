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
    finetuning = 'last' # Finetuning (with last saved model for checkpoint) or not for the DIP optimization
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
    scaling_input = config["scaling"]

    # Define PET input dimensions according to input data dimensions
    PETImage_shape_str = read_input_dim(subroot+'Data/castor_output_it60.hdr')
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
    image_gt = fijii_np(subroot+'Data/phantom/phantom_act.img',shape=(PETImage_shape))


    ## Loading RAW stack of images
    # Loading DIP input (we do not have CT-map, so random image created in block 1)
    # Creating random image input for DIP while we do not have CT, but need to be removed after
    create_random_input(net,PETImage_shape,config) # to be removed when CT will be used instead of random input. Here we CAN create random input as this script is only run once
    image_net_input = load_input(net,PETImage_shape,config) # Normalization of DIP input. DO NOT CREATE RANDOM INPUT IN BLOCK 2 !!! ONLY AT THE BEGINNING, IN BLOCK 1
    # image_net_input_norm, maxe_input = norm_imag(image_net_input) # Normalization of DIP input
    image_net_input_norm,mean_im,std_im = rescale_imag(image_net_input,scaling_input) # Scaling of DIP input
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
    torch.save(image_net_input_torch,subroot + 'Data/initialization/image_net_input_torch.pt')

    image_net_input_torch = torch.load(subroot + 'Data/initialization/image_net_input_torch.pt')# Here we CAN create random input as this script is only run once
    
    # Loading DIP x_label (corrupted image) from block1
    image_corrupt = fijii_np(subroot+'Comparison/im_corrupt_beginning.img',shape=(PETImage_shape))
    #image_corrupt = fijii_np(subroot+'Comparison/MLEM/MLEM_converge_sans_post_filtre.img',shape=(PETImage_shape)) # Does not "denoise" so well
    #image_corrupt = fijii_np(subroot+'Data/initialization/BSREM_it30_REF_cropped.img',shape=(PETImage_shape))
    #image_corrupt = fijii_np(subroot+'Comparison/MLEM/MLEM_it2.img',shape=(PETImage_shape))

    # Normalization of x_label image
    # image_corrupt_norm_scale, maxe = norm_imag(image_corrupt) # Normalization of x_label image
    image_corrupt_norm,mean_label,std_label= rescale_imag(image_corrupt) # Scaling of x_label image

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
        checkpoint_simple_path_exp = subroot+'Block2/checkpoint/'+format(test)  + '/' + suffix_func(config) + '/'

        model = load_model(image_net_input_torch, config, finetuning, admm_it, model, model_class, subroot, checkpoint_simple_path_exp, training=True)

        # Start training
        print('Starting optimization, iteration',admm_it)
        trainer = create_pl_trainer(finetuning, processing_unit, sub_iter_DIP, admm_it, net, checkpoint_simple_path, test, checkpoint_simple_path_exp,name=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        trainer.fit(model, train_dataloader)

        return model

    # Initializing first output
    os.system('rm -rf ' + subroot+'Block2/checkpoint/'+format(test)  + '/' + suffix_func(config) + '/' + '/last.ckpt') # Otherwise, pl will use checkpoint from other run
    #sys.exit()
    model = train_process(config, finetuning, processing_unit, 1, -1, image_net_input_torch, image_corrupt_torch) # Not useful to make iterations, we just want to initialize writer. admm_it must be set to -1, otherwise seeking for a checkpoint file...
    # Saving variables
    if (net == 'DIP_VAE'):
        out, mu, logvar, z = model(image_net_input_torch)
    else:
        out = model(image_net_input_torch)

    # Descaling like at the beginning
    out_descale = descale_imag(out, mean_label, std_label, scaling_input)
    # Saving image output
    net_outputs_path = subroot+'Block2/out_cnn/' + format(test) + '/out_' + net + '_post_reco_epoch=' + format(0) + suffix_func(config) + '.img'
    save_img(out_descale, net_outputs_path)
    # Squeeze image by loading it
    out_descale = fijii_np(net_outputs_path,shape=(PETImage_shape)) # loading DIP output
    
    writer = model.logger.experiment # Assess to new variable, otherwise error : weakly-referenced object ...
    write_image_tensorboard(writer,image_corrupt,"Corrupted image to fit") # Showing corrupted image
    for epoch in range(0,sub_iter_DIP,sub_iter_DIP//10):      
        write_image_tensorboard(writer,image_net_input,"DIP input (FULL CONTRAST)",epoch,full_contrast=True) # DIP input in tensorboard

        if (epoch > 0):
            # Train model using previously trained network (at iteration before)
            model = train_process(config, finetuning, processing_unit, sub_iter_DIP//10, admm_it, image_net_input_torch, image_corrupt_torch)

            # Saving variables
            if (net == 'DIP_VAE'):
                out, mu, logvar, z = model(image_net_input_torch)
            else:
                out = model(image_net_input_torch)

            # Descale like at the beginning
            out_descale = descale_imag(out, mean_label, std_label, scaling_input)
            # Saving image output
            net_outputs_path = subroot+'Block2/out_cnn/' + format(test) + '/out_' + net + '_post_reco_epoch=' + format(epoch) + suffix_func(config) + '.img'
            save_img(out_descale, net_outputs_path)
            # Squeeze image by loading it
            out_descale = fijii_np(net_outputs_path,shape=(PETImage_shape)) # loading DIP output
            # Metrics for NN output
            compute_metrics(PETImage_shape,out_descale,image_gt,epoch,PSNR_recon,PSNR_norm_recon,MSE_recon,MA_cold_recon,CRC_hot_recon,CRC_bkg_recon,IR_bkg_recon,bias_cold_recon,bias_hot_recon,writer=writer,write_tensorboard=True)
            # Saving (now DESCALED) image output
            save_img(out_descale, net_outputs_path)

        # Write images over epochs
        write_image_tensorboard(writer,out_descale,"Image over epochs (" + net + "output)",epoch) # Showing all images with same contrast to compare them together
        write_image_tensorboard(writer,out_descale,"Image over epochs (" + net + "output, FULL CONTRAST)",epoch,full_contrast=True) # Showing each image with contrast = 1
        writer.close()

    print('Finish')

"""
Receiving variables from block 1 part and initializing variables
"""

net = 'DIP' # Network architecture
processing_unit = 'CPU' # Processing unit (CPU or GPU)

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
#'''
config = {
    "lr" : tune.grid_search([0.0041]), # 0.01 for DIP, 0.001 for DD
    "sub_iter_DIP" : tune.grid_search([200]), # 10 for DIP, 100 for DD
    "rho" : tune.grid_search([0.0003]),
    "opti_DIP" : tune.grid_search(['Adam']),
    "mlem_sequence" : tune.grid_search([False]),
    "d_DD" : tune.grid_search([4]), # not above 4, otherwise 112 is too little as output size / not above 6, otherwise 128 is too little as output size
    "k_DD" : tune.grid_search([32]),
    "skip_connections" : tune.grid_search([0,1,2,3])
}
#'''

if processing_unit == 'CPU':
    resources_per_trial = {"cpu": 1, "gpu": 0}
elif processing_unit == 'GPU':
    resources_per_trial = {"cpu": 0, "gpu": 0.1} # "gpu": 1 / config_combination
    #resources_per_trial = {"cpu": 0, "gpu": 1} # "gpu": 1 / config_combination
elif processing_unit == 'both':
    resources_per_trial = {"cpu": 10, "gpu": 1} # not efficient

anaysis_raytune = tune.run(partial(post_reconstruction,root=os.getcwd()), config=config, local_dir = os.getcwd() + '/runs', resources_per_trial = resources_per_trial)#, progress_reporter = reporter)