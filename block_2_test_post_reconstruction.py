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
from pathlib import Path
import warnings
from datetime import datetime

# Math
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt

# Pytorch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

# Local files to import
from utils_func import *

"""
Receiving variables from block 1 part and initializing variables
"""

# Configuration dictionnary for hyperparameters to tune
config = {
    "lr" : 0.001,
    "sub_iter_DIP" : 200,
    "rho" : 0.003,
    "opti_DIP" : 'Adam',
    "mlem_sequence" : None, # None means we are in post reconstruction mode
    "d_DD" : 3,
    "k_DD" : 32
}

max_iter = 1 # Outer iterations
test = 24 # Label of the experiment
net = 'DIP_VAE' # Network architecture
processing_unit = 'CPU' # Processing unit (CPU or GPU)
finetuning = 'False' # Finetuning (with best model or last saved model for checkpoint) or not for the DIP optimizations
# PETImage_shape_str = sys.argv[6] # PET input dimensions (string, tF)
root = os.getcwd() # Directory root
suffix =  'TEST_post_reconstruction_' + suffix_func(config) # suffix to make difference between raytune runs (different hyperparameters)

subroot = root+'/data/Algo/'

np.save(subroot + 'Config/config' + suffix + '.npy', config) # Save this configuration of hyperparameters, and reload it at the beginning of block 2 thanks to suffix (passed in subprocess call argumentsZ)

config = np.load(subroot + 'Config/config' + suffix + '.npy',allow_pickle='TRUE').item()
# Config dictionnary for hyperparameters
# lr = config["lr"]
sub_iter_DIP = config["sub_iter_DIP"]
# rho = config["rho"]
# opti_DIP = config["opti_DIP"]

# Define PET input dimensions according to input data dimensions
PETImage_shape_str = read_input_dim()
PETImage_shape = input_dim_str_to_list(PETImage_shape_str)

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
image_corrupt = fijii_np(subroot+'Comparaison/MLEM/MLEM_converge.img',shape=(PETImage_shape))
#image_corrupt = fijii_np(subroot+'Comparaison/MLEM/MLEM_it2.img',shape=(PETImage_shape))

# Normalization of x_label image
# image_corrupt_norm_scale, maxe = norm_imag(image_corrupt) # Normalization of x_label image
image_corrupt_norm,mean_label,std_label= stand_imag(image_corrupt) # Standardization of x_label image

## Transforming numpy variables to torch tensors


# Corrupted image x_label, numpy --> torch
image_corrupt_torch = torch.Tensor(image_corrupt_norm)
# Adding dimensions to fit network architecture
image_corrupt_torch = image_corrupt_torch.view(1,1,PETImage_shape[0],PETImage_shape[1])

def train_process(config, finetuning, processing_unit, sub_iter_DIP, max_iter, image_net_input_torch, image_corrupt_torch):
    # Implements Dataset
    train_dataset = torch.utils.data.TensorDataset(image_net_input_torch, image_corrupt_torch)
    # train_dataset = Dataset(image_net_input_torch, image_corrupt_torch)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1) # num_workers is 0 by default, which means the training process will work sequentially inside the main process

    # Choose network architecture as model
    model, model_class = choose_net(net, config)

    checkpoint_simple_path = 'runs/' # To log loss in tensorboard thanks to Logger
    checkpoint_simple_path_exp = '' # We do not need its value in this script
    
    # Start training
    print('Starting optimization, iteration',max_iter)
    trainer = create_pl_trainer(finetuning, processing_unit, sub_iter_DIP, checkpoint_simple_path, test, checkpoint_simple_path_exp,name=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    trainer.fit(model, train_dataloader)

    return model

model = train_process(config, finetuning, processing_unit, sub_iter_DIP, max_iter, image_net_input_torch, image_corrupt_torch)

"""
Saving variables and model
"""

writer = model.logger.experiment # Assess to new variable, otherwise error : weakly-referenced object ...
for epoch in range(0,sub_iter_DIP,sub_iter_DIP//10):
    # Load saved (STANDARDIZED) images
    net_outputs_path = subroot+'Block2/out_cnn/' + format(test) + '/out_' + net + '_post_reco_epoch=' + format(epoch) + '.img'
    out = fijii_np(net_outputs_path,shape=(PETImage_shape))
    # Destandardize like at the beginning
    out_destand = destand_numpy_imag(out, mean_label, std_label)
    # Saving (now DESTANDARDIZED) image output
    save_img(out_destand, net_outputs_path)
    # Write images over epochs
    write_image_tensorboard(writer,out_destand,"Image over epochs (" + net + "output)",epoch) # Showing all images with same contrast to compare them together
    write_image_tensorboard(writer,out_destand,"Image over epochs (" + net + "output, FULL CONTRAST)",epoch,full_contrast=True) # Showing each image with contrast = 1
    writer.close()

print('Finish')