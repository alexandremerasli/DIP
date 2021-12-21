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
variable_scaled : normalized, standardized (or nothing) variable
variable_torch : torch representation of variable
"""

## Python libraries

# Useful
import sys
import warnings

# Math
import numpy as np

# Pytorch
import torch
from torchsummary import summary

# Local files to import
from utils_func import *

"""
Receiving variables from block 1 part and initializing variables
"""

admm_it = int(sys.argv[1]) # Current ADMM iteration 
test = int(sys.argv[2]) # Label of the experiment
net = sys.argv[3] # Network architecture
processing_unit = sys.argv[4] # Processing unit (CPU or GPU)
finetuning = sys.argv[5] # Finetuning (with best model or last saved model for checkpoint) or not for the DIP optimizations
PETImage_shape_str = sys.argv[6] # PET input dimensions (string, tF)
root = sys.argv[7] # Directory root
suffix =  sys.argv[8] # Suffix to make difference between hyperparameters runs

subroot = root+'/data/Algo/'

config = np.load(subroot + 'Config/config' + suffix + '.npy',allow_pickle='TRUE').item()
# Config dictionnary for hyperparameters
# lr = config["lr"]
sub_iter_DIP = config["sub_iter_DIP"]
# rho = config["rho"]
# opti_DIP = config["opti_DIP"]
scaling_input = config["scaling"]

# Defining PET input dimensions
PETImage_shape = input_dim_str_to_list(PETImage_shape_str)

## Loading RAW stack of images
# Loading DIP input (we do not have CT-map, so random image created in block 1)
image_net_input_torch = torch.load(subroot + 'Data/initialization/image_' + net + '_input_torch.pt')# DO NOT CREATE RANDOM INPUT IN BLOCK 2 !!! ONLY AT THE BEGINNING, IN BLOCK 1. JUST LOAD IT FROM BLOCK 1

# Loading DIP x_label (corrupted image) from block1
image_corrupt = fijii_np(subroot+'Block2/x_label/' + format(test)+'/'+ format(admm_it) +'_x_label' + suffix + '.img',shape=(PETImage_shape))
# Scaling of x_label image
image_net_input_scale,param1_scale_im_corrupt,param2_scale_im_corrupt= rescale_imag(image_corrupt) # Scaling of x_label image

## Transforming numpy variables to torch tensors


# Corrupted image x_label, numpy --> torch
image_corrupt_torch = torch.Tensor(image_net_input_scale)
# Adding dimensions to fit network architecture
image_corrupt_torch = image_corrupt_torch.view(1,1,PETImage_shape[0],PETImage_shape[1])

"""
Training the model using checkpoint to load model
"""

'''
from torch.utils.data import Dataset
class Dataset():

    # load the dataset
    def __init__(self, X, y):
        # store the inputs and outputs
        self.X = X
        self.y = y
     
    # number of rows in the dataset
    def __len__(self):
        return self.X.size()[0]
     
    # get a row at an index
    def __getitem__(self, idx):

        X_sample = self.X[idx]
        y_sample = self.y[idx]
        
        return X_sample, y_sample
'''
#from memory_profiler import profile
 
#@profile
def train_process(config, finetuning, processing_unit, sub_iter_DIP, admm_it, image_net_input_torch, image_corrupt_torch):
    # Implements Dataset
    train_dataset = torch.utils.data.TensorDataset(image_net_input_torch, image_corrupt_torch)
    # train_dataset = Dataset(image_net_input_torch, image_corrupt_torch)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1) # num_workers is 0 by default, which means the training process will work sequentially inside the main process

    # Choose network architecture as model
    model, model_class = choose_net(net, config)

    #'''
    if (processing_unit == 'CPU'):
        #Summary of the network
        if (net == 'DIP' or net == 'DIP_VAE'):
            summary(model, input_size=(1,PETImage_shape[0],PETImage_shape[1])) # for DIP
        else:
            if (net == 'DD'):
                input_size_DD = int(PETImage_shape[0] / (2**config["d_DD"])) # if original Deep Decoder (i.e. only with decoder part)
                summary(model, input_size=(config["k_DD"],input_size_DD,input_size_DD)) # For Deep Decoder, # if original Deep Decoder (i.e. only with decoder part)
            elif (net == 'DD_AE'):
                input_size_DD = PETImage_shape[0] # if auto encoder based on Deep Decoder
                summary(model, input_size=(1,input_size_DD,input_size_DD)) # For Deep Decoder,  # if auto encoder based on Deep Decoder
    else:
        warnings.warn("GPU have problems then between inputs and weights, so use CPU if you want to see torch summary")
    #'''
    
    # Loading using previous model if we want to do finetuning
    checkpoint_simple_path = subroot+'Block2/checkpoint/'
    checkpoint_simple_path_exp = subroot+'Block2/checkpoint/'+format(test)  + '/' + suffix + '/'

    model = load_model(image_net_input_torch, config, finetuning, admm_it, model, model_class, subroot, checkpoint_simple_path_exp, training=True)

    # Start training
    print('Starting optimization, iteration',admm_it)
    trainer = create_pl_trainer(finetuning, processing_unit, sub_iter_DIP, admm_it, net, checkpoint_simple_path, test, checkpoint_simple_path_exp)

    trainer.fit(model, train_dataloader)

    return model

model = train_process(config, finetuning, processing_unit, sub_iter_DIP, admm_it, image_net_input_torch, image_corrupt_torch)

"""
Saving variables and model
"""
if (net == 'DIP_VAE'):
    out, mu, logvar, z = model(image_net_input_torch)
else:
    out = model(image_net_input_torch)

# Destandardize like at the beginning
out_descale = descale_imag(out,param1_scale_im_corrupt,param2_scale_im_corrupt,scaling_input)
# Saving image output
save_img(out_descale, subroot+'Block2/out_cnn/' + format(test) + '/out_' + net + '' + format(admm_it) + suffix + '.img')

print('Finish')