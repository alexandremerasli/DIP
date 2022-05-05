import numpy as np
import matplotlib.pyplot as plt
from ray import tune
import os
#from main import config

def parametersIncompatibility(config,task=None):
    
    # Delete hyperparameters specific to others optimizer 

    if (method != "AML"):
        config.pop("A_AML", None)
    if (method == 'BSREM' or method == 'nested' or method == 'Gong'):
        config.pop("post_smoothing", None)
    if (method != "ADMMLim" and method != "nested"):
        config.pop("nb_iter_second_admm", None)
        config.pop("alpha", None)
    if (method != "ADMMLim" and method != "nested" and method != "Gong"):
        config.pop("sub_iter_PLL", None)
    if (method != "nested" and method != "Gong" and task != "post_reco"):
        config.pop("lr", None)
        config.pop("sub_iter_DIP", None)
        config.pop("opti_DIP", None)
        config.pop("skip_connections", None)
        config.pop("scaling", None)
        config.pop("input", None)
        config.pop("d_DD", None)
        config.pop("k_DD", None)
    if method == 'Gong':
        config["scaling"] = "nothing"
        config["scaling"] = "normalization"
        config["sub_iter_PLL"] = 50
    if method == 'nested':
        config["scaling"] = "standardization"
        config["sub_iter_PLL"] = 10
    config.pop("d_DD", None)
    config.pop("k_DD", None)
    
    if (method == 'MLEM' or method == 'AML'):
        config.pop("rho", None)
    print("aaaaaaaaaaa")
    print(config)
    return config


def suffix_func(hyperparameters_config,NNEPPS=False):
    hyperparameters_config_copy = dict(hyperparameters_config)
    print(hyperparameters_config_copy)
    hyperparameters_config_copy = parametersIncompatibility(hyperparameters_config_copy)
    print(hyperparameters_config_copy)
    if (NNEPPS==False):
        hyperparameters_config_copy.pop('NNEPPS',None)
    hyperparameters_config_copy.pop('nb_iter_second_admm',None)
    suffix = "config"
    for key, value in hyperparameters_config_copy.items():
        suffix +=  "_" + key[:min(len(key),5)] + "=" + str(value)
    return suffix
    
def path_from_config(hyperparameters_config,config,root):
    path = root + 'image0/replicate_1/' + method + '/'
    if (method == "Gong" or method == "nested"):
        path += 'Block2/out_cnn/24/out_DIP9' + suffix_func(hyperparameters_config) + '.img'
    elif (method == "BSREM"):
        path += "config_rho=" + str(config["rho"]) + "_mlem_=False/BSREM_it30.img"
    elif (method == "MLEM"):
        path += "config_mlem_=False_post_=" + str(config["post_smoothing"]) + "/MLEM_it300.img"

    return path

def fijii_np(path,shape,type='<f'):
    """"Transforming raw data to numpy array"""
    file_path=(path)
    dtype = np.dtype(type)
    fid = open(file_path, 'rb')
    data = np.fromfile(fid,dtype)
    image = data.reshape(shape)
    return image

def show_image(hyperparameters_config,config):
    root = os.getcwd() + '/data/Algo/'
    PETImage_shape = (112,112)

    if (method == "Gong" or method == "nested"):
        img1_np = fijii_np(path_from_config(hyperparameters_config,config,root), shape=(PETImage_shape),type='<f')
    else:
        img1_np = fijii_np(path_from_config(hyperparameters_config,config,root), shape=(PETImage_shape),type='<d')

    plt.figure()
    #plt.imshow(np.abs(img1_np), cmap='gray_r')
    plt.imshow(img1_np, cmap='gray_r')
    plt.title('img1')
    plt.colorbar()
    print("image saved")
    plt.savefig(root+'img1.png')


# Configuration dictionnary for general parameters (not hyperparameters)
fixed_config = {
    "image" : tune.grid_search(['image0']), # Image from database
    "net" : tune.grid_search(['DIP']), # Network to use (DIP,DD,DD_AE,DIP_VAE)
    "random_seed" : tune.grid_search([True]), # If True, random seed is used for reproducibility (must be set to False to vary weights initialization)
    "method" : tune.grid_search(['nested']), # Reconstruction algorithm (nested, Gong, or algorithms from CASToR (MLEM, BSREM, AML, etc.))
    "processing_unit" : tune.grid_search(['CPU']), # CPU or GPU
    "nb_threads" : tune.grid_search([64]), # Number of desired threads. 0 means all the available threads
    "FLTNB" : tune.grid_search(['double']), # FLTNB precision must be set as in CASToR (double necessary for ADMMLim and nested)
    "debug" : False, # Debug mode = run without raytune and with one iteration
    "max_iter" : tune.grid_search([30]), # Number of global iterations for usual optimizers (MLEM, BSREM, AML etc.) and for nested and Gong
    "nb_subsets" : tune.grid_search([28]), # Number of subsets in chosen reconstruction algorithm (automatically set to 1 for ADMMLim)
    "finetuning" : tune.grid_search(['last']),
    "experiment" : tune.grid_search([24]),
    "image_init_path_without_extension" : tune.grid_search(['1_im_value_cropped']), # Initial image of the reconstruction algorithm (taken from data/algo/Data/initialization)
    #"f_init" : tune.grid_search(['1_im_value_cropped']),
    "penalty" : tune.grid_search(['MRF']), # Penalty used in CASToR for PLL algorithms
    "replicates" : tune.grid_search(list(range(1,1+1))), # List of desired replicates. list(range(1,n+1)) means n replicates
    "average_replicates" : tune.grid_search([False]), # List of desired replicates. list(range(1,n+1)) means n replicates
}
# Configuration dictionnary for hyperparameters to tune
hyperparameters_config = {
    "rho" : tune.grid_search([0.0003]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (nested and Gong)
    #"rho" : tune.grid_search([0.003,0.0003,0.00003]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (nested and Gong)
    ## network hyperparameters
    "lr" : tune.grid_search([0.05]), # Learning rate in network optimization
    "sub_iter_DIP" : tune.grid_search([100]), # Number of epochs in network optimization
    "opti_DIP" : tune.grid_search(['Adam']), # Optimization algorithm in neural network training (Adam, LBFGS)
    "skip_connections" : tune.grid_search([0]), # Number of skip connections in DIP architecture (0, 1, 2, 3)
    #"skip_connections" : tune.grid_search([0,1,2,3]), # Number of skip connections in DIP architecture (0, 1, 2, 3)
    "scaling" : tune.grid_search(['normalization']), # Pre processing of neural network input (nothing, uniform, normalization, standardization)
    "input" : tune.grid_search(['CT']), # Neural network input (random or CT)
    #"input" : tune.grid_search(['CT','random']), # Neural network input (random or CT)
    "d_DD" : tune.grid_search([4]), # d for Deep Decoder, number of upsampling layers. Not above 4, otherwise 112 is too little as output size / not above 6, otherwise 128 is too little as output size
    "k_DD" : tune.grid_search([32]), # k for Deep Decoder
    ## ADMMLim - OPTITR hyperparameters
    "sub_iter_PLL" : tune.grid_search([50]), # Number of inner iterations in ADMMLim (if mlem_sequence is False) or in OPTITR (for Gong)
    "nb_iter_second_admm": tune.grid_search([10]), # Number outer iterations in ADMMLim
    "alpha" : tune.grid_search([0.005]), # alpha (penalty parameter) in ADMMLim
    ## hyperparameters from CASToR algorithms 
    # Optimization transfer (OPTITR) hyperparameters
    "mlem_sequence" : tune.grid_search([False]), # Given sequence (with decreasing number of subsets) to quickly converge. True or False
    # AML hyperparameters
    "A_AML" : tune.grid_search([-100]), # AML lower bound A
    # Post smoothing by CASToR after reconstruction
    #"post_smoothing" : tune.grid_search([0]), # Post smoothing by CASToR after reconstruction
    "post_smoothing" : tune.grid_search([3,6,9,12,15]), # Post smoothing by CASToR after reconstruction
    # NNEPPS post processing
    "NNEPPS" : tune.grid_search([False]), # NNEPPS post-processing. True or False
}

# Merge 2 dictionaries
split_config = {
    "hyperparameters" : list(hyperparameters_config.keys())
}
config = {**fixed_config, **hyperparameters_config, **split_config}


config_copy = dict(config)
hyperparameters_config_copy = dict(hyperparameters_config)
for key, value in config_copy.items():
    if key !="debug" and key != "hyperparameters":
        config_copy[key] = value["grid_search"][0]

for key, value in hyperparameters_config_copy.items():
    hyperparameters_config_copy[key] = value["grid_search"][0]

method = config_copy["method"]

show_image(hyperparameters_config_copy,config_copy)