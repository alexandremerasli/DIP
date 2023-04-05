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
    if ('ADMMLim' not in method and method != "nested"):
        config.pop("nb_outer_iteration", None)
        config.pop("alpha", None)
        config.pop("adaptive_parameters", None)
        config.pop("mu_adaptive", None)
        config.pop("tau", None)
        config.pop("xi", None)
    if ('ADMMLim' not in method and method != "nested" and method != "Gong"):
        config.pop("nb_inner_iteration", None)
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
        config["nb_inner_iteration"] = 50
    if method == 'nested':
        config["scaling"] = "standardization"
        config["nb_inner_iteration"] = 10
    config.pop("d_DD", None)
    config.pop("k_DD", None)
    
    if (method == 'MLEM' or method == 'AML'):
        config.pop("rho", None)
    print("aaaaaaaaaaa")
    print(config)
    return config


def suffix_func(config,NNEPPS=False):
    config_copy = dict(config)
    print(config_copy)
    config_copy = parametersIncompatibility(config_copy)
    print(config_copy)
    if (NNEPPS==False):
        config_copy.pop('NNEPPS',None)
    config_copy.pop('nb_outer_iteration',None)
    suffix = "config"
    for key, value in config_copy.items():
        suffix +=  "_" + key[:min(len(key),5)] + "=" + str(value)
    return suffix
    
def path_from_config(config,root):
    path = root + 'image0/replicate_1/' + method + '/'
    if (method == "Gong" or method == "nested"):
        path += 'Block2/out_cnn/24/out_DIP9' + suffix_func(config) + '.img'
        #path += 'Block2/out_cnn/24/out_DIP0' + suffix_func(config) + '.img'
    elif (method == "BSREM"):
        path += "config_rho=" + str(config["rho"]) + "_mlem_=False/BSREM_it30.img"
    elif (method == "MLEM"):
        path += "config_mlem_=False_post_=" + str(config["post_smoothing"]) + "/MLEM_it300.img"

    return path

def fijii_np(path,shape,type_im='<f'):
    """"Transforming raw data to numpy array"""
    file_path=(path)
    dtype = np.dtype(type_im)
    fid = open(file_path, 'rb')
    data = np.fromfile(fid,dtype)
    image = data.reshape(shape)
    return image

def show_image(config):
    root = os.getcwd() + '/data/Algo/'
    PETImage_shape = (112,112)

    if (method == "Gong" or method == "nested"):
        img1_np = fijii_np(path_from_config(config,root), shape=(PETImage_shape),type_im='<f')
    else:
        img1_np = fijii_np(path_from_config(config,root), shape=(PETImage_shape),type_im='<d')

    plt.figure()
    #plt.imshow(np.abs(img1_np), cmap='gray_r')
    plt.imshow(img1_np, cmap='gray_r',vmin=0,vmax=500)
    plt.title('img1')
    plt.colorbar()
    print("image saved")
    plt.savefig(root+'img1.png')

def show_image_path(path):
    root = os.getcwd() + '/data/Algo/'
    PETImage_shape = (112,112)
    img1_np = fijii_np(path, shape=(PETImage_shape),type_im='<f')

    plt.figure()
    #plt.imshow(np.abs(img1_np), cmap='gray_r')
    plt.imshow(img1_np, cmap='gray_r',vmin=0,vmax=500)
    #plt.imshow(img1_np, cmap='gray_r')
    plt.title('img1')
    plt.colorbar()
    print("image saved")
    plt.savefig(root+'img_non_Gong.png')


# Configuration dictionnary for general parameters (not hyperparameters)
settings_config = {
    "image" : tune.grid_search(['image0']), # Image from database
    "net" : tune.grid_search(['DIP']), # Network to use (DIP,DD,DD_AE,DIP_VAE)
    "random_seed" : tune.grid_search([True]), # If True, random seed is used for reproducibility (must be set to False to vary weights initialization)
    "method" : tune.grid_search(['BSREM']), # Reconstruction algorithm (nested, Gong, or algorithms from CASToR (MLEM, BSREM, AML, etc.))
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
config = {
    "rho" : tune.grid_search([0.03]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (nested and Gong)
    #"rho" : tune.grid_search([0.003,0.0003,0.00003]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (nested and Gong)
    ## network hyperparameters
    "lr" : tune.grid_search([0.05]), # Learning rate in network optimization
    "sub_iter_DIP" : tune.grid_search([100]), # Number of epochs in network optimization
    "opti_DIP" : tune.grid_search(['Adam']), # Optimization algorithm in neural network training (Adam, LBFGS)
    "skip_connections" : tune.grid_search([3]), # Number of skip connections in DIP architecture (0, 1, 2, 3)
    #"skip_connections" : tune.grid_search([0,1,2,3]), # Number of skip connections in DIP architecture (0, 1, 2, 3)
    "scaling" : tune.grid_search(['normalization']), # Pre processing of neural network input (nothing, uniform, normalization, standardization)
    "input" : tune.grid_search(['CT']), # Neural network input (random or CT)
    #"input" : tune.grid_search(['CT','random']), # Neural network input (random or CT)
    "d_DD" : tune.grid_search([4]), # d for Deep Decoder, number of upsampling layers. Not above 4, otherwise 112 is too little as output size / not above 6, otherwise 128 is too little as output size
    "k_DD" : tune.grid_search([32]), # k for Deep Decoder
    ## ADMMLim - OPTITR hyperparameters
    "nb_inner_iteration" : tune.grid_search([50]), # Number of inner iterations in ADMMLim (if mlem_sequence is False) or in OPTITR (for Gong)
    "nb_outer_iteration": tune.grid_search([10]), # Number outer iterations in ADMMLim
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
    "hyperparameters" : list(config.keys())
}
config = {**config, **config, **split_config}


config_copy = dict(config)
config_copy = dict(config)
for key, value in config_copy.items():
    if key !="debug" and key != "hyperparameters":
        config_copy[key] = value["grid_search"][0]

for key, value in config_copy.items():
    config_copy[key] = value["grid_search"][0]

method = config_copy["method"]

show_image(config_copy,config_copy)

#show_image_path("/home/meraslia/workspace_reco/nested_admm/data/Algo/image0/replicate_1/Gong/Block2/out_cnn/24/out_DIP_post_reco_epoch=99config_rho=0.0003_lr=0.5_sub_i=100_opti_=Adam_skip_=3_scali=nothing_input=CT_sub_i=50_mlem_=False.img")
#show_image_path("/home/meraslia/workspace_reco/nested_admm/data/Algo/Data/im_corrupt_beginning_10.img")
show_image_path("/home/meraslia/workspace_reco/nested_admm/data/Algo/Data/database_v2/image0/image0_atn.raw")