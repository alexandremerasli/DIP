#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: alexandre (2021-2022)
"""

## Python libraries
# Useful
import os
from ray import tune

#import sys
#stdoutOrigin=sys.stdout 
#sys.stdout = open("test_log.txt", "w")

# Configuration dictionary for general settings parameters
    # Parameters in this dictionary are not added in suffix of the output folder as they are not hyperparameters
settings_config = {
    "image" : tune.grid_search(['image2_0']), # Image from database (data/Algo/Data/database_v2)
    "random_seed" : tune.grid_search([True]), # If True, random seed is used for reproducibility (must be set to False to vary weights initialization)
    "method" : tune.grid_search(['BSREM']), # Reconstruction algorithm (DNA, DIPRecon, or algorithms from CASToR (MLEM, BSREM, AML, etc.))
    "processing_unit" : tune.grid_search(['CPU']), # Run NN training with pytorch on CPU or GPU
    "nb_threads" : tune.grid_search([1]), # Number of desired threads in reconstruction with CASToR. 0 means all the available threads will be used
    "FLTNB" : tune.grid_search(['float']), # FLTNB precision must be set as in CASToR. Default is float (meaning float32 in numpy)
    "ray" : True, # If True, run computation with raytune parallel computation (for several settings in parallel). Set it to False to debug code
    "tensorboard" : False, # If True, show results (images, metrics) in tensorboard during or after reconstruction. Set it to False to save time
    "all_images_DIP_when" : tune.grid_search(['True_init']), # For DIP-based algorithms or DIP denoising, option to choose which DIP outputs to save. Can be set to "True" (save all DIP outputs), "False" (10 images like in tensorboard (quicker, for visualization), "Unique" (store only last image), "True_init" (save all DIP images at initialization and then last one for each outer iteration)
    "experiment" : tune.grid_search([24]),
    "replicates" : tune.grid_search(list(range(1,40+1))), # List of desired replicates to work with in parallel. list(range(1,n+1)) means n replicates
    # "replicates" : tune.grid_search([4]), # List of desired replicates to work with in parallel. list(range(1,n+1)) means n replicates
    "castor_foms" : tune.grid_search([True]), # Set to True to compute CASToR Figure Of Merits or residuals for ADMMReg. Must be set to False with list mode data
}
# Configuration dictionary for previous hyperparameters, but fixed to simplify
fixed_config = {
    "max_iter" : tune.grid_search([30]), # Number of iterations for usual optimizers (MLEM, BSREM, AML etc.) and outer iterations for DNA and DIPRecon
    "nb_subsets" : tune.grid_search([28]), # Number of subsets in chosen reconstruction algorithm (automatically set to 1 for ADMMReg)
    "finetuning" : tune.grid_search(['last']),
    "penalty" : tune.grid_search(['MRF']), # Penalty used in CASToR for PLL algorithms (MRF)
    "unnested_1st_outer_iter" : tune.grid_search([False]), # If True, unnested are computed after 1st outer iteration (because rho is set to 0). If False, needs to set f_init to initialize the network, as in DIPRecon paper, and rho is not changed.
    "sub_iter_DIP_init" : tune.grid_search([1000]), # Number of DIP iterations at DNA/DIPRecon initialization. Could be overrided if early stopping point is reached using "DIP_early_stopping_when" parameter
    "nb_inner_sub_iteration" : tune.grid_search([1]), # Number of inner subiterations in DNA (number of iterations of gradient descent in ADMM-Reg (if mlem_sequence is False). It should be 1 as it is coded for now in CASToR
    "xi" : tune.grid_search([1]), # Factor to balance primal and dual residual convergence speed in adaptive tau computation in ADMMReg
    "net" : tune.grid_search(['DIP']), # Neural Network (NN) architecture to use ("DIP" (U-Net from DIPRecon paper), "DD" (Deep Decoder"), "DD_AE" (DD based autoencoder), "DIP_VAE" (DIP-based Variational AutoEncoder)))
    "DIP_early_stopping_when" : tune.grid_search(["never"]), # Use DIP early stopping - ES ("never" means no ES, "init" means ES only at initialization, "all" means ES at each iteration)
    "windowSize" : tune.grid_search([50]), # WMV window size
    "patienceNumber" : tune.grid_search([100]), # Patience number in moving variance algorithms
}
# Configuration dictionary for hyperparameters to tune
hyperparameters_config = {
    "image_init_path_without_extension" : tune.grid_search(['1_im_value_cropped']), # Initial image of the reconstruction algorithm (taken from subroot + "/Data/initialization")
    "rho" : tune.grid_search([0.003,8e-4,0.008,0.03]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (DNA and DIPRecon)
    "rho" : tune.grid_search([0.0002]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (DNA and DIPRecon)
    "adaptive_parameters_DIP" : tune.grid_search(["nothing"]), # which parameters are adaptive ? Must be set to nothing, alpha, or tau (which means alpha and tau)
    "mu_DIP" : tune.grid_search([10]), # Factor to balance primal and dual residual in adaptive alpha computation in ADMMReg
    "tau_DIP" : tune.grid_search([2]), # Factor to multiply alpha in adaptive alpha computation in ADMMReg. If adaptive tau, it corresponds to tau max
    ## network hyperparameters
    "lr" : tune.grid_search([1e-4,4e-4,7e-4,1e-3,4e-3,7e-3,1e-2,4e-2,7e-2,0.1,0.4,0.7,1]), # Learning rate in network optimization
    "lr" : tune.grid_search([1e-4,1e-3,1e-2,0.1,1]), # Learning rate in network optimization
    #"lr" : tune.grid_search([1e-9,1e-8,1e-7,1e-6,1e-5]), # Learning rate in network optimization
    "lr" : tune.grid_search([1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1,10,100]), # Learning rate in network optimization
    "lr" : tune.grid_search([1e-5,1e-4,1e-3,1e-2,0.1,1]), # Learning rate in network optimization
    #"lr" : tune.grid_search([10]), # Learning rate in network optimization
    "sub_iter_DIP" : tune.grid_search([500]), # Number of epochs in network optimization
    "opti_DIP" : tune.grid_search(['Adam','LBFGS']), # Optimization algorithm in neural network training (Adam, LBFGS)
    "opti_DIP" : tune.grid_search(['LBFGS']), # Optimization algorithm in neural network training (Adam, LBFGS)
    "opti_DIP" : tune.grid_search(['Adadelta']), # Optimization algorithm in neural network training (Adam, LBFGS)
    "opti_DIP" : tune.grid_search(['Adam']), # Optimization algorithm in neural network training (Adam, LBFGS)
    "skip_connections" : tune.grid_search([0]), # Number of skip connections in DIP architecture (0, 1, 2, 3)
    "scaling" : tune.grid_search(['standardization']), # Pre processing of neural network input (nothing, uniform, normalization, standardization)
    "input" : tune.grid_search(['random']), # Neural network input (random or anatomical )
    "input" : tune.grid_search(["anatomical"]), # Neural network input (random or anatomical )
    #"input" : tune.grid_search(["anatomical",'random']), # Neural network input (random or anatomical )
    "d_DD" : tune.grid_search([4]), # d for Deep Decoder, number of upsampling layers. Not above 4, otherwise 112 is too little as output size / not above 6, otherwise 128 is too little as output size
    "k_DD" : tune.grid_search([32]), # k for Deep Decoder
    ## ADMMReg - OPTITR hyperparameters
    #"nb_inner_iteration": tune.grid_search([30]), # Number of inner iterations in DNA and DIPRecon (respectively number of iterations of ADMM-Reg and OPTITR)
    #"nb_inner_iteration": tune.grid_search([3]), # Number of inner iterations in DNA and DIPRecon (respectively number of iterations of ADMM-Reg and OPTITR)
    "nb_inner_iteration": tune.grid_search([30]), # Number of inner iterations in DNA and DIPRecon (respectively number of iterations of ADMM-Reg and OPTITR)
    "alpha" : tune.grid_search([1]), # alpha (penalty parameter) in ADMMReg
    "adaptive_parameters" : tune.grid_search(["both"]), # which parameters are adaptive ? Must be set to nothing, alpha, or both (which means alpha and tau)
    "mu_adaptive" : tune.grid_search([2]), # Factor to balance primal and dual residual in adaptive alpha computation in ADMMReg
    "tau" : tune.grid_search([2]), # Factor to multiply alpha in adaptive alpha computation in ADMMReg
    "tau_max" : tune.grid_search([100]), # Maximum value for tau in adaptive tau in ADMMReg
    "stoppingCriterionValue" : tune.grid_search([0.01]), # Value of the stopping criterion in ADMMReg
    "saveSinogramsUAndV" : tune.grid_search([1]), # 1 means save sinograms u and v from CASToR, otherwise it means do not save them
    ## hyperparameters from CASToR algorithms 
    # Optimization transfer (OPTITR) hyperparameters
    "mlem_sequence" : tune.grid_search([False]), # Given sequence (with decreasing number of subsets) to quickly converge. True or False
    # AML/APPGML hyperparameters
    "A_AML" : tune.grid_search([-100,-500,-10000]), # AML lower bound A
    "A_AML" : tune.grid_search([-100]), # AML lower bound A
    # Post smoothing by CASToR after reconstruction
    "post_smoothing" : tune.grid_search([0]), # Post smoothing by CASToR after reconstruction
    #"post_smoothing" : tune.grid_search([6,9,12,15]), # Post smoothing by CASToR after reconstruction
    # NNEPPS post processing
    "NNEPPS" : tune.grid_search([False]), # NNEPPS post-processing. True or False
}

# Merge 3 dictionaries
split_config = {
    "fixed_hyperparameters" : list(fixed_config.keys()),
    "hyperparameters" : list(hyperparameters_config.keys())
}
config = {**settings_config, **fixed_config, **hyperparameters_config, **split_config}

root = os.getcwd()
subroot = "/data/Algo/"

# write random seed in a file to get it in network architectures
os.system("rm -rf " + os.getcwd() +"/seed.txt")
file_seed = open(os.getcwd() + "/seed.txt","w+")
file_seed.write(str(settings_config["random_seed"]["grid_search"][0]))
file_seed.close()

# Local files to import, AFTER CONFIG TO SET RANDOM SEED OR NOT
from iADMM_DIP import iADMM_DIP
from iCastorAlgo import iCastorAlgo
from iPostReconstruction import iPostReconstruction
from iResults import iResults
from iMeritsADMMReg import iMeritsADMMReg
from iMeritsDIP_ADMM import iMeritsDIP_ADMM
from iResultsAlreadyComputed import iResultsAlreadyComputed
from iResultsADMMReg_VS_APPGML import iResultsADMMReg_VS_APPGML
from iTradeoffCurves import iTradeoffCurves

for method in config["method"]['grid_search']:
    # Choose task to do
    #task = 'full_reco_with_network' # Run DIPRecon or DNA
    #task = 'castor_reco' # Run CASToR reconstruction with given optimizer
    # task = 'post_reco' # Run network denoising after a given reconstructed image im_corrupt
    #task = 'show_results_post_reco'
    #task = 'show_results'
    #task = 'show_metrics_results_already_computed'
    #task = 'show_metrics_ADMMReg'
    #task = 'show_metrics_DNA'
    #task = 'compare_2_methods'

    # Choose task to do if not defined
    if 'task' not in globals():
        if (method == "DIPRecon" or method == "DNA"):
            task = 'full_reco_with_network'

        elif ('ADMMReg' in method or method == 'MLEM' or method == 'OPTITR' or method == 'OSEM' or method == 'BSREM' or method == 'AML' or method == 'APPGML'):
            task = 'castor_reco'

    '''
    if task != "show_metrics_results_already_computed":
        # DIPRecon reconstruction
        if (method == "DIPRecon" and len(config["method"]["grid_search"]) == 1):
            print("configuration fiiiiiiiiiiiiiiiiiiile")
            #config = np.load(root + 'config_DIP.npy',allow_pickle='TRUE').item()
            from DIPRecon_configuration import config_func_MIC
            #config = config_func()
            config = config_func_MIC()

        # DNA reconstruction
        if (method == "DNA" and len(config["method"]["grid_search"]) == 1):
            print("configuration fiiiiiiiiiiiiiiiiiiile")
            from DNA_configuration import config_func_MIC
            #config = config_func()
            config = config_func_MIC()

        # MLEM reconstruction
        if (method == 'MLEM' and len(config["method"]["grid_search"]) == 1):
            print("configuration fiiiiiiiiiiiiiiiiiiile")
            from MLEM_configuration import config_func_MIC
            #config = config_func()
            config = config_func_MIC()

        # OSEM reconstruction
        if (method == 'OSEM' and len(config["method"]["grid_search"]) == 1):
            print("configuration fiiiiiiiiiiiiiiiiiiile")
            from OSEM_configuration import config_func_MIC
            #config = config_func()
            config = config_func_MIC()

        # BSREM reconstruction
        if (method == 'BSREM' and len(config["method"]["grid_search"]) == 1):
            print("configuration fiiiiiiiiiiiiiiiiiiile")
            from BSREM_configuration import config_func_MIC
            #config = config_func()
            config = config_func_MIC()

        # APPGML reconstruction
        if ('APPGML' in method and len(config["method"]["grid_search"]) == 1):
            print("configuration fiiiiiiiiiiiiiiiiiiile")
            from APPGML_configuration import config_func_MIC
            #config = config_func()
            config = config_func_MIC()

        # ADMMReg reconstruction
        if (method == 'ADMMReg' and len(config["method"]["grid_search"]) == 1):
            print("configuration fiiiiiiiiiiiiiiiiiiile")
            from ADMMReg_configuration import config_func_MIC
            #config = config_func()
            config = config_func_MIC()
    '''
    config_tmp = dict(config)
    #config_tmp["method"] = tune.grid_search([method]) # Put only 1 method to remove useless hyperparameters from settings_config and hyperparameters_config



    # Choose class to run according to task
    if (task == 'full_reco_with_network'): # Run DIPRecon or DNA
        classTask = iADMM_DIP(hyperparameters_config)
    elif (task == 'castor_reco'): # Run CASToR reconstruction with given optimizer
        classTask = iCastorAlgo(config)
    elif (task == 'post_reco'): # Run network denoising after a given reconstructed image im_corrupt
        classTask = iPostReconstruction(config)
    elif (task == 'show_results'): # Show already computed results over iterations
        classTask = iResults(config)
    elif (task == 'show_results_post_reco'): # Show already computed results over iterations of post reconstruction mode
        config["task"] = "show_results_post_reco"
        classTask = iResults(config)
    elif (task == 'show_metrics_ADMMReg'): # Show ADMMReg FOMs over iterations
        classTask = iMeritsADMMReg(config)
    elif (task == 'show_metrics_DNA'): # Show DNA or DIPRecon FOMs over iterations
        classTask = iMeritsDIP_ADMM(config)
    elif (task == 'show_metrics_results_already_computed'): # Show already computed results averaging over replicates
        classTask = iResultsAlreadyComputed(config)
    elif (task == 'compare_2_methods'): # Show already computed results averaging over replicates
        classTask = iResultsADMMReg_VS_APPGML(config)

    # Incompatible parameters (should be written in vGeneral I think)
    if (("DNA" in method or "DIPRecon" in method) and config["rho"]["grid_search"][0] == 0 and task == "castor_reco"):
        raise ValueError("DNA must be launched with rho > 0")
    elif ((method != "DIPRecon" and method != "DNA") and task == "post_reco"):
        raise ValueError("Only DIPRecon or DNA can be run in post reconstruction mode, not CASToR reconstruction algorithms. Please comment this line.")
    elif ((method == "DIPRecon" or method == "DNA") and config["rho"]["grid_search"][0] == 0 and task != "post_reco"):
        raise ValueError("Please set rho > 0 for DNA or DIPRecon reconstruction (or set task to post reconstruction).")
    elif (config["DIP_early_stopping_when"]["grid_search"][0] != "never" and (config["windowSize"]["grid_search"][0] >= config["sub_iter_DIP"]["grid_search"][0] and config["EMV_or_WMV"]["grid_search"][0] == "WMV")):

        raise ValueError("Please set window size less than number of DIP iterations for Window Moving Variance.")
    elif ((config["sub_iter_DIP_init"]["grid_search"][0] <= config["patienceNumber"]["grid_search"][0]) or (config["sub_iter_DIP"]["grid_search"][0] <= config["patienceNumber"]["grid_search"][0] and config["DIP_early_stopping_when"]["grid_search"][0] == "all")):
        raise ValueError("Please set patienceNumber higher than sub_iter_DIP")
    if ("DIP_it_if_no_ES_found" in config):
        if (config["DIP_it_if_no_ES_found"]["grid_search"][0] > config["sub_iter_DIP_init"]["grid_search"][0]):
            raise ValueError("Please set DIP_it_if_no_ES_found higher than sub_iter_DIP_init")

    #'''
    os.system("rm -rf " + root + subroot + 'suffixes_for_last_run_' + method + '.txt')
    os.system("rm -rf " + root + subroot + 'replicates_for_last_run_' + method + '.txt')

    # Launch task
    if task != "show_metrics_results_already_computed":
        classTask.runRayTune(config_tmp,root,task)
        #classTask.runRayTune(config,root,task)
    #'''

if (task != "post_reco"):
    config_without_grid_search = dict(config)
    task = 'show_metrics_results_already_computed_following_step'

    for key,value in config_without_grid_search.items():
        if (type(value) == type(config_without_grid_search)):
            if ("grid_search" in value):
                config_without_grid_search[key] = value['grid_search']
                
                if len(config_without_grid_search[key]) > 1:
                    print(key)

                #if len(config_without_grid_search[key]) == 1:
                if key != 'rho' and key != 'replicates' and key != 'method':
                    if key != 'A_AML' and key != 'post_smoothing' and key != 'lr':
                        config_without_grid_search[key] = config_without_grid_search[key][0]

    classTask = iTradeoffCurves(config_without_grid_search)
    config_without_grid_search["ray"] = False
    classTask.config_with_grid_search = config
    classTask.runRayTune(config_without_grid_search,root,task)

'''
classTask = iResultsADMMReg_VS_APPGML(config_without_grid_search)
config_without_grid_search["ray"] = False
classTask.config_with_grid_search = config
classTask.runRayTune(config_without_grid_search,root,task)
'''
#sys.stdout.close()
#sys.stdout=stdoutOrigin