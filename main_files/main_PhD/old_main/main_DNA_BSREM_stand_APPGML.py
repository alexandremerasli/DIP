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

# Configuration dictionnary for general settings parameters (not hyperparameters)
settings_config = {
    "image" : tune.grid_search(['image4_0']), # Image from database
    "random_seed" : tune.grid_search([True]), # If True, random seed is used for reproducibility (must be set to False to vary weights initialization)
    "method" : tune.grid_search(["DNA"]), # Reconstruction algorithm (DNA, DIPRecon, or algorithms from CASToR (MLEM, BSREM, AML, etc.))
    "processing_unit" : tune.grid_search(['CPU']), # CPU or GPU
    "nb_threads" : tune.grid_search([1]), # Number of desired threads. 0 means all the available threads
    "FLTNB" : tune.grid_search(['float']), # FLTNB precision must be set as in CASToR (double necessary for ADMMReg and DNA)
    "debug" : False, # Debug mode = run without raytune and with one iteration
    "ray" : True, # Ray mode = run with raytune if True, to run several settings in parallel
    "tensorboard" : False, # Tensorboard mode = show results in tensorboard
    "all_images_DIP" : tune.grid_search(['Last']), # Option to store only 10 images like in tensorboard (quicker, for visualization, set it to "True" by default). Can be set to "True", "False", "Unique" (store only last image)
    "experiment" : tune.grid_search([24]),
    "replicates" : tune.grid_search(list(range(1,100+1))), # List of desired replicates. list(range(1,n+1)) means n replicates
    "replicates" : tune.grid_search([26]), # List of desired replicates. list(range(1,n+1)) means n replicates
    "replicates" : tune.grid_search([35,36,37,38,39,40]), # List of desired replicates. list(range(1,n+1)) means n replicates
     #"replicates" : tune.grid_search(list(range(1,40+1))), # List of desired replicates. list(range(1,n+1)) means n replicates
    "average_replicates" : tune.grid_search([False]), # List of desired replicates. list(range(1,n+1)) means n replicates
    "castor_foms" : tune.grid_search([True]), # Set to True to compute CASToR Figure Of Merits (likelihood, residuals for ADMMReg)
}
# Configuration dictionnary for previous hyperparameters, but fixed to simplify
fixed_config = {
    "max_iter" : tune.grid_search([100]), # Number of global iterations for usual optimizers (MLEM, BSREM, AML etc.) and for DNA and DIPRecon
    "nb_subsets" : tune.grid_search([28]), # Number of subsets in chosen reconstruction algorithm (automatically set to 1 for ADMMReg)
    "finetuning" : tune.grid_search(['last']),
    "penalty" : tune.grid_search(['MRF']), # Penalty used in CASToR for PLL algorithms
    "unnested_1st_global_iter" : tune.grid_search([False]), # If True, unnested are computed after 1st global iteration (because rho is set to 0). If False, needs to set f_init to initialize the network, as in DIPRecon paper, and rho is not changed.
    "sub_iter_DIP_initial_and_final" : tune.grid_search([1000]), # Number of epochs in first global iteration (pre iteraiton) in network optimization (only for DIPRecon for now)
    "nb_inner_iteration" : tune.grid_search([1]), # Number of inner iterations in ADMMReg (if mlem_sequence is False). (3 sub iterations are done within 1 inner iteration in CASToR)
    "xi" : tune.grid_search([1]), # Factor to balance primal and dual residual convergence speed in adaptive tau computation in ADMMReg
    "xi_DIP" : tune.grid_search([1]), # Factor to balance primal and dual residual convergence speed in adaptive tau computation in DIPRecon and DNA
    "net" : tune.grid_search(['DIP']), # Network to use (DIP,DD,DD_AE,DIP_VAE)
    "DIP_early_stopping" : tune.grid_search([False]), # Use DIP early stopping with moving variance strategy
    "EMV_or_WMV" : tune.grid_search(["EMV"]), # Use DIP early stopping with WMV or EMV
    "alpha_EMV" : tune.grid_search([0.1]), # EMV forgetting factor alpha
    "windowSize" : tune.grid_search([50]), # Network to use (DIP,DD,DD_AE,DIP_VAE)
    "patienceNumber" : tune.grid_search([100]), # Network to use (DIP,DD,DD_AE,DIP_VAE)
    "recoInDNA" : tune.grid_search(["APPGML"]), # Which algorithm to use in DNA (ADMMReg or APPGML)
}
# Configuration dictionnary for hyperparameters to tune
hyperparameters_config = {
    "image_init_path_without_extension" : tune.grid_search(['BSREM_it30']), # Initial image of the reconstruction algorithm (taken from subroot + "/Data/initialization")
    "rho" : tune.grid_search([0.003,8e-4,0.008,0.03]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (DNA and DIPRecon)
    "rho" : tune.grid_search([0.003]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (DNA and DIPRecon)
    "adaptive_parameters_DIP" : tune.grid_search(["nothing"]), # which parameters are adaptive ? Must be set to nothing, alpha, or tau (which means alpha and tau)
    "mu_DIP" : tune.grid_search([10]), # Factor to balance primal and dual residual in adaptive alpha computation in ADMMReg
    "tau_DIP" : tune.grid_search([2]), # Factor to multiply alpha in adaptive alpha computation in ADMMReg. If adaptive tau, it corresponds to tau max
    ## network hyperparameters
    "lr" : tune.grid_search([0.01]), # Learning rate in network optimization
    "sub_iter_DIP" : tune.grid_search([100]), # Number of epochs in network optimization
    "opti_DIP" : tune.grid_search(['Adam']), # Optimization algorithm in neural network training (Adam, LBFGS)
    "skip_connections" : tune.grid_search([3]), # Number of skip connections in DIP architecture (0, 1, 2, 3)
    "scaling" : tune.grid_search(['standardization']), # Pre processing of neural network input (nothing, uniform, normalization, standardization)
    "input" : tune.grid_search(['random']), # Neural network input (random or anatomical )
    #"input" : tune.grid_search(["anatomical",'random']), # Neural network input (random or anatomical )
    "d_DD" : tune.grid_search([4]), # d for Deep Decoder, number of upsampling layers. Not above 4, otherwise 112 is too little as output size / not above 6, otherwise 128 is too little as output size
    "k_DD" : tune.grid_search([32]), # k for Deep Decoder
    ## ADMMReg - OPTITR hyperparameters
    "nb_outer_iteration": tune.grid_search([30]), # Number of outer iterations in ADMMReg (and DNA) and OPTITR (for DIPRecon)
    #"nb_outer_iteration": tune.grid_search([3]), # Number of outer iterations in ADMMReg (and DNA) and OPTITR (for DIPRecon)
    "nb_outer_iteration": tune.grid_search([1]), # Number of outer iterations in ADMMReg (and DNA) and OPTITR (for DIPRecon)
    "alpha" : tune.grid_search([1]), # alpha (penalty parameter) in ADMMReg
    "adaptive_parameters" : tune.grid_search(["both"]), # which parameters are adaptive ? Must be set to nothing, alpha, or both (which means alpha and tau)
    "mu_adaptive" : tune.grid_search([2]), # Factor to balance primal and dual residual in adaptive alpha computation in ADMMReg
    "tau" : tune.grid_search([100]), # Factor to multiply alpha in adaptive alpha computation in ADMMReg
    "tau_max" : tune.grid_search([100]), # Maximum value for tau in adaptive tau in ADMMReg
    "stoppingCriterionValue" : tune.grid_search([0]), # Value of the stopping criterion in ADMMReg
    "saveSinogramsUAndV" : tune.grid_search([0]), # 1 means save sinograms u and v from CASToR, otherwise it means do not save them. If adaptive tau, it corresponds to tau max
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

    '''
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
    config_tmp["method"] = tune.grid_search([method]) # Put only 1 method to remove useless hyperparameters from settings_config and hyperparameters_config

    '''
    if (method == 'BSREM'):
        config_tmp["rho"]['grid_search'] = [0.01,0.02,0.03,0.04,0.05]

    if (method == "DIPRecon"):
        config_tmp["nb_inner_iteration"]['grid_search'] = [50]
        #config_tmp["lr"]['grid_search'] = [0.5]
        #config_tmp["rho"]['grid_search'] = [0.0003]
        config_tmp["lr"]['grid_search'] = [0.5]
        config_tmp["rho"]['grid_search'] = [0.0003]
    elif (method == 'DNA'):
        config_tmp["nb_inner_iteration"]['grid_search'] = [10]
        #config_tmp["lr"]['grid_search'] = [0.01] # super DNA
        #config_tmp["rho"]['grid_search'] = [0.003] # super DNA
        config_tmp["lr"]['grid_search'] = [0.05]
        config_tmp["rho"]['grid_search'] = [0.0003]
    '''

    # Choose task to do (move this after raytune !!!)
    if (method == "DIPRecon" or method == "DNA"):
        task = 'full_reco_with_network'

    elif ('ADMMReg' in method or method == 'MLEM' or method == 'OPTITR' or method == 'OSEM' or method == 'BSREM' or method == 'AML' or method == 'APPGML'):
        task = 'castor_reco'

    #task = 'full_reco_with_network' # Run DIPRecon or DNA
    #task = 'castor_reco' # Run CASToR reconstruction with given optimizer
    #task = 'post_reco' # Run network denoising after a given reconstructed image im_corrupt
    #task = 'show_results_post_reco'
    #task = 'show_results'
    #task = 'show_metrics_results_already_computed'
    #task = 'show_metrics_ADMMReg'
    #task = 'show_metrics_DNA'
    #task = 'compare_2_methods'

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
        config["average_replicates"] = tune.grid_search([True])
        classTask = iResultsADMMReg_VS_APPGML(config)

    # Incompatible parameters (should be written in vGeneral I think)
    if (method == "DNA" and config["rho"]["grid_search"][0] == 0 and task == "castor_reco"):
        raise ValueError("DNA must be launched with rho > 0")
    elif ((method != "DIPRecon" and method != "DNA") and task == "post_reco"):
        raise ValueError("Only DIPRecon or DNA can be run in post reconstruction mode, not CASToR reconstruction algorithms. Please comment this line.")
    elif ((method == "DIPRecon" or method == "DNA") and config["all_images_DIP"]["grid_search"][0] != "True" and config["DIP_early_stopping"]["grid_search"][0] == "True"):
        raise ValueError("Please set all_images_DIP to True to save all images for DNA or DIPRecon reconstruction if using moving variance algorithms")
    elif ((method == "DIPRecon" or method == "DNA") and config["rho"]["grid_search"][0] == 0 and task != "post_reco"):
        raise ValueError("Please set rho > 0 for DNA or DIPRecon reconstruction (or set task to post reconstruction).")
    elif (config["windowSize"]["grid_search"][0] >= config["sub_iter_DIP"]["grid_search"][0] and config["DIP_early_stopping"]["grid_search"][0]):
        raise ValueError("Please set window size less than number of DIP iterations for Window Moving Variance.")
    elif (config["debug"] and config["ray"]):
        raise ValueError("Debug mode must is used without ray")
    elif (task == "post_reco" and config["DIP_early_stopping"]["grid_search"][0] == True and config["all_images_DIP"]["grid_search"][0] == "False"):
        raise ValueError("post reco mode need to save all images if ES")

    #'''
    os.system("rm -rf " + root + subroot + 'suffixes_for_last_run_' + method + '.txt')
    os.system("rm -rf " + root + subroot + 'replicates_for_last_run_' + method + '.txt')

    # Launch task
    classTask.runRayTune(config_tmp,root,task)
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
classTask.runRayTune(config_without_grid_search,root,task)
'''
#sys.stdout.close()
#sys.stdout=stdoutOrigin