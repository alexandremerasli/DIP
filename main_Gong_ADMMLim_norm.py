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
    "image" : tune.grid_search(['image2_0']), # Image from database
    "random_seed" : tune.grid_search([True]), # If True, random seed is used for reproducibility (must be set to False to vary weights initialization)
    "method" : tune.grid_search(['Gong']), # Reconstruction algorithm (nested, Gong, or algorithms from CASToR (MLEM, BSREM, AML, etc.))
    "processing_unit" : tune.grid_search(['CPU']), # CPU or GPU
    "nb_threads" : tune.grid_search([1]), # Number of desired threads. 0 means all the available threads
    "FLTNB" : tune.grid_search(['double']), # FLTNB precision must be set as in CASToR (double necessary for ADMMLim and nested)
    "debug" : False, # Debug mode = run without raytune and with one iteration
    "ray" : True, # Ray mode = run with raytune if True, to run several settings in parallel
    "tensorboard" : True, # Tensorboard mode = show results in tensorboard
    "all_images_DIP" : tune.grid_search(['Last']), # Option to store only 10 images like in tensorboard (quicker, for visualization, set it to "True" by default). Can be set to "True", "False", "Last" (store only last image)
    "experiment" : tune.grid_search([24]),
    "replicates" : tune.grid_search(list(range(1,100+1))), # List of desired replicates. list(range(1,n+1)) means n replicates
    #"replicates" : tune.grid_search(list(range(1,1+1))), # List of desired replicates. list(range(1,n+1)) means n replicates
    "average_replicates" : tune.grid_search([False]), # List of desired replicates. list(range(1,n+1)) means n replicates
    "castor_foms" : tune.grid_search([True]), # Set to True to compute CASToR Figure Of Merits (likelihood, residuals for ADMMLim)
}
# Configuration dictionnary for previous hyperparameters, but fixed to simplify
fixed_config = {
    "max_iter" : tune.grid_search([100]), # Number of global iterations for usual optimizers (MLEM, BSREM, AML etc.) and for nested and Gong
    "nb_subsets" : tune.grid_search([28]), # Number of subsets in chosen reconstruction algorithm (automatically set to 1 for ADMMLim)
    "finetuning" : tune.grid_search(['last']),
    "penalty" : tune.grid_search(['MRF']), # Penalty used in CASToR for PLL algorithms
    "unnested_1st_global_iter" : tune.grid_search([False]), # If True, unnested are computed after 1st global iteration (because rho is set to 0). If False, needs to set f_init to initialize the network, as in Gong paper, and rho is not changed.
    "sub_iter_DIP_initial_and_final" : tune.grid_search([1000]), # Number of epochs in first global iteration (pre iteraiton) in network optimization (only for Gong for now)
    "nb_inner_iteration" : tune.grid_search([1]), # Number of inner iterations in ADMMLim (if mlem_sequence is False). (3 sub iterations are done within 1 inner iteration in CASToR)
    "xi" : tune.grid_search([1]), # Factor to balance primal and dual residual convergence speed in adaptive tau computation in ADMMLim
    "xi_DIP" : tune.grid_search([1]), # Factor to balance primal and dual residual convergence speed in adaptive tau computation in Gong and nested
    "net" : tune.grid_search(['DIP']), # Network to use (DIP,DD,DD_AE,DIP_VAE)
    "DIP_early_stopping" : tune.grid_search([False]), # Use DIP early stopping with WMV strategy
    "windowSize" : tune.grid_search([10]), # Network to use (DIP,DD,DD_AE,DIP_VAE)
    "patienceNumber" : tune.grid_search([100]), # Network to use (DIP,DD,DD_AE,DIP_VAE)
}
# Configuration dictionnary for hyperparameters to tune
hyperparameters_config = {
    "image_init_path_without_extension" : tune.grid_search(['ADMMLim_it100']), # Initial image of the reconstruction algorithm (taken from data/algo/Data/initialization)
    "rho" : tune.grid_search([0.003,8e-4,0.008,0.03]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (nested and Gong)
    "rho" : tune.grid_search([0.003]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (nested and Gong)
    "adaptive_parameters_DIP" : tune.grid_search(["nothing"]), # which parameters are adaptive ? Must be set to nothing, alpha, or tau (which means alpha and tau)
    "mu_DIP" : tune.grid_search([10]), # Factor to balance primal and dual residual in adaptive alpha computation in ADMMLim
    "tau_DIP" : tune.grid_search([2]), # Factor to multiply alpha in adaptive alpha computation in ADMMLim. If adaptive tau, it corresponds to tau max
    ## network hyperparameters
    "lr" : tune.grid_search([0.01]), # Learning rate in network optimization
    "sub_iter_DIP" : tune.grid_search([100]), # Number of epochs in network optimization
    "opti_DIP" : tune.grid_search(['Adam']), # Optimization algorithm in neural network training (Adam, LBFGS)
    "skip_connections" : tune.grid_search([3]), # Number of skip connections in DIP architecture (0, 1, 2, 3)
    "scaling" : tune.grid_search(['positive_normalization']), # Pre processing of neural network input (nothing, uniform, normalization, standardization)
    #"scaling" : tune.grid_search(['standardization','positive_normalization']), # Pre processing of neural network input (nothing, uniform, normalization, standardization)
    "input" : tune.grid_search(['random']), # Neural network input (random or CT)
    #"input" : tune.grid_search(['CT','random']), # Neural network input (random or CT)
    "d_DD" : tune.grid_search([4]), # d for Deep Decoder, number of upsampling layers. Not above 4, otherwise 112 is too little as output size / not above 6, otherwise 128 is too little as output size
    "k_DD" : tune.grid_search([32]), # k for Deep Decoder
    ## ADMMLim - OPTITR hyperparameters
    "nb_outer_iteration": tune.grid_search([30]), # Number of outer iterations in ADMMLim (and nested) and OPTITR (for Gong)
    #"nb_outer_iteration": tune.grid_search([3]), # Number of outer iterations in ADMMLim (and nested) and OPTITR (for Gong)
    "nb_outer_iteration": tune.grid_search([2]), # Number of outer iterations in ADMMLim (and nested) and OPTITR (for Gong)
    "alpha" : tune.grid_search([1]), # alpha (penalty parameter) in ADMMLim
    "adaptive_parameters" : tune.grid_search(["both"]), # which parameters are adaptive ? Must be set to nothing, alpha, or both (which means alpha and tau)
    "mu_adaptive" : tune.grid_search([2]), # Factor to balance primal and dual residual in adaptive alpha computation in ADMMLim
    "tau" : tune.grid_search([2]), # Factor to multiply alpha in adaptive alpha computation in ADMMLim
    "tau_max" : tune.grid_search([100]), # Maximum value for tau in adaptive tau in ADMMLim
    "stoppingCriterionValue" : tune.grid_search([0.001]), # Value of the stopping criterion in ADMMLim
    "saveSinogramsUAndV" : tune.grid_search([1]), # 1 means save sinograms u and v from CASToR, otherwise it means do not save them. If adaptive tau, it corresponds to tau max
    ## hyperparameters from CASToR algorithms 
    # Optimization transfer (OPTITR) hyperparameters
    "mlem_sequence" : tune.grid_search([False]), # Given sequence (with decreasing number of subsets) to quickly converge. True or False
    # AML/APGMAP hyperparameters
    "A_AML" : tune.grid_search([-100,-500,-10000]), # AML lower bound A
    "A_AML" : tune.grid_search([-10,-100]), # AML lower bound A
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

# write random seed in a file to get it in network architectures
os.system("rm -rf " + os.getcwd() +"/seed.txt")
file_seed = open(os.getcwd() + "/seed.txt","w+")
file_seed.write(str(settings_config["random_seed"]["grid_search"][0]))
file_seed.close()

# Local files to import, AFTER CONFIG TO SET RANDOM SEED OR NOT
from iNestedADMM import iNestedADMM
from iComparison import iComparison
from iPostReconstruction import iPostReconstruction
from iResults import iResults
from iMeritsADMMLim import iMeritsADMMLim
from iMeritsNested import iMeritsNested
from iResultsAlreadyComputed import iResultsAlreadyComputed
from iResultsADMMLim_VS_APGMAP import iResultsADMMLim_VS_APGMAP
from iFinalCurves import iFinalCurves

for method in config["method"]['grid_search']:

    '''
    # Gong reconstruction
    if (config["method"]["grid_search"][0] == 'Gong' and len(config["method"]["grid_search"]) == 1):
        print("configuration fiiiiiiiiiiiiiiiiiiile")
        #config = np.load(root + 'config_DIP.npy',allow_pickle='TRUE').item()
        from Gong_configuration import config_func_MIC
        #config = config_func()
        config = config_func_MIC()

    # nested reconstruction
    if (config["method"]["grid_search"][0] == 'nested' and len(config["method"]["grid_search"]) == 1):
        print("configuration fiiiiiiiiiiiiiiiiiiile")
        from nested_configuration import config_func_MIC
        #config = config_func()
        config = config_func_MIC()

    # MLEM reconstruction
    if (config["method"]["grid_search"][0] == 'MLEM' and len(config["method"]["grid_search"]) == 1):
        print("configuration fiiiiiiiiiiiiiiiiiiile")
        from MLEM_configuration import config_func_MIC
        #config = config_func()
        config = config_func_MIC()

    # OSEM reconstruction
    if (config["method"]["grid_search"][0] == 'OSEM' and len(config["method"]["grid_search"]) == 1):
        print("configuration fiiiiiiiiiiiiiiiiiiile")
        from OSEM_configuration import config_func_MIC
        #config = config_func()
        config = config_func_MIC()

    # BSREM reconstruction
    if (config["method"]["grid_search"][0] == 'BSREM' and len(config["method"]["grid_search"]) == 1):
        print("configuration fiiiiiiiiiiiiiiiiiiile")
        from BSREM_configuration import config_func_MIC
        #config = config_func()
        config = config_func_MIC()

    # APGMAP reconstruction
    if ('APGMAP' in config["method"]["grid_search"][0] and len(config["method"]["grid_search"]) == 1):
        print("configuration fiiiiiiiiiiiiiiiiiiile")
        from APGMAP_configuration import config_func_MIC
        #config = config_func()
        config = config_func_MIC()

    # ADMMLim reconstruction
    if (config["method"]["grid_search"][0] == 'ADMMLim' and len(config["method"]["grid_search"]) == 1):
        print("configuration fiiiiiiiiiiiiiiiiiiile")
        from ADMMLim_configuration import config_func_MIC
        #config = config_func()
        config = config_func_MIC()
    '''
    config_tmp = dict(config)
    config_tmp["method"] = tune.grid_search([method]) # Put only 1 method to remove useless hyperparameters from settings_config and hyperparameters_config

    '''
    if (method == 'BSREM'):
        config_tmp["rho"]['grid_search'] = [0.01,0.02,0.03,0.04,0.05]

    if (method == 'Gong'):
        config_tmp["nb_inner_iteration"]['grid_search'] = [50]
        #config_tmp["lr"]['grid_search'] = [0.5]
        #config_tmp["rho"]['grid_search'] = [0.0003]
        config_tmp["lr"]['grid_search'] = [0.5]
        config_tmp["rho"]['grid_search'] = [0.0003]
    elif (method == 'nested'):
        config_tmp["nb_inner_iteration"]['grid_search'] = [10]
        #config_tmp["lr"]['grid_search'] = [0.01] # super nested
        #config_tmp["rho"]['grid_search'] = [0.003] # super nested
        config_tmp["lr"]['grid_search'] = [0.05]
        config_tmp["rho"]['grid_search'] = [0.0003]
    '''

    # Choose task to do (move this after raytune !!!)
    if (config["method"]["grid_search"][0] == 'Gong' or config["method"]["grid_search"][0] == 'nested'):
        task = 'full_reco_with_network'

    elif ('ADMMLim' in config["method"]["grid_search"][0] or config["method"]["grid_search"][0] == 'MLEM' or config["method"]["grid_search"][0] == 'OPTITR' or config["method"]["grid_search"][0] == 'OSEM' or config["method"]["grid_search"][0] == 'BSREM' or config["method"]["grid_search"][0] == 'AML' or config["method"]["grid_search"][0] == 'APGMAP'):
        task = 'castor_reco'

    #task = 'full_reco_with_network' # Run Gong or nested ADMM
    #task = 'castor_reco' # Run CASToR reconstruction with given optimizer
    #task = 'post_reco' # Run network denoising after a given reconstructed image im_corrupt
    #task = 'show_results_post_reco'
    #task = 'show_results'
    #task = 'show_metrics_results_already_computed'
    #task = 'show_metrics_ADMMLim'
    #task = 'show_metrics_nested'
    #task = 'compare_2_methods'

    if (task == 'full_reco_with_network'): # Run Gong or nested ADMM
        classTask = iNestedADMM(hyperparameters_config)
    elif (task == 'castor_reco'): # Run CASToR reconstruction with given optimizer
        classTask = iComparison(config)
    elif (task == 'post_reco'): # Run network denoising after a given reconstructed image im_corrupt
        classTask = iPostReconstruction(config)
    elif (task == 'show_results'): # Show already computed results over iterations
        classTask = iResults(config)
    elif (task == 'show_results_post_reco'): # Show already computed results over iterations of post reconstruction mode
        config["task"] = "show_results_post_reco"
        classTask = iResults(config)
    elif (task == 'show_metrics_ADMMLim'): # Show ADMMLim FOMs over iterations
        classTask = iMeritsADMMLim(config)
    elif (task == 'show_metrics_nested'): # Show nested or Gong FOMs over iterations
        classTask = iMeritsNested(config)
    elif (task == 'show_metrics_results_already_computed'): # Show already computed results averaging over replicates
        classTask = iResultsAlreadyComputed(config)
    elif (task == 'compare_2_methods'): # Show already computed results averaging over replicates
        config["average_replicates"] = tune.grid_search([True])
        classTask = iResultsADMMLim_VS_APGMAP(config)

    # Incompatible parameters (should be written in vGeneral I think)
    if (config["method"]["grid_search"][0] == 'nested' and config["rho"]["grid_search"][0] == 0 and task == "castor_reco"):
        raise ValueError("nested must be launched with rho > 0")
    elif ((config["method"]["grid_search"][0] != 'Gong' and config["method"]["grid_search"][0] != 'nested') and task == "post_reco"):
        raise ValueError("Only Gong or nested can be run in post reconstruction mode, not CASToR reconstruction algorithms. Please comment this line.")
    elif ((config["method"]["grid_search"][0] == 'Gong' or config["method"]["grid_search"][0] == 'nested') and config["all_images_DIP"]["grid_search"][0] != "True" and config["DIP_early_stopping"]["grid_search"][0] == "True"):
        raise ValueError("Please set all_images_DIP to True to save all images for nested or Gong reconstruction if using WMV.")
    elif ((config["method"]["grid_search"][0] == 'Gong' or config["method"]["grid_search"][0] == 'nested') and config["rho"]["grid_search"][0] == 0 and task != "post_reco"):
        raise ValueError("Please set rho > 0 for nested or Gong reconstruction (or set task to post reconstruction).")
    elif (config["windowSize"]["grid_search"][0] >= config["sub_iter_DIP"]["grid_search"][0] and config["DIP_early_stopping"]["grid_search"][0]):
        raise ValueError("Please set window size less than number of DIP iterations for Window Moving Variance.")
    elif (config["debug"] and config["ray"]):
        raise ValueError("Debug mode must is used without ray")
    elif (task == "post_reco" and config["DIP_early_stopping"]["grid_search"][0] == True and config["all_images_DIP"]["grid_search"][0] == "False"):
        raise ValueError("post reco mode need to save all images if ES")

    #'''
    os.system("rm -rf " + root + '/data/Algo/' + 'suffixes_for_last_run_' + method + '.txt')
    os.system("rm -rf " + root + '/data/Algo/' + 'replicates_for_last_run_' + method + '.txt')

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

    classTask = iFinalCurves(config_without_grid_search)
    config_without_grid_search["ray"] = False
    classTask.runRayTune(config_without_grid_search,root,task)

'''
classTask = iResultsADMMLim_VS_APGMAP(config_without_grid_search)
config_without_grid_search["ray"] = False
classTask.runRayTune(config_without_grid_search,root,task)
'''
#sys.stdout.close()
#sys.stdout=stdoutOrigin