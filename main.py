#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: alexandre (2021-2022)
"""

## Python libraries
# Useful
import os
from ray import tune

# Local files to import
from iNestedADMM import iNestedADMM
from iComparison import iComparison
from iPostReconstruction import iPostReconstruction
from iResults import iResults

# Configuration dictionnary for general parameters (not hyperparameters)
fixed_config = {
    "image" : tune.grid_search(['image0']),
    "net" : tune.grid_search(['DIP']), # Network to use (DIP,DD,DD_AE,DIP_VAE)
    "method" : tune.grid_search(['ADMMLim','AML']),
    "processing_unit" : tune.grid_search(['CPU']),
    "max_iter" : tune.grid_search([10]),
    "finetuning" : tune.grid_search(['last']),
    "experiment" : tune.grid_search([24]),
    "image_init_path_without_extension" : tune.grid_search(['1_im_value_cropped']),
    #"f_init" : tune.grid_search(['1_im_value_cropped']),
    "penalty" : tune.grid_search(['DIP_ADMM']),
    "NNEPPS" : tune.grid_search([False]),
}
# Configuration dictionnary for hyperparameters to tune
hyperparameters_config = {
    "rho" : tune.grid_search([0]), # Penalty strength (beta) in MAP algorithms 
    ## network hyperparameters
    "lr" : tune.grid_search([0.01]), # 0.01 for DIP, 0.001 for DD
    "sub_iter_DIP" : tune.grid_search([10]), # 10 for DIP, 100 for DD
    "opti_DIP" : tune.grid_search(['Adam']),
    "skip_connections" : tune.grid_search([3]),
    "scaling" : tune.grid_search(['standardization']),
    "input" : tune.grid_search(['random']),
    "d_DD" : tune.grid_search([4]), # not above 4, otherwise 112 is too little as output size / not above 6, otherwise 128 is too little as output size
    "k_DD" : tune.grid_search([32]),
    ## ADMMLim hyperparameters
    "sub_iter_MAP" : tune.grid_search([10]), # Block 1 iterations (Sub-problem 1 - MAP) if mlem_sequence is False
    "nb_iter_second_admm": tune.grid_search([10]), # Number of ADMM iterations (ADMM before NN)
    "alpha" : tune.grid_search([0.005]), # alpha from ADMM in ADMMLim
    ## hyperparameters from CASToR algorithms 
    # Optimization transfer (OPTITR) hyperparameters
    "mlem_sequence" : tune.grid_search([True]),
    # AML hyperparameters
    "A_AML" : tune.grid_search([-100,-500,-1000,-1e4,-1e5,-1e6]),
}

# Merge 2 dictionaries
split_config = {
    "hyperparameters" : list(hyperparameters_config.keys())
}
config = {**fixed_config, **hyperparameters_config, **split_config}

root = os.getcwd()

# Choose task to do (move this after raytune !!!)
if (config["method"] == 'Gong' or config["method"] == 'nested'):
    task = 'full_reco_with_network'

elif (config["method"] == 'ADMMLim' or config["method"] == 'MLEM' or config["method"] == 'BSREM'):
    task = 'castor_reco'

#task = 'full_reco_with_network'
task = 'castor_reco'
#task = 'post_reco'
#task = 'show_results'

if (task == 'full_reco_with_network'): # Run Gong or nested ADMM
    classTask = iNestedADMM(hyperparameters_config)
elif (task == 'castor_reco'): # Run CASToR reconstruction with given optimizer
    classTask = iComparison(config)
elif (task == 'post_reco'): # Run network denoising after a given reconstructed image im_corrupt
    classTask = iPostReconstruction(config)
elif (task == 'show_results'): # Show already computed results
    classTask = iResults(config)#,config["max_iter"],config["PETImage_shape"],config["phantom"],config["subroot"],config["suffix"],config["net"])

# Launch task
classTask.runRayTune(config,root)