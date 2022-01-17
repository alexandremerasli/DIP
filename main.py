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
from utils.utils_func import *
from iNestedADMM import iNestedADMM
from iPostReconstruction import iPostReconstruction

# Configuration dictionnary for general parameters (not hyperparameters)
fixed_config = {
    "image" : tune.grid_search(['image0']),
    "net" : tune.grid_search(['DIP']), # Network to use (DIP,DD,DD_AE,DIP_VAE)
    "method" : tune.grid_search(['nested']),
    "processing_unit" : tune.grid_search(['CPU']),
    "max_iter" : tune.grid_search([10]),
    "finetuning" : tune.grid_search(['last']),
    "test" : tune.grid_search([24])
}
# Configuration dictionnary for hyperparameters to tune
hyperparameters_config = {
    "rho" : tune.grid_search([0.0003]),
    # network hyperparameters
    "lr" : tune.grid_search([0.01]), # 0.01 for DIP, 0.001 for DD
    "sub_iter_DIP" : tune.grid_search([100]), # 10 for DIP, 100 for DD
    "opti_DIP" : tune.grid_search(['Adam']),
    "skip_connections" : tune.grid_search([1]),
    "scaling" : tune.grid_search(['standardization']),
    "input" : tune.grid_search(['random']),
    "d_DD" : tune.grid_search([4]), # not above 4, otherwise 112 is too little as output size / not above 6, otherwise 128 is too little as output size
    "k_DD" : tune.grid_search([32]),
    # ADMMLim hyperparameters
    "sub_iter_MAP" : tune.grid_search([10]), # Block 1 iterations (Sub-problem 1 - MAP) if mlem_sequence is False
    "nb_iter_second_admm": tune.grid_search([10
    ]), # Number of ADMM iterations (ADMM before NN)
    "mlem_sequence" : tune.grid_search([False]),
    "alpha" : tune.grid_search([0.005])
}

# Merge 2 dictionaries
split_config = {
    "hyperparameters" : list(hyperparameters_config.keys())
}
config = {**fixed_config, **hyperparameters_config, **split_config}

# Initializer images and variables
root = os.getcwd()
classReco = iNestedADMM(hyperparameters_config,root)
#classReco = iPostReconstruction(config,root)
classReco.runRayTune(config,root)