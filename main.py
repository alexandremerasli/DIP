#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: alexandre (2021-2022)
"""

## Python libraries
# Useful
import os
import argparse
from functools import partial
from ray import tune

# Local files to import
from utils.utils_func import *
from iNestedADMM import iNestedADMM
from iPostReconstruction import iPostReconstruction

# Configuration dictionnary for hyperparameters to tune
#'''
config = {
    "image" : tune.grid_search(['image0']),
    "net" : tune.grid_search(['DIP']), # Network to use (DIP,DD,DIP_VAE)
    "method" : tune.grid_search(['nested']),
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
    "nb_iter_second_admm": tune.grid_search([10]), # Number of ADMM iterations (ADMM before NN)
    "mlem_sequence" : tune.grid_search([False]),
    "alpha" : tune.grid_search([0.005])
}
#'''

## Arguments for linux command to launch script
# Creating arguments
parser = argparse.ArgumentParser(description='DIP + ADMM computation')
parser.add_argument('--proc', type=str, dest='proc', help='processing unit (CPU, GPU or both)')
parser.add_argument('--max_iter', type=int, dest='max_iter', help='number of outer iterations')
parser.add_argument('--finetuning', type=str, dest='finetuning', help='finetuning or not for the DIP optimizations', nargs='?', const='False')

# Retrieving arguments in this python script
args = parser.parse_args()

# For VS Code (without command line)
if (args.proc is None): # Must check if all args are None
    args.proc = 'CPU'
    args.max_iter = 10 # Outer iterations
    args.finetuning = 'last' # Finetuning or not for the DIP optimizations (block 2)





# Initializer images and variables
root = os.getcwd()
classReco = iNestedADMM(config,args,root)
#classReco = iPostReconstruction(config,args,root)
classReco.runRayTune(config,args,root)