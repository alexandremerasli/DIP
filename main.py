#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: alexandre (2021-2022)
"""

## Python libraries
# Useful
from cProfile import run
import os
from ray import tune

# Local files to import
from iNestedADMM import iNestedADMM
from iComparison import iComparison
from iPostReconstruction import iPostReconstruction
from iResults import iResults
from iResultsReplicates import iResultsReplicates

# Configuration dictionnary for general parameters (not hyperparameters)
fixed_config = {
    "image" : tune.grid_search(['image0']),
    "net" : tune.grid_search(['DIP']), # Network to use (DIP,DD,DD_AE,DIP_VAE)
    "method" : tune.grid_search(['AML']),
    "processing_unit" : tune.grid_search(['CPU']),
    "max_iter" : tune.grid_search([10]),
    "finetuning" : tune.grid_search(['last']),
    "experiment" : tune.grid_search([24]),
    "image_init_path_without_extension" : tune.grid_search(['1_im_value_cropped']),
    #"f_init" : tune.grid_search(['1_im_value_cropped']),
    "penalty" : tune.grid_search(['DIP_ADMM']),
    "replicates" : tune.grid_search(list(range(1,10+1))),
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
    "A_AML" : tune.grid_search([0,-100,-500]),
    #"A_AML" : tune.grid_search([0]),
    # NNEPPS post processing
    "NNEPPS" : tune.grid_search([True]),
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
#task = 'castor_reco'
#task = 'post_reco'
task = 'show_results'
#task = 'show_results_replicates'

if (task == 'full_reco_with_network'): # Run Gong or nested ADMM
    classTask = iNestedADMM(hyperparameters_config)
elif (task == 'castor_reco'): # Run CASToR reconstruction with given optimizer
    classTask = iComparison(config)
elif (task == 'post_reco'): # Run network denoising after a given reconstructed image im_corrupt
    classTask = iPostReconstruction(config)
elif (task == 'show_results'): # Show already computed results over iterations
    classTask = iResults(config)
elif (task == 'show_results_replicates'): # Show already computed results averaging over replicates
    classTask = iResultsReplicates(config)

# Launch task
os.system("rm -rf " + root + '/data/Algo/' + 'suffixes_for_last_run.txt')
classTask.runRayTune(config,root,task)

PSNR_recon = []
PSNR_norm_recon = []
MSE_recon = []
MA_cold_recon = []
CRC_hot_recon = []
CRC_bkg_recon = []
IR_bkg_recon = [] 

from csv import reader as reader_csv
import numpy as np
import matplotlib.pyplot as plt
suffixes = []
with open(root + '/data/Algo/' + '/suffixes_for_last_run.txt') as f:
    suffixes.append(f.readlines())

print(suffixes) 
print(config["method"]['grid_search'][0])
# Load metrics from last runs to merge them in one figure
if config["method"]['grid_search'][0] == "AML":
    A_AML_list = config["A_AML"]['grid_search']
NNEPPS_list = config["NNEPPS"]['grid_search']
if config["method"]['grid_search'][0] == "AML": 
    #for beta in A_AML_list:
        #for NNEPPS in NNEPPS_list:
    for suffix in suffixes[0]:
        metrics_file = root + '/data/Algo/' + '/metrics/' + config["method"]['grid_search'][0] + '/' + suffix.rstrip("\n") + '/' + 'CRC in hot region vs IR in background.csv'
        with open(metrics_file, 'r') as myfile:
            spamreader = reader_csv(myfile,delimiter=';')
            rows_csv = list(spamreader)
            rows_csv[0] = [float(i) for i in rows_csv[0]]
            rows_csv[1] = [float(i) for i in rows_csv[1]]
            rows_csv[2] = [float(i) for i in rows_csv[2]]
            rows_csv[3] = [float(i) for i in rows_csv[3]]
            rows_csv[4] = [float(i) for i in rows_csv[4]]
            rows_csv[5] = [float(i) for i in rows_csv[5]]
            rows_csv[6] = [float(i) for i in rows_csv[6]]

            PSNR_recon.append(np.array(rows_csv[0]))
            PSNR_norm_recon.append(np.array(rows_csv[1]))
            MSE_recon.append(np.array(rows_csv[2]))
            MA_cold_recon.append(np.array(rows_csv[3]))
            CRC_hot_recon.append(np.array(rows_csv[4]))
            CRC_bkg_recon.append(np.array(rows_csv[5]))
            IR_bkg_recon.append(np.array(rows_csv[6]))

            print(PSNR_recon)
            print(PSNR_norm_recon)
            print(MSE_recon)
            print(MA_cold_recon)
            print(CRC_hot_recon)
            print(CRC_bkg_recon)
            print(IR_bkg_recon)



    for run_id in range(len(PSNR_recon)):
        print(IR_bkg_recon[run_id])
        plt.plot(IR_bkg_recon[run_id],CRC_hot_recon[run_id],linestyle='None',marker='x')
    
    plt.xlabel('IR')
    plt.ylabel('CRC')
    # Saving this figure locally
    plt.savefig(root + '/data/Algo/' + 'replicate_0/Images/tmp/' + 'CRC in hot region vs IR in background' + '.png')
    from textwrap import wrap
    wrapped_title = "\n".join(wrap(suffix, 50))
    plt.title(wrapped_title,fontsize=12)

        # + '_beta_' + beta + '/' + 'CRC in hot region vs IR in background.csv'
#elif config["method"]['grid_search'][0] == "ADMMLim":
#    metrics_path = root + 'metrics/' + 'Comparison/' + config["method"]['grid_search'][0] + '/' + suffix + '/' 
