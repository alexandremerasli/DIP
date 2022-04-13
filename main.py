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
from iResultsAlreadyComputed import iResultsAlreadyComputed

# Configuration dictionnary for general parameters (not hyperparameters)
fixed_config = {
    "image" : tune.grid_search(['image0']), # Image from database
    "net" : tune.grid_search(['DIP']), # Network to use (DIP,DD,DD_AE,DIP_VAE)
    "method" : tune.grid_search(['Gong']), # Reconstruction algorithm (nested, Gong, or algorithms from CASToR (MLEM, BSREM, AML, etc.))
    "processing_unit" : tune.grid_search(['CPU']), # CPU or GPU
    "nb_threads" : tune.grid_search([64]), # Number of desired threads. 0 means all the available threads
    "debug" : True, # Debug mode = run without raytune and with one iteration
    "max_iter" : tune.grid_search([10]), # 
    "nb_subsets" : tune.grid_search([28]), # Number of subsets in chosen reconstruction algorithm (automatically set to 1 for ADMMLim)
    "finetuning" : tune.grid_search(['last']),
    "experiment" : tune.grid_search([24]),
    "image_init_path_without_extension" : tune.grid_search(['1_im_value_cropped']), # Initial image of the reconstruction algorithm (taken from data/algo/Data/initialization)
    #"f_init" : tune.grid_search(['1_im_value_cropped']),
    "penalty" : tune.grid_search(['MRF']), # Penalty used in CASToR for PLL algorithms
    "post_smoothing" : tune.grid_search([True]), # Post smoothing by CASToR after reconstruction
    "replicates" : tune.grid_search(list(range(1,1+1))), # List of desired replicates. list(range(1,n+1)) means n replicates
}
# Configuration dictionnary for hyperparameters to tune
hyperparameters_config = {
    "rho" : tune.grid_search([0.003]), # Penalty strength (beta) in PLL algorithms
    ## network hyperparameters
    "lr" : tune.grid_search([0.041]), # 0.01 for DIP, 0.001 for DD
    "sub_iter_DIP" : tune.grid_search([200]), # 10 for DIP, 100 for DD
    "opti_DIP" : tune.grid_search(['Adam']), # Optimization algorithm in neural network training (Adam, LBFGS)
    "skip_connections" : tune.grid_search([0]), # Number of skip connections in DIP architecture (0, 1, 2, 3)
    "scaling" : tune.grid_search(['standardization']), # Pre processing of neural network input (nothing, uniform, normalization, standardization)
    "input" : tune.grid_search(['CT']), # Neural network input (random or CT)
    "d_DD" : tune.grid_search([4]), # d for Deep Decoder, number of upsampling layers. Not above 4, otherwise 112 is too little as output size / not above 6, otherwise 128 is too little as output size
    "k_DD" : tune.grid_search([32]), # k for Deep Decoder
    ## ADMMLim hyperparameters
    "sub_iter_PLL" : tune.grid_search([10]), # Number of inner iterations in ADMMLim (if mlem_sequence is False)
    "nb_iter_second_admm": tune.grid_search([10]), # Number outer iterations in ADMMLim
    "alpha" : tune.grid_search([0.0005,0.005,0.05,0.5]), # alpha (penalty parameter) in ADMMLim
    "alpha" : tune.grid_search([0.005]), # alpha (penalty parameter) in ADMMLim
    ## hyperparameters from CASToR algorithms 
    # Optimization transfer (OPTITR) hyperparameters
    "mlem_sequence" : tune.grid_search([True]), # Given sequence (with decreasing number of subsets) to quickly converge. True or False
    # AML hyperparameters
    "A_AML" : tune.grid_search([-10000,-500,-100]), # AML lower bound A
    # NNEPPS post processing
    "NNEPPS" : tune.grid_search([True,False]), # NNEPPS post-processing. True or False
    "NNEPPS" : tune.grid_search([False]),
}

# Merge 2 dictionaries
split_config = {
    "hyperparameters" : list(hyperparameters_config.keys())
}
config = {**fixed_config, **hyperparameters_config, **split_config}

root = os.getcwd()

# Choose task to do (move this after raytune !!!)
if (config["method"]["grid_search"][0] == 'Gong' or config["method"]["grid_search"][0] == 'nested'):
    task = 'full_reco_with_network'

elif (config["method"]["grid_search"][0] == 'ADMMLim' or config["method"]["grid_search"][0] == 'MLEM' or config["method"]["grid_search"][0] == 'BSREM' or config["method"]["grid_search"][0] == 'AML'):
    task = 'castor_reco'

#task = 'full_reco_with_network'
#task = 'castor_reco'
#task = 'post_reco'
#task = 'show_results'
#task = 'show_results_replicates'
#task = 'show_metrics_results_already_computed'

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
elif (task == 'show_metrics_results_already_computed'): # Show already computed results averaging over replicates
    classTask = iResultsAlreadyComputed(config)

#'''
for method in config["method"]['grid_search']:
    os.system("rm -rf " + root + '/data/Algo/' + 'suffixes_for_last_run_' + method + '.txt')

# Launch task
classTask.runRayTune(config,root,task)
#'''

from csv import reader as reader_csv
import numpy as np
import matplotlib.pyplot as plt

for ROI in ['hot','cold']:

    suffixes_legend = []

    if classTask.debug:
        method_list = [config["method"]]
    else:
        method_list = config["method"]['grid_search']
    for method in method_list:
        print("method",method)
        suffixes = []

        PSNR_recon = []
        PSNR_norm_recon = []
        MSE_recon = []
        MA_cold_recon = []
        CRC_hot_recon = []
        CRC_bkg_recon = []
        IR_bkg_recon = []

        if ROI == 'hot':
            metrics = CRC_hot_recon
        else:
            metrics = MA_cold_recon 


        with open(root + '/data/Algo' + '/suffixes_for_last_run_' + method + '.txt') as f:
            suffixes.append(f.readlines())

        print("suffixes = ", suffixes)
        # Load metrics from last runs to merge them in one figure

        for suffix in suffixes[0]:
            metrics_file = root + '/data/Algo' + '/metrics/' + method + '/' + suffix.rstrip("\n") + '/' + 'metrics.csv'
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

                '''
                print(PSNR_recon)
                print(PSNR_norm_recon)
                print(MSE_recon)
                print(MA_cold_recon)
                print(CRC_hot_recon)
                print(CRC_bkg_recon)
                print(IR_bkg_recon)
                '''

        plt.figure()
        for run_id in range(len(PSNR_recon)):
            plt.plot(IR_bkg_recon[run_id],metrics[run_id],'-o')

        plt.xlabel('IR')
        plt.ylabel('CRC')

        for i in range(len(suffixes[0])):
            
            l = suffixes[0][i].replace('=','_')
            l = l.replace('\n','_')
            l = l.split('_')
            legend = ''
            for p in range(len(l)):
                if l[p] == "AML":
                    legend += "A : " + l[p+1] + ' / ' 
                if l[p] == "NNEPP":
                    legend += "NNEPPS : " + l[p+1]
            suffixes_legend.append(legend)
    plt.legend(suffixes_legend)
    # Saving this figure locally
    plt.savefig(root + '/data/Algo/' + 'debug/'*classTask.debug + 'metrics/' + 'CRC in ' + ROI + ' region vs IR in background' + '.png')
    from textwrap import wrap
    wrapped_title = "\n".join(wrap(suffix, 50))
    plt.title(wrapped_title,fontsize=12)