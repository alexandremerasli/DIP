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
    "image" : tune.grid_search(['image0']), # Image from database
    "random_seed" : tune.grid_search([True]), # If True, random seed is used for reproducibility (must be set to False to vary weights initialization)
    "method" : tune.grid_search(['BSREM']), # Reconstruction algorithm (nested, Gong, or algorithms from CASToR (MLEM, BSREM, AML, etc.))
    "processing_unit" : tune.grid_search(['CPU']), # CPU or GPU
    "nb_threads" : tune.grid_search([1]), # Number of desired threads. 0 means all the available threads
    "FLTNB" : tune.grid_search(['double']), # FLTNB precision must be set as in CASToR (double necessary for ADMMLim and nested)
    "debug" : False, # Debug mode = run without raytune and with one iteration
    "ray" : True, # Ray mode = run with raytune if True, to run several settings in parallel
    "tensorboard" : True, # Tensorboard mode = show results in tensorboard
    "all_images_DIP" : tune.grid_search(['True']), # Option to store only 10 images like in tensorboard (quicker, for visualization, set it to "True" by default). Can be set to "True", "False", "Last" (store only last image)
    "experiment" : tune.grid_search([24]),
    "image_init_path_without_extension" : tune.grid_search(['1_im_value_cropped']), # Initial image of the reconstruction algorithm (taken from data/algo/Data/initialization)
    #"f_init" : tune.grid_search(['1_im_value_cropped']),
    "replicates" : tune.grid_search(list(range(1,100+1))), # List of desired replicates. list(range(1,n+1)) means n replicates
    "replicates" : tune.grid_search(list(range(1,2+1))), # List of desired replicates. list(range(1,n+1)) means n replicates
    "average_replicates" : tune.grid_search([False]), # List of desired replicates. list(range(1,n+1)) means n replicates
    "castor_foms" : tune.grid_search([True]), # Set to True to compute CASToR Figure Of Merits (likelihood, residuals for ADMMLim)
}
# Configuration dictionnary for previous hyperparameters, but fixed to simplify
fixed_config = {
    "max_iter" : tune.grid_search([10]), # Number of global iterations for usual optimizers (MLEM, BSREM, AML etc.) and for nested and Gong
    "nb_subsets" : tune.grid_search([28]), # Number of subsets in chosen reconstruction algorithm (automatically set to 1 for ADMMLim)
    "finetuning" : tune.grid_search(['False']),
    "penalty" : tune.grid_search(['MRF']), # Penalty used in CASToR for PLL algorithms
    "unnested_1st_global_iter" : tune.grid_search([True]), # If True, unnested are computed after 1st global iteration (because rho is set to 0). If False, needs to set f_init to initialize the network, as in Gong paper, and rho is not changed.
    "sub_iter_DIP_initial" : tune.grid_search([10]), # Number of epochs in first global iteration (pre iteraiton) in network optimization (only for Gong for now)
    "nb_inner_iteration" : tune.grid_search([1]), # Number of inner iterations in ADMMLim (if mlem_sequence is False) or in OPTITR (for Gong). CASToR output is doubled because of 2 inner iterations for 1 inner iteration
    "xi" : tune.grid_search([1]), # Factor to balance primal and dual residual convergence speed in adaptive tau computation in ADMMLim
    "net" : tune.grid_search(['DIP']), # Network to use (DIP,DD,DD_AE,DIP_VAE)
    "windowSize" : tune.grid_search([100]), # Network to use (DIP,DD,DD_AE,DIP_VAE)
    "patienceNumber" : tune.grid_search([500]), # Network to use (DIP,DD,DD_AE,DIP_VAE)
}
# Configuration dictionnary for hyperparameters to tune
hyperparameters_config = {
    "rho" : tune.grid_search([0,3,3e-1,3e-2,3e-3,3e-4,3e-5,3e-6,3e-7]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (nested and Gong)
    "rho" : tune.grid_search([3e-3,3e-4,3e-5]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (nested and Gong)
    "rho" : tune.grid_search([0.03]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (nested and Gong)
    #"rho" : tune.grid_search([0.05]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (nested and Gong)
    #"rho" : tune.grid_search([0]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (nested and Gong)
    ## network hyperparameters
    "lr" : tune.grid_search([0.001,0.005,0.01,0.05,0.1]), # Learning rate in network optimization
    "lr" : tune.grid_search([0.005]), # Learning rate in network optimization
    "sub_iter_DIP" : tune.grid_search([1000]), # Number of epochs in network optimization
    "opti_DIP" : tune.grid_search(['Adam']), # Optimization algorithm in neural network training (Adam, LBFGS)
    "skip_connections" : tune.grid_search([0]), # Number of skip connections in DIP architecture (0, 1, 2, 3)
    #"skip_connections" : tune.grid_search([0,1,2,3]), # Number of skip connections in DIP architecture (0, 1, 2, 3)
    "scaling" : tune.grid_search(['standardization']), # Pre processing of neural network input (nothing, uniform, normalization, standardization)
    "input" : tune.grid_search(['CT']), # Neural network input (random or CT)
    #"input" : tune.grid_search(['CT','random']), # Neural network input (random or CT)
    "d_DD" : tune.grid_search([4]), # d for Deep Decoder, number of upsampling layers. Not above 4, otherwise 112 is too little as output size / not above 6, otherwise 128 is too little as output size
    "k_DD" : tune.grid_search([32]), # k for Deep Decoder
    ## ADMMLim - OPTITR hyperparameters
    "nb_outer_iteration": tune.grid_search([10000]), # Number outer iterations in ADMMLim
    "alpha" : tune.grid_search([1]), # alpha (penalty parameter) in ADMMLim
    "adaptive_parameters" : tune.grid_search(["tau"]), # which parameters are adaptive ? Must be set to nothing, alpha, or tau (which means alpha and tau)
    "mu_adaptive" : tune.grid_search([2]), # Factor to balance primal and dual residual in adaptive alpha computation in ADMMLim
    "tau" : tune.grid_search([100]), # Factor to multiply alpha in adaptive alpha computation in ADMMLim
    ## hyperparameters from CASToR algorithms 
    # Optimization transfer (OPTITR) hyperparameters
    "mlem_sequence" : tune.grid_search([False]), # Given sequence (with decreasing number of subsets) to quickly converge. True or False
    # AML/APGMAP hyperparameters
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
from iResultsAlreadyComputed import iResultsAlreadyComputed

for method in config["method"]['grid_search']:

    '''
    # Gong reconstruction
    if (config["method"]["grid_search"][0] == 'Gong' and len(config["method"]["grid_search"]) == 1):
        print("configuration fiiiiiiiiiiiiiiiiiiile")
        #config = np.load(root + 'config_Gong.npy',allow_pickle='TRUE').item()
        from Gong_configuration import config_func, config_func_MIC
        #config = config_func()
        config = config_func_MIC()

    # nested reconstruction
    if (config["method"]["grid_search"][0] == 'nested' and len(config["method"]["grid_search"]) == 1):
        print("configuration fiiiiiiiiiiiiiiiiiiile")
        from nested_configuration import config_func, config_func_MIC
        #config = config_func()
        config = config_func_MIC()

    # MLEM reconstruction
    if (config["method"]["grid_search"][0] == 'MLEM' and len(config["method"]["grid_search"]) == 1):
        print("configuration fiiiiiiiiiiiiiiiiiiile")
        from MLEM_configuration import config_func_MIC
        #config = config_func()
        config = config_func_MIC()

    # BSREM reconstruction
    if (config["method"]["grid_search"][0] == 'BSREM' and len(config["method"]["grid_search"]) == 1):
        print("configuration fiiiiiiiiiiiiiiiiiiile")
        from BSREM_configuration import config_func_MIC
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

    elif ('ADMMLim' in config["method"]["grid_search"][0] or config["method"]["grid_search"][0] == 'MLEM' or config["method"]["grid_search"][0] == 'BSREM' or config["method"]["grid_search"][0] == 'AML' or config["method"]["grid_search"][0] == 'APGMAP'):
        task = 'castor_reco'

    #task = 'full_reco_with_network' # Run Gong or nested ADMM
    #task = 'castor_reco' # Run CASToR reconstruction with given optimizer
    #task = 'post_reco' # Run network denoising after a given reconstructed image im_corrupt
    #task = 'show_results_post_reco'
    #task = 'show_results'
    #task = 'show_metrics_ADMMLim'
    task = 'show_metrics_results_already_computed'

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
    elif (task == 'show_metrics_results_already_computed'): # Show already computed results averaging over replicates
        classTask = iResultsAlreadyComputed(config)

    # Incompatible parameters (should be written in vGeneral I think)
    if (config["method"]["grid_search"][0] == 'nested' and config["rho"]["grid_search"][0] == 0 and task == "castor_reco"):
        raise ValueError("nested must be launched with rho > 0")
    elif (config["method"]["grid_search"][0] == 'Gong' and config["max_iter"]["grid_search"][0]  == 1):
        raise ValueError("Gong must be run with at least 2 global iterations to compute metrics")
    elif ((config["method"]["grid_search"][0] != 'Gong' and config["method"]["grid_search"][0] != 'nested') and task == "post_reco"):
        raise ValueError("Only Gong or nested can be run in post reconstruction mode, not CASToR reconstruction algorithms. Please comment this line.")
    elif ((config["method"]["grid_search"][0] == 'Gong' or config["method"]["grid_search"][0] == 'nested') and config["all_images_DIP"]["grid_search"][0] != "True"):
        raise ValueError("Please set all_images_DIP to True to save all images for nested or Gong reconstruction.")
    elif ((config["method"]["grid_search"][0] == 'Gong' or config["method"]["grid_search"][0] == 'nested') and config["rho"]["grid_search"][0] == 0 and task != "post_reco"):
        raise ValueError("Please set rho > 0 for nested or Gong reconstruction.")
    elif (config["windowSize"]["grid_search"][0] >= config["sub_iter_DIP"]["grid_search"][0]):
        raise ValueError("Please set window size less than number of DIP iterations for Window Moving Variance.")
    elif (config["debug"] and config["ray"]):
        raise ValueError("Debug mode must is used without ray")


    #'''
    os.system("rm -rf " + root + '/data/Algo/' + 'suffixes_for_last_run_' + method + '.txt')
    os.system("rm -rf " + root + '/data/Algo/' + 'replicates_for_last_run_' + method + '.txt')

    # Launch task
    classTask.runRayTune(config_tmp,root,task)
    #'''



def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

from csv import reader as reader_csv
import numpy as np
import matplotlib.pyplot as plt
import re

for ROI in ['hot','cold']:
    plt.figure()

    suffixes_legend = []
    replicates_legend = []

    if classTask.debug:
        method_list = [config["method"]]
    else:
        method_list = config["method"]['grid_search']
    for method in method_list: # Loop over methods
        suffixes = []
        replicates = []
        rho_list = []
        replicates_list = []

        PSNR_recon = []
        PSNR_norm_recon = []
        MSE_recon = []
        SSIM_recon = []
        MA_cold_recon = []
        AR_hot_recon = []
        AR_bkg_recon = []
        IR_bkg_recon = []

        IR_final = []
        metrics_final = []
        
        with open(root + '/data/Algo' + '/suffixes_for_last_run_' + method + '.txt') as f:
            suffixes.append(f.readlines())
        with open(root + '/data/Algo' + '/replicates_for_last_run_' + method + '.txt') as f:
            replicates.append(f.readlines())

        print(root + '/data/Algo' + '/replicates_for_last_run_' + method + '.txt')

        # Sort replicates from file
        replicate_idx = [replicates[0][idx].rstrip() for idx in range(len(replicates[0]))]
        idx_replicates_sort = np.argsort(replicate_idx)
        # Sort suffixes from file by rho values 
        sorted_suffixes = list(suffixes[0])
        sorted_suffixes.sort(key=natural_keys)

        # Load metrics from last runs to merge them in one figure
        for idx in idx_replicates_sort: # Loop over rhos and replicates, for each sorted rho, take sorted replicate
            suffix = sorted_suffixes[idx].rstrip("\n")
            replicate = replicates[0][idx].rstrip()
            metrics_file = root + '/data/Algo' + '/metrics/' + config["image"]['grid_search'][0] + '/' + str(replicate) + '/' + method + '/' + suffix + '/' + 'metrics.csv'
            try:
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
                    rows_csv[7] = [float(i) for i in rows_csv[7]]

                    PSNR_recon.append(np.array(rows_csv[0]))
                    PSNR_norm_recon.append(np.array(rows_csv[1]))
                    MSE_recon.append(np.array(rows_csv[2]))
                    SSIM_recon.append(np.array(rows_csv[3]))
                    MA_cold_recon.append(np.array(rows_csv[4]))
                    AR_hot_recon.append(np.array(rows_csv[5]))
                    AR_bkg_recon.append(np.array(rows_csv[6]))
                    IR_bkg_recon.append(np.array(rows_csv[7]))

            except:
                print("No such file : " + metrics_file)

        # Select metrics to plot according to ROI
        if ROI == 'hot':
            #metrics = [abs(hot) for hot in AR_hot_recon] # Take absolute value of AR hot for tradeoff curves
            metrics = AR_hot_recon # Take absolute value of AR hot for tradeoff curves
        else:
            #metrics = [abs(cold) for cold in MA_cold_recon] # Take absolute value of MA cold for tradeoff curves
            metrics = MA_cold_recon # Take absolute value of MA cold for tradeoff curves

        # Keep useful information to plot from metrics
        if (method == "nested" or method == "Gong"):
            for case in range(np.array(IR_bkg_recon).shape[0]):
                if (method == "Gong"):
                    IR_final.append(np.array(IR_bkg_recon)[case,:-1])
                    metrics_final.append(np.array(metrics)[case,:-1])
                if (method == "nested"):
                    IR_final.append(np.array(IR_bkg_recon)[case,:config["max_iter"]['grid_search'][0]])
                    metrics_final.append(np.array(metrics)[case,:config["max_iter"]['grid_search'][0]])
        elif (method == "BSREM" or method == "MLEM" or method == "ADMMLim" or method == "AML" or method == "APGMAP"):
            IR_final.append(IR_bkg_recon)
            metrics_final.append(metrics)
        
        IR_final = IR_final[0]
        metrics_final = metrics_final[0]

        # Retrieve number of rhos and replicates
        nb_rho = len(config["rho"]["grid_search"])
        nb_replicates = int(len(metrics_final) / nb_rho)

        # Compute number of displayable iterations for each rho
        len_mini_list = np.zeros((nb_rho,nb_replicates),dtype=int)
        len_mini = np.zeros((nb_rho),dtype=int)
        for rho_idx in range(nb_rho):
            for replicate in range(nb_replicates):
                len_mini_list[rho_idx][replicate] = len(metrics_final[replicate + nb_replicates*rho_idx])
            len_mini[rho_idx] = int(np.min(len_mini_list[rho_idx]))

        # Plot 2 figures for each ROI : tradeoff curve (metric VS IR) and bias with iterations
        for fig_nb in range(2):
            fig, ax = plt.subplots()
            for rho_idx in range(nb_rho):
                avg = np.zeros((len_mini[rho_idx],),dtype=np.float64)
                std = np.zeros((len_mini[rho_idx],),dtype=np.float64)
                for replicate in range(nb_replicates):
                    case = replicate + nb_replicates*rho_idx
                    # Plot tradeoff curves
                    if (fig_nb == 0):
                        idx_sort = np.argsort(IR_final[case])
                        if (method == "nested" or method == "Gong"):
                            ax.plot(100*IR_final[case][0],metrics_final[case][0],'o', mfc='none',color='black',label='_nolegend_') # IR in %
                        else:
                            ax.plot(100*IR_final[case][idx_sort],metrics_final[case][idx_sort],'-o') # IR in %
                        ax.set_xlabel('Image Roughness in the background (%)', fontsize = 18)
                        ax.set_ylabel('Absolute bias (AU)', fontsize = 18)

                    if (fig_nb == 1):
                        ax.plot(np.arange(0,len(metrics_final[case])),metrics_final[case],label='_nolegend_') # Plot bias curves with iterations for each replicate
                        # Compute average of bias curves with iterations
                        avg += np.array(metrics_final[case][:len_mini[rho_idx]]) / nb_replicates

                if (fig_nb == 1):
                    # Compute std bias curves with iterations
                    for replicate in range(nb_replicates):
                        std += np.sqrt((np.array(metrics_final[case][:len_mini[rho_idx]]) - avg)**2 / nb_replicates)

                    # Plot average and std of bias curves with iterations
                    ax.plot(np.arange(0,len(avg)),avg,color='black')
                    ax.fill_between(np.arange(0,len(avg)), avg - std, avg + std, alpha = 0.4)
                    replicates_legend.append("average over replicates")
                    ax.legend(replicates_legend)
                    ax.set_xlabel('Iterations', fontsize = 18)
                    ax.set_ylabel('Bias (AU)', fontsize = 18)
                    ax.set_title(method + " reconstruction for " + str(nb_replicates) + " replicates")
    
            # Saving figures locally in png
            if (fig_nb == 0):
                if ROI == 'hot':
                    title = method + " rho = " + str(config_tmp["rho"]['grid_search'][0]) + 'AR in ' + ROI + ' region vs IR in background' + '.png'    
                elif ROI == 'cold':
                    title = method + " rho = " + str(config_tmp["rho"]['grid_search'][0]) + 'AR in ' + ROI + ' region vs IR in background' + '.png'    
            elif (fig_nb == 1):
                if ROI == 'hot':
                    title = method + " rho = " + str(config_tmp["rho"]['grid_search'][0]) + 'AR in ' + ROI + ' region for ' + str(nb_replicates) + ' replicates' + '.png'
                elif ROI == 'cold':
                    title = method + " rho = " + str(config_tmp["rho"]['grid_search'][0]) + 'AR in ' + ROI + ' region for ' + str(nb_replicates) + ' replicates' + '.png'
            fig.savefig(root + '/data/Algo/' + 'debug/'*classTask.debug + 'metrics/' + title)
            from textwrap import wrap
            wrapped_title = "\n".join(wrap(suffix, 50))
            #ax.set_title(wrapped_title,fontsize=12)

#sys.stdout.close()
#sys.stdout=stdoutOrigin