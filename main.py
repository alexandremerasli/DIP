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
    "method" : tune.grid_search(['APGMAP']), # Reconstruction algorithm (nested, Gong, or algorithms from CASToR (MLEM, BSREM, AML, etc.))
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
    "rho" : tune.grid_search([0.01,0.02,0.03,0.04,0.05]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (nested and Gong)
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
    "post_smoothing" : tune.grid_search([5]), # Post smoothing by CASToR after reconstruction
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

from csv import reader as reader_csv
import numpy as np
import matplotlib.pyplot as plt

for ROI in ['hot','cold']:
    plt.figure()

    suffixes_legend = []
    replicates_legend = []

    if classTask.debug:
        method_list = [config["method"]]
    else:
        method_list = config["method"]['grid_search']
    for method in method_list:
        suffixes = []
        replicates = []

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

        # Load metrics from last runs to merge them in one figure
        for idx in range(len(suffixes[0])):
            suffix = suffixes[0][idx]
            replicate = replicates[0][idx].rstrip()
            metrics_file = root + '/data/Algo' + '/metrics/' + config["image"]['grid_search'][0] + '/' + str(replicate) + '/' + method + '/' + suffix.rstrip("\n") + '/' + 'metrics.csv'
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

                    '''
                    print("metriiiiiiiiiiiiiiiics")
                    print(PSNR_recon)
                    print(PSNR_norm_recon)
                    print(MSE_recon)
                    print(SSIM_recon)
                    print(MA_cold_recon)
                    print(AR_hot_recon)
                    print(AR_bkg_recon)
                    print(IR_bkg_recon)
                    '''
            except:
                print("No such file : " + metrics_file)

        if ROI == 'hot':
            #metrics = [abs(hot) for hot in AR_hot_recon] # Take absolute value of AR hot for tradeoff curves
            metrics = AR_hot_recon # Take absolute value of AR hot for tradeoff curves
        else:
            #metrics = [abs(cold) for cold in MA_cold_recon] # Take absolute value of MA cold for tradeoff curves
            metrics = MA_cold_recon # Take absolute value of MA cold for tradeoff curves

        if (method == "nested" or method == "Gong"):
            for case in range(np.array(IR_bkg_recon).shape[0]):
                if (method == "Gong"):
                    IR_final.append(np.array(IR_bkg_recon)[case,:-1])
                    metrics_final.append(np.array(metrics)[case,:-1])
                if (method == "nested"):
                    IR_final.append(np.array(IR_bkg_recon)[case,:config["max_iter"]['grid_search'][0]])
                    metrics_final.append(np.array(metrics)[case,:config["max_iter"]['grid_search'][0]])
        elif (method == "BSREM" or method == "MLEM" or method == "ADMMLim" or method == "AML" or method == "APGMAP"):

            #IR_final.append(np.array([IR_bkg_recon[elem][-1] for elem in range(len(IR_bkg_recon))]))
            #metrics_final.append(np.array([metrics[elem][-1] for elem in range(len(metrics))]))
            IR_final.append(np.array([IR_bkg_recon[elem][:] for elem in range(len(IR_bkg_recon))]))
            metrics_final.append(np.array([metrics[elem][:] for elem in range(len(metrics))]))



            
        fig1, ax1 = plt.subplots()
        
        ax1.set_xlabel('Image Roughness in the background (%)', fontsize = 18)
        ax1.set_ylabel('Absolute bias (AU)', fontsize = 18)

        IR_final = IR_final[0]
        metrics_final = metrics_final[0]

        for case in range(len(IR_final)):
            idx_sort = np.argsort(IR_final[case])
            print(metrics_final[0].shape)
            print(metrics_final[case][-1])
            print(metrics_final[case][:])
            print(idx_sort)
            if method == 'ADMMLim' or method == 'APGMAP':
                p = ax1.plot(100*IR_final[case][idx_sort],metrics_final[case][idx_sort],'-o') # IR in %
            elif (method == "nested" or method == "Gong"):
                ax1.plot(100*IR_final[case][0],metrics_final[case][0],'o', mfc='none',color='black',label='_nolegend_') # IR in %
            else:
                ax1.plot(100*IR_final[case][idx_sort],metrics_final[case][idx_sort],'-o') # IR in %

        '''
        if (method == "nested" or method == "Gong"):
            for i in range(len(suffixes[0])):
                l = suffixes[0][i].replace('=','_')
                l = l.replace('\n','_')
                l = l.split('_')
                legend = method + ' - '
                for p in range(len(l)):
                    if l[p] == "skip":
                        legend += "skip : " + l[p+2] + ' / ' 
                    if l[p] == "input":
                        legend += "input : " + l[p+1]
                suffixes_legend.append(legend)

        else:
        '''
        if (method == 'Gong'):
            legend_method = 'DIPRecon'
        elif (method == 'nested'):
            legend_method = 'nested ADMM'
        elif (method == 'MLEM'):
            legend_method = 'MLEM + filter'
        else:
            legend_method = method
        suffixes_legend.append(legend_method)
        ax1.legend(suffixes_legend)












        if ROI == 'hot':
            metrics = AR_hot_recon
        else:
            metrics = MA_cold_recon # Do not take absolute value of MA cold for bias for different replicates

        IR_final = []
        metrics_final = []

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

        fig2, ax2 = plt.subplots()
        
        ax2.set_xlabel('Iterations', fontsize = 18)
        ax2.set_ylabel('Bias (AU)', fontsize = 18)

        if (method != "nested" and method != "Gong"):
            metrics_final = metrics_final[0]


        len_mini_list = []
        for replicate in range(len(metrics_final)):
            len_mini_list.append(len(metrics_final[replicate]))

        len_mini = int(min(len_mini_list))

        avg = np.zeros((len_mini,),dtype=np.float64)
        for replicate in range(len(metrics_final)):
            #ax2.plot(np.arange(0,len(metrics_final[replicate])),metrics_final[replicate])
            ax2.plot(np.arange(0,len(metrics_final[replicate])),metrics_final[replicate],label='_nolegend_')
            if (method == "nested" or method == "Gong"):
                ax2.plot(np.arange(0,len(metrics_final[replicate])),metrics_final[replicate], mfc='none',color='black',label='_nolegend_')
            
            avg += np.array(metrics_final[replicate][:len_mini]) / len(metrics_final)

            #replicates_legend.append("replicate_" + str(replicate+1))
 
        ax2.plot(np.arange(0,len(avg)),avg,color='black')
        print("avg",avg)
        replicates_legend.append("average over replicates")
        
        ax2.legend(replicates_legend)
        ax2.set_title(method + " reconstruction for " + str(len(metrics_final)) + " replicates")
 







        """
            
        fig3, ax3 = plt.subplots()
        
        ax3.set_xlabel('Image Roughness in the background (%)', fontsize = 18)
        ax3.set_ylabel('Absolute bias (AU)', fontsize = 18)

        #IR_final = IR_final[0]
        #metrics_final = metrics_final[0]

        for case in range(len(IR_final)):
            idx_sort = np.argsort(IR_final[case])
            print(metrics_final[0].shape)
            print(metrics_final[case][-1])
            print(metrics_final[case][:])
            print(idx_sort)
            if method == 'ADMMLim' or method == 'APGMAP':
                p = ax3.plot(100*IR_final[case][idx_sort],avg[idx_sort],'-o') # IR in %
            elif (method == "nested" or method == "Gong"):
                ax3.plot(100*IR_final[case][0],avg[0],'o', mfc='none',color='black',label='_nolegend_') # IR in %
            else:
                ax3.plot(100*IR_final[case][idx_sort],avg[idx_sort],'-o') # IR in %

        '''
        if (method == "nested" or method == "Gong"):
            for i in range(len(suffixes[0])):
                l = suffixes[0][i].replace('=','_')
                l = l.replace('\n','_')
                l = l.split('_')
                legend = method + ' - '
                for p in range(len(l)):
                    if l[p] == "skip":
                        legend += "skip : " + l[p+2] + ' / ' 
                    if l[p] == "input":
                        legend += "input : " + l[p+1]
                suffixes_legend.append(legend)

        else:
        '''
        if (method == 'Gong'):
            legend_method = 'DIPRecon'
        elif (method == 'nested'):
            legend_method = 'nested ADMM'
        elif (method == 'MLEM'):
            legend_method = 'MLEM + filter'
        else:
            legend_method = method
        suffixes_legend.append(legend_method)
        ax3.legend(suffixes_legend)

        """






    # Saving this figure locally
    if ROI == 'hot':
        fig1.savefig(root + '/data/Algo/' + 'debug/'*classTask.debug + 'metrics/' + method + " rho = " + str(config_tmp["rho"]['grid_search'][0]) + 'AR in ' + ROI + ' region vs IR in background' + '.png')
    elif ROI == 'cold':
        fig1.savefig(root + '/data/Algo/' + 'debug/'*classTask.debug + 'metrics/' + method + " rho = " + str(config_tmp["rho"]['grid_search'][0]) + 'MA in ' + ROI + ' region vs IR in background' + '.png')
    from textwrap import wrap
    wrapped_title = "\n".join(wrap(suffix, 50))
    #ax3.set_title(wrapped_title,fontsize=12)

    # Saving this figure locally
    if ROI == 'hot':
        fig2.savefig(root + '/data/Algo/' + 'debug/'*classTask.debug + 'metrics/' + method + " rho = " + str(config_tmp["rho"]['grid_search'][0]) + ' : AR in ' + ROI + ' region for ' + str(len(metrics_final)) + ' replicates' + '.png')
    elif ROI == 'cold':
        fig2.savefig(root + '/data/Algo/' + 'debug/'*classTask.debug + 'metrics/' + method + " rho = " + str(config_tmp["rho"]['grid_search'][0]) + ' : MA in ' + ROI + ' region for ' + str(len(metrics_final)) + ' replicates' + '.png')
    from textwrap import wrap
    wrapped_title = "\n".join(wrap(suffix, 50))
    ax2.set_title(wrapped_title,fontsize=12)

    """
    # Saving this figure locally
    if ROI == 'hot':
        fig3.savefig(root + '/data/Algo/' + 'debug/'*classTask.debug + 'metrics/' + method + " rho = " + str(config_tmp["rho"]['grid_search'][0]) + 'AR in ' + ROI + ' region vs IR in background' + '.png')
    elif ROI == 'cold':
        fig3.savefig(root + '/data/Algo/' + 'debug/'*classTask.debug + 'metrics/' + method + " rho = " + str(config_tmp["rho"]['grid_search'][0]) + 'MA in ' + ROI + ' region vs IR in background' + '.png')
    from textwrap import wrap
    wrapped_title = "\n".join(wrap(suffix, 50))
    ax3.set_title(wrapped_title,fontsize=12)
    """
#sys.stdout.close()
#sys.stdout=stdoutOrigin