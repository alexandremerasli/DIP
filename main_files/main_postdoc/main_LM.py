#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: alexandre (2021-2022)
"""

## Python libraries
# Useful
from ray import tune
import importlib
import sys
import os
from os.path import join
sys.path.append((os.getcwd()))  # Add the workspace directory to the Python path

def uncompatible_parameters(config):
    method = config["method"]["grid_search"][0]
    if (("DNA" in method or "DIPRecon" in method) and config["rho"]["grid_search"][0] == 0 and task == "castor_reco"):
        raise ValueError("DNA must be launched with rho > 0")
    elif ((method != "DIPRecon" and method != "DNA") and task == "post_reco"):
        raise ValueError("Only DIPRecon or DNA can be run in post reconstruction mode, not CASToR reconstruction algorithms. Please comment this line.")
    elif ((method == "DIPRecon" or method == "DNA") and config["rho"]["grid_search"][0] == 0 and task != "post_reco"):
        raise ValueError("Please set rho > 0 for DNA or DIPRecon reconstruction (or set task to post reconstruction).")
    elif (config["DIP_early_stopping_when"]["grid_search"][0] != "never" and (config["windowSize"]["grid_search"][0] >= config["sub_iter_DIP"]["grid_search"][0] and config["EMV_or_WMV"]["grid_search"][0] == "WMV")):
        raise ValueError("Please set window size less than number of DIP iterations for Window Moving Variance.")
    elif ((config["sub_iter_DIP_init"]["grid_search"][0] <= config["patienceNumber"]["grid_search"][0]) or (config["sub_iter_DIP"]["grid_search"][0] <= config["patienceNumber"]["grid_search"][0] and (config["DIP_early_stopping_when"]["grid_search"][0] == "init" or config["DIP_early_stopping_when"]["grid_search"][0] == "all")) or (config["sub_iter_DIP"]["grid_search"][0] <= config["patienceNumber"]["grid_search"][0] and config["DIP_early_stopping_when"]["grid_search"][0] == "all")):
        raise ValueError("Please set patienceNumber higher than sub_iter_DIP")
    elif (config["DIP_it_if_no_ES_found"]["grid_search"][0] > config["sub_iter_DIP_init"]["grid_search"][0]):
        raise ValueError("Please set DIP_it_if_no_ES_found higher than sub_iter_DIP_init")

def class_for_task(config,task):
    if (task == 'full_reco_with_network'): # Run DIPRecon or DNA
        from iADMM_DIP import iADMM_DIP
        classTask = iADMM_DIP(config)
        # raise ValueError("needs hyperparameters_config")
    elif (task == 'castor_reco'): # Run CASToR reconstruction with given optimizer
        from iCastorAlgo import iCastorAlgo
        classTask = iCastorAlgo(config)
    elif (task == 'post_reco'): # Run network denoising after a given reconstructed image im_corrupt
        from iPostReconstruction import iPostReconstruction
        classTask = iPostReconstruction(config)
    elif (task == 'show_results'): # Show already computed results over iterations
        from iResults import iResults
        classTask = iResults(config)
    elif (task == 'show_results_post_reco'): # Show already computed results over iterations of post reconstruction mode
        from iResults import iResults
        classTask = iResults(config)
    elif (task == 'show_metrics_ADMMReg'): # Show ADMMReg FOMs over iterations
        config["task"] = "show_results_post_reco"
        from iMeritsADMMReg import iMeritsADMMReg
        classTask = iMeritsADMMReg(config)
    elif (task == 'show_metrics_DNA'): # Show DNA or DIPRecon FOMs over iterations
        from iMeritsDIP_ADMM import iMeritsDIP_ADMM
        classTask = iMeritsDIP_ADMM(config)
    # elif (task == 'show_metrics_results_already_computed'): # Show already computed results averaging over replicates
    #     from iResultsAlreadyComputed import iResultsAlreadyComputed
    #     classTask = iResultsAlreadyComputed(config)
    elif ('compare_2_methods' in task): # Show already computed results averaging over replicates
        from iResultsADMMReg_VS_APPGML import iResultsADMMReg_VS_APPGML
        classTask = iResultsADMMReg_VS_APPGML(config)

    return classTask

def choose_task(config):
    # Task to run reconstruction according to the method
    method = config["method"]["grid_search"][0]
    if (method == "DIPRecon" or method == "DNA"):
        task = 'full_reco_with_network'

    elif ('ADMMReg' in method or method == 'MLEM' or method == 'OPTITR' or method == 'OSEM' or method == 'BSREM' or method == 'AML' or method == 'APPGML'):
        task = 'castor_reco'

    # Override task here if needed
    # task = 'full_reco_with_network' # Run DIPRecon or DNA
    # task = 'castor_reco' # Run CASToR reconstruction with given optimizer
    # task = 'post_reco' # Run network denoising after a given reconstructed image im_corrupt
    # task = 'show_results_post_reco'
    # task = 'show_results'
    # task = 'show_metrics_results_already_computed'
    # task = 'show_metrics_ADMMReg'
    # task = 'show_metrics_DNA'
    # task = 'compare_2_methods'
    # task = 'compare_2_methods_post_reco'

    return task

nb_computation = 1
config_files = ["LM_OSEM"]
config_files = ["my_LM_DIPRecon"]
config_files = ["test_debug"]

i=-1
num_meth=0
for lib_string in config_files:
    i+=1
    if (i>=nb_computation): # Restart count for new setting
        i=-1
        num_meth +=1
    # try:
    if (True):
        # subfolder_config = "" # PhD settings
        subfolder_config = "LM" # List-Mode (LM) settings
        sys.path.append(join('all_config',subfolder_config))  # Add the parent directory of config files to the Python path
        lib = importlib.import_module(lib_string)
        config = lib.config_func_MIC()
        config["image"] = tune.grid_search(['image40_1_114'])    
        config["image"] = tune.grid_search(['imageUHR_IEC'])
        config["image"] = tune.grid_search(['imageUHR_IEC4_8'])
        config["image"] = tune.grid_search(['image40_1'])
        config["replicates"] = tune.grid_search(list(range(1,1+1)))
        config["max_iter"] = tune.grid_search([3])
        config["ray"] = False

        root = os.getcwd()
        subroot = "/data/Algo/"
        
        # Write random seed in a file to get it in network architectures
        os.system("rm -rf " + os.getcwd() +"/seed.txt")
        file_seed = open(os.getcwd() + "/seed.txt","w+")
        file_seed.write(str(config["random_seed"]["grid_search"][0]))
        file_seed.close()

        for method in config["method"]['grid_search']:
            config_tmp = dict(config)
            config_tmp["method"] = tune.grid_search([method]) # Put only 1 method to remove useless hyperparameters from settings_config and hyperparameters_config

            # Choose task to do (move this after raytune !!!)
            task = choose_task(config)

            # Local files to import, AFTER CONFIG TO SET RANDOM SEED OR NOT
            classTask = class_for_task(config,task)

            # Incompatible parameters (should be written in vGeneral I think)
            uncompatible_parameters(config)

            # Launch task
            classTask.runRayTune(config_tmp,root,task)