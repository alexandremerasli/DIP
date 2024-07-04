#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: alexandre (2021-2022)
"""

## Python libraries
# Useful
import os
from ray import tune
import importlib

def uncompatible_parameters(config):
    method = config["method"]["grid_search"][0]
    if (method == "DNA" and config["rho"]["grid_search"][0] == 0 and task == "castor_reco"):
        raise ValueError("DNA must be launched with rho > 0")
    elif ((method != "DIPRecon" and method != "DNA") and task == "post_reco"):
        raise ValueError("Only DIPRecon or DNA can be run in post reconstruction mode, not CASToR reconstruction algorithms. Please comment this line.")
    elif ((method == "DIPRecon" or method == "DNA") and config["all_images_DIP"]["grid_search"][0] != "True" and config["DIP_early_stopping"]["grid_search"][0] == "True"):
        raise ValueError("Please set all_images_DIP to True to save all images for DNA or DIPRecon reconstruction if using moving variance algorithms")
    elif ((method == "DIPRecon" or method == "DNA") and config["rho"]["grid_search"][0] == 0 and task != "post_reco"):
        raise ValueError("Please set rho > 0 for DNA or DIPRecon reconstruction (or set task to post reconstruction).")
    elif (config["windowSize"]["grid_search"][0] >= config["sub_iter_DIP"]["grid_search"][0] and config["EMV_or_WMV"]["grid_search"][0] == "WMV"):
        raise ValueError("Please set window size less than number of DIP iterations for Window Moving Variance.")
    elif (config["debug"] and config["ray"]):
        raise ValueError("Debug mode must is used without ray")
    elif (task == "post_reco" and config["DIP_early_stopping"]["grid_search"][0] == True and config["all_images_DIP"]["grid_search"][0] == "False"):
        raise ValueError("post reco mode need to save all images if ES")

def class_for_task(config,task):
    if (task == 'full_reco_with_network'): # Run DIPRecon or DNA
        from iADMM_DIP import iADMM_DIP
        classTask = iADMM_DIP(config)
        # raise ValueError("needs hyperparameters_config")
    elif (task == 'castor_reco'): # Run CASToR reconstruction with given optimizer
        from iComparison import iComparison
        classTask = iComparison(config)
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
        config["average_replicates"] = tune.grid_search([True])
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

# config_files = ["DNA_random_3_skip_10it", "DNA_CT_2_skip_10it", "DNA_CT_1_skip_10it"]
# config_files = ["DIPRecon_CT_1_skip", "DIPRecon_CT_2_skip"]#, "DIPRecon_CT_3_skip"]
# config_files = ["DIPRecon_CT_3_skip","DIPRecon_CT_1_skip","DIPRecon_CT_2_skip"]
# config_files = ["DIPRecon_CT_1_skip","DIPRecon_CT_2_skip"]
# config_files = ['APPGML_configuration']
# config_files = ['ADMMReg_configuration']
# config_files = ["DNA_ADMMReg_more_ADMMReg_it_10_configuration']
# config_files = ["DNA_APPGML_1it_configuration']
# config_files = ["DNA_ADMMReg_more_ADMMReg_it_30_configuration']
# config_files = ['OSEM_configuration']
# config_files = ['DIPRecon_skip3_3_my_settings']
# config_files = ['BSREM_configuration']
# config_files = 8*["DNA_MIC_dropout"]
# config_files = ["DNA_MIC_dropout"]
# config_files = ["DNA_MIC_cookie_2D"]
nb_computation = 5
config_files = 2*nb_computation*["DNA_MIC_brain_2D_diff5","DNA_MIC_brain_2D_diff5_SC2","DNA_MIC_brain_2D_diff5_SC1"]#,"DNA_MIC_brain_2D_diff1"]
# config_files = 2*nb_computation*["DNA_MIC_brain_2D_diff1"]
config_files = 2*nb_computation*["DNA_MIC_brain_2D_diff5"] #,"DNA_MIC_brain_2D_diff1"]
config_files = sorted(config_files)
# config_files = ["DNA_MIC_APPGML_brain_2D"]
# config_files = ["DNA_MIC_brain_2D_DNA_ADMMReg']
# config_files = ["DNA_MIC_cookie_2D_DNA_ADMMReg']
# config_files = ["DNA_MIC_several_inputs_brain_2D']
# config_files = [f[:-3] for f in os.listdir('all_config') if os.path.isfile(os.path.join('all_config', f))]

# config_files = ['ADMMReg_configuration']

i=-1
num_meth=0
for lib_string in config_files:
    if (i>=nb_computation): # Restart count for new setting
        i=-1
        num_meth +=1
    i+=1
    try:
    # if (True):
        lib = importlib.import_module('all_config.' + lib_string)
        config = lib.config_func_MIC()
        # config["image"] = tune.grid_search(['image4_0'])
        # config["image"] = tune.grid_search(['image010_3D'])
        if 'brain' in lib_string:
            config["image"] = tune.grid_search(['image50_1'])
            # config["image"] = tune.grid_search(['image40_1'])
        else:
            config["image"] = tune.grid_search(['image40_1'])
            # config["image"] = tune.grid_search(['image50_1'])
        config["replicates"] = tune.grid_search(list(range(1+int(40/nb_computation)*i,1+int(40/nb_computation)*i+int(40/nb_computation))))
        if (num_meth%2==0):
            config["nb_outer_iteration"] = tune.grid_search([10])
        elif (num_meth%2==1):
            config["nb_outer_iteration"] = tune.grid_search([2])
        else:
            raise ValueError("bug num_meth")
        # config["replicates"] = tune.grid_search(list(range(1,1+1)))
        # config["replicates"] = tune.grid_search([1])
        config["max_iter"] = tune.grid_search([300])
        # config["post_reco_in_suffix"] = tune.grid_search([False]) # If want to show EMV results which were not on post reconstruction, in DNA init
        # config["read_only_MV_csv"] = tune.grid_search([True])
        config["ray"] = True

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

            # classTask = iTradeoffCurves(config_without_grid_search)
            # config_without_grid_search["ray"] = False
            # classTask.config_with_grid_search = config
            # classTask.runRayTune(config_without_grid_search,root,task)

        '''
        classTask = iResultsADMMReg_VS_APPGML(config_without_grid_search)
        config_without_grid_search["ray"] = False
        classTask.runRayTune(config_without_grid_search,root,task)
        '''
        #sys.stdout.close()
        #sys.stdout=stdoutOrigin
    except:
        print(lib_string + " did not work")