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
    if (config["method"]["grid_search"][0] == 'nested' and config["rho"]["grid_search"][0] == 0 and task == "castor_reco"):
        raise ValueError("nested must be launched with rho > 0")
    elif ((config["method"]["grid_search"][0] != 'Gong' and config["method"]["grid_search"][0] != 'nested') and task == "post_reco"):
        raise ValueError("Only Gong or nested can be run in post reconstruction mode, not CASToR reconstruction algorithms. Please comment this line.")
    elif ((config["method"]["grid_search"][0] == 'Gong' or config["method"]["grid_search"][0] == 'nested') and config["all_images_DIP"]["grid_search"][0] != "True" and config["DIP_early_stopping"]["grid_search"][0] == "True"):
        raise ValueError("Please set all_images_DIP to True to save all images for nested or Gong reconstruction if using WMV.")
    elif ((config["method"]["grid_search"][0] == 'Gong' or config["method"]["grid_search"][0] == 'nested') and config["rho"]["grid_search"][0] == 0 and task != "post_reco"):
        raise ValueError("Please set rho > 0 for nested or Gong reconstruction (or set task to post reconstruction).")
    elif (config["windowSize"]["grid_search"][0] >= config["sub_iter_DIP"]["grid_search"][0] and config["EMV_or_WMV"]["grid_search"][0] == "WMV"):
        raise ValueError("Please set window size less than number of DIP iterations for Window Moving Variance.")
    elif (config["debug"] and config["ray"]):
        raise ValueError("Debug mode must is used without ray")
    elif (task == "post_reco" and config["DIP_early_stopping"]["grid_search"][0] == True and config["all_images_DIP"]["grid_search"][0] == "False"):
        raise ValueError("post reco mode need to save all images if ES")

def class_for_task(config,task):
    if (task == 'full_reco_with_network'): # Run Gong or nested ADMM
        from iNestedADMM import iNestedADMM
        classTask = iNestedADMM(config)
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
    elif (task == 'show_metrics_ADMMLim'): # Show ADMMLim FOMs over iterations
        config["task"] = "show_results_post_reco"
        from iMeritsADMMLim import iMeritsADMMLim
        classTask = iMeritsADMMLim(config)
    elif (task == 'show_metrics_nested'): # Show nested or Gong FOMs over iterations
        from iMeritsNested import iMeritsNested
        classTask = iMeritsNested(config)
    # elif (task == 'show_metrics_results_already_computed'): # Show already computed results averaging over replicates
    #     from iResultsAlreadyComputed import iResultsAlreadyComputed
    #     classTask = iResultsAlreadyComputed(config)
    elif (task == 'compare_2_methods'): # Show already computed results averaging over replicates
        config["average_replicates"] = tune.grid_search([True])
        from iResultsADMMLim_VS_APGMAP import iResultsADMMLim_VS_APGMAP
        classTask = iResultsADMMLim_VS_APGMAP(config)

    return classTask

def choose_task(config):
    # Task to run reconstruction according to the method
    if (config["method"]["grid_search"][0] == 'Gong' or config["method"]["grid_search"][0] == 'nested'):
        task = 'full_reco_with_network'

    elif ('ADMMLim' in config["method"]["grid_search"][0] or config["method"]["grid_search"][0] == 'MLEM' or config["method"]["grid_search"][0] == 'OPTITR' or config["method"]["grid_search"][0] == 'OSEM' or config["method"]["grid_search"][0] == 'BSREM' or config["method"]["grid_search"][0] == 'AML' or config["method"]["grid_search"][0] == 'APGMAP'):
        task = 'castor_reco'

    # Override task here if needed
    # task = 'full_reco_with_network' # Run Gong or nested ADMM
    # task = 'castor_reco' # Run CASToR reconstruction with given optimizer
    # task = 'post_reco' # Run network denoising after a given reconstructed image im_corrupt
    # task = 'show_results_post_reco'
    # task = 'show_results'
    # task = 'show_metrics_results_already_computed'
    # task = 'show_metrics_ADMMLim'
    # task = 'show_metrics_nested'
    # task = 'compare_2_methods'

    return task

# config_files = ["nested_random_3_skip_10it", "nested_CT_2_skip_10it", "nested_CT_1_skip_10it"]
config_files = ["Gong_CT_1_skip", "Gong_CT_2_skip"]#, "Gong_CT_3_skip"]
config_files = ["Gong_CT_3_skip","Gong_CT_1_skip","Gong_CT_2_skip"]
config_files = ["Gong_CT_1_skip","Gong_CT_2_skip"]
config_files = ['APGMAP_configuration']
# config_files = ['ADMMLim_configuration']
config_files = ['nested_ADMMLim_more_ADMMLim_it_80_configuration']
# config_files = ['nested_ADMMLim_more_ADMMLim_it_10_configuration']
# config_files = ['nested_APPGML_1it_configuration']
# config_files = ['nested_ADMMLim_more_ADMMLim_it_30_configuration']
# config_files = ['OSEM_configuration']
# config_files = ['Gong_skip3_3_my_settings']
# config_files = ['BSREM_configuration']
# config_files = [f[:-3] for f in os.listdir('all_config') if os.path.isfile(os.path.join('all_config', f))]

# config_files = ['ADMMLim_configuration']

for lib_string in config_files:
    # try:
    if (True):
        lib = importlib.import_module('all_config.' + lib_string)
        config = lib.config_func_MIC()
        # config["image"] = tune.grid_search(['image4_0'])
        config["image"] = tune.grid_search(['image010_3D'])
        # config["image"] = tune.grid_search(['image50_1'])
        config["replicates"] = tune.grid_search(list(range(1,1+1)))
        # config["replicates"] = tune.grid_search([11])
        config["max_iter"] = tune.grid_search([100])
        # config["post_reco_in_suffix"] = tune.grid_search([False]) # If want to show EMV results which were not on post reconstruction, in DNA init
        # config["read_only_MV_csv"] = tune.grid_search([True])
        config["ray"] = False

        root = os.getcwd()

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

            # classTask = iFinalCurves(config_without_grid_search)
            # config_without_grid_search["ray"] = False
            # classTask.config_with_grid_search = config
            # classTask.runRayTune(config_without_grid_search,root,task)

        '''
        classTask = iResultsADMMLim_VS_APGMAP(config_without_grid_search)
        config_without_grid_search["ray"] = False
        classTask.runRayTune(config_without_grid_search,root,task)
        '''
        #sys.stdout.close()
        #sys.stdout=stdoutOrigin
    # except:
    #     print(lib_string + " did not work")