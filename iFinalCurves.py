## Python libraries
# Math
from ast import Raise
from audioop import avg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Useful
from csv import reader as reader_csv
import re
from ray import tune
import os

# Local files to import
from vGeneral import vGeneral
from iResults import iResults
from iResultsAlreadyComputed import iResultsAlreadyComputed

class iFinalCurves(vGeneral):
    def __init__(self,config, *args, **kwargs):
        print("__init__")

    def initializeSpecific(self,config,root, *args, **kwargs):
        print("init")
        # show the plots in python or not !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def runComputation(self,config_all_methods,root):
        config_grid_search = self.config_with_grid_search
        method_list = config_all_methods["method"]

        MIC_config = True
        csv_before_MIC = False

        # Plot APGMAP vs ADMMLim (True)
        APGMAP_vs_ADMMLim = False

        # Plot tradeoff with SSIM (False) or AR (True)
        quantitative_tradeoff = False

        # Convert Gong to DIPRecon
        DIPRecon = False
        for i in range(len(method_list)):
            if "Gong" in method_list[i]:
                method_list[i] = method_list[i].replace("Gong","DIPRecon")
                if method_list[i] == "DIPRecon":
                    DIPRecon = True

        config = dict.fromkeys(method_list) # Create one config dictionnary for each method
        nb_rho = dict.fromkeys(method_list)
        nb_other_dim = dict.fromkeys(method_list)
        config_other_dim = dict.fromkeys(method_list)
        nb_replicates = dict.fromkeys(method_list)

        for method in method_list: # Loop over methods
            if (MIC_config):
                # Gong reconstruction
                if ('DIPRecon' in method):
                    print("configuration fiiiiiiiiiiiiiiiiiiile")
                    #config[method] = np.load(root + 'config_DIP.npy',allow_pickle='TRUE').item()
                    from Gong_configuration import config_func_MIC
                    #config[method] = config_func()
                    config[method] = config_func_MIC()
                    if ('stand' in method):
                        config[method]["scaling"] = {'grid_search': ["standardization"]}
                    elif ('norm' in method):
                        config[method]["scaling"] = {'grid_search': ["positive_normalization"]}
                    else:
                        if (DIPRecon):
                            config[method]["scaling"] = {'grid_search': ["standardization"]}
                        else:
                            raise ValueError("stand norm DIPRecon")

                    config[method]["method"] = "DIPRecon"
                    
                # nested reconstruction
                if ('nested' in method):
                    print("configuration fiiiiiiiiiiiiiiiiiiile")
                    from nested_configuration import config_func_MIC
                    #config[method] = config_func()
                    config[method] = config_func_MIC()
                    if ('ADMMLim' in method):
                        config[method]["max_iter"] = {'grid_search': [99]}
                    elif ('BSREM' in method):
                        config[method]["max_iter"] = {'grid_search': [300]}

                    config[method]["method"] = "nested"

                # MLEM reconstruction
                if (method == 'MLEM'):
                    print("configuration fiiiiiiiiiiiiiiiiiiile")
                    from MLEM_configuration import config_func_MIC
                    #config[method] = config_func()
                    config[method] = config_func_MIC()

                # OSEM reconstruction
                if (method == 'OSEM'):
                    print("configuration fiiiiiiiiiiiiiiiiiiile")
                    from OSEM_configuration import config_func_MIC
                    #config[method] = config_func()
                    config[method] = config_func_MIC()

                # BSREM reconstruction
                if (method == 'BSREM'):
                    print("configuration fiiiiiiiiiiiiiiiiiiile")
                    from BSREM_configuration import config_func_MIC
                    #config[method] = config_func()
                    config[method] = config_func_MIC()
                    config[method]["method"] = method

                # APGMAP reconstruction
                if ('APGMAP' in method):
                    APGMAP_vs_ADMMLim = True
                    print("configuration fiiiiiiiiiiiiiiiiiiile")
                    from APGMAP_configuration import config_func_MIC
                    #config[method] = config_func()
                    config[method] = config_func_MIC()
                    #for method2 in method_list: # Loop over methods
                    #    if ('APGMAP' not in method2 and 'ADMMLim' not in method2):
                    #        APGMAP_vs_ADMMLim = False
                            #config[method]['A_AML'] = {'grid_search': [100]}
                    config[method]["method"] = method

                # ADMMLim reconstruction
                if (method == 'ADMMLim'):
                    print("configuration fiiiiiiiiiiiiiiiiiiile")
                    from ADMMLim_configuration import config_func_MIC
                    #config[method] = config_func()
                    config[method] = config_func_MIC()



                # nested reconstruction
                if ('nested_ADMMLim_u_v' in method):
                    print("configuration fiiiiiiiiiiiiiiiiiiile")
                    from nested_ADMMLim_u_v_configuration import config_func_MIC
                    config[method] = config_func_MIC()
                    config[method]["method"] = "nested"
                    
                # nested reconstruction
                if ('nested_ADMMLim_more_ADMMLim_it_10' in method):
                    print("configuration fiiiiiiiiiiiiiiiiiiile")
                    from nested_ADMMLim_more_ADMMLim_it_10_configuration import config_func_MIC
                    config[method] = config_func_MIC()
                    config[method]["method"] = "nested"

                # nested reconstruction
                if (method == 'nested_ADMMLim_more_ADMMLim_it_30'):
                    print("configuration fiiiiiiiiiiiiiiiiiiile")
                    from nested_ADMMLim_more_ADMMLim_it_30_configuration import config_func_MIC
                    config[method] = config_func_MIC()
                    config[method]["method"] = "nested"

                # nested reconstruction
                if (method == 'nested_ADMMLim_more_ADMMLim_it_80'):
                    print("configuration fiiiiiiiiiiiiiiiiiiile")
                    from nested_ADMMLim_more_ADMMLim_it_80_configuration import config_func_MIC
                    config[method] = config_func_MIC()
                    config[method]["method"] = "nested"

                # nested reconstruction
                if ('nested_APPGML_4subsets' in method):
                    APGMAP_vs_ADMMLim = True
                    print("configuration fiiiiiiiiiiiiiiiiiiile")
                    from nested_APPGML_4subsets_configuration import config_func_MIC
                    config[method] = config_func_MIC()
                    config[method]["method"] = "nested"

                # nested reconstruction
                if (method == 'nested_APPGML_14subsets'):
                    print("configuration fiiiiiiiiiiiiiiiiiiile")
                    from nested_APPGML_14subsets_configuration import config_func_MIC
                    config[method] = config_func_MIC()
                    config[method]["method"] = "nested"
                
                # nested reconstruction
                if (method == 'nested_APPGML_28subsets'):
                    print("configuration fiiiiiiiiiiiiiiiiiiile")
                    from nested_APPGML_28subsets_configuration import config_func_MIC
                    config[method] = config_func_MIC()
                    config[method]["method"] = "nested"

                # nested reconstruction
                if (method == 'nested_APPGML_1it' or method == 'nested_APPGML_1subset'):
                    print("configuration fiiiiiiiiiiiiiiiiiiile")
                    from nested_APPGML_1it_configuration import config_func_MIC
                    config[method] = config_func_MIC()
                    config[method]["method"] = "nested"

                # nested reconstruction
                if ('nested_APPGML_4it' in method):
                    APGMAP_vs_ADMMLim = True
                    print("configuration fiiiiiiiiiiiiiiiiiiile")
                    from nested_APPGML_4it_configuration import config_func_MIC
                    config[method] = config_func_MIC()
                    config[method]["method"] = "nested"

                # nested reconstruction
                if (method == 'nested_APPGML_14it'):
                    print("configuration fiiiiiiiiiiiiiiiiiiile")
                    from nested_APPGML_14it_configuration import config_func_MIC
                    config[method] = config_func_MIC()
                    config[method]["method"] = "nested"
                
                # nested reconstruction
                if (method == 'nested_APPGML_28it'):
                    print("configuration fiiiiiiiiiiiiiiiiiiile")
                    from nested_APPGML_28it_configuration import config_func_MIC
                    config[method] = config_func_MIC()
                    config[method]["method"] = "nested"

                # nested reconstruction
                if (method == 'nested_CT_2_skip_3it'):
                    print("configuration fiiiiiiiiiiiiiiiiiiile")
                    from nested_CT_2_skip_3it import config_func_MIC
                    config[method] = config_func_MIC()
                    config[method]["method"] = "nested"


                # nested reconstruction
                if (method == 'nested_CT_3_skip_3it'):
                    print("configuration fiiiiiiiiiiiiiiiiiiile")
                    from nested_CT_3_skip_3it import config_func_MIC
                    config[method] = config_func_MIC()
                    config[method]["method"] = "nested"

                # nested reconstruction
                if (method == 'nested_CT_2_skip_10it'):
                    print("configuration fiiiiiiiiiiiiiiiiiiile")
                    from nested_CT_2_skip_10it import config_func_MIC
                    config[method] = config_func_MIC()
                    config[method]["method"] = "nested"


                # nested reconstruction
                if (method == 'nested_CT_3_skip_10it'):
                    print("configuration fiiiiiiiiiiiiiiiiiiile")
                    from nested_CT_3_skip_10it import config_func_MIC
                    config[method] = config_func_MIC()
                    config[method]["method"] = "nested"
            else:
                config[method] = self.config_with_grid_search
                config[method]["method"] = method

            # Initialize config files with good phantom        
            config[method]["image"] = {'grid_search': [config_all_methods["image"]]}
            # Initialize config files with good replicates
            #config[method]["replicates"] = {'grid_search': [config_all_methods["replicates"]]}
            # Launch task
            config_tmp = dict(config[method])
            config_tmp["method"] = tune.grid_search([method]) # Put only 1 method to remove useless hyperparameters from settings_config and hyperparameters_config
            os.system("rm -rf " + root + '/data/Algo/' + 'suffixes_for_last_run_' + method + '.txt')
            os.system("rm -rf " + root + '/data/Algo/' + 'replicates_for_last_run_' + method + '.txt')
            classTask = iResultsAlreadyComputed(config[method])
            task = 'show_metrics_results_already_computed'
            classTask.runRayTune(config_tmp,root,task) # Only to write suffix and replicate files
            #'''

            # Remove keyword "grid search" in config
        
            config[method] = dict(config[method])
            
            for key,value in config[method].items():
                if (type(value) == type(config[method])):
                    if ("grid_search" in value):
                        config[method][key] = value['grid_search']
                        if key != 'rho' and key != 'replicates' and key != 'method':
                            if key != 'A_AML' and key != 'post_smoothing' and key != 'lr':
                                config[method][key] = config[method][key][0]

        # Show figures for each ROI, with all asked methods
        import matplotlib
        font = {'family' : 'normal',
        'size'   : 14}
        matplotlib.rc('font', **font)

        if (quantitative_tradeoff):
            if (self.phantom == "image2_0"):
                ROI_list = ['cold','hot']
            elif (self.phantom == "image4_0" or self.phantom == "image4000_0" or self.phantom == "image40_0"):
                ROI_list = ['cold','hot_TEP','hot_TEP_match_square_recon','hot_perfect_match_recon']
        else:
            ROI_list = ['phantom']
        #for ROI in ['cold','hot']: #image2_0
        for ROI in ROI_list:
            # Initialize 3 figures
            fig, ax = [None] * 3, [None] * 3
            for fig_nb in range(3):
                fig[fig_nb], ax[fig_nb] = plt.subplots()

            replicates_legend = [None] * 3
            replicates_legend = [[],[],[]]

            for method in method_list: # Compute 
                if (method == 'ADMMLim' or 'DIPRecon' in method):
                    self.i_init = 30 # Remove first iterations
                    self.i_init = 20 # Remove first iterations
                    self.i_init = 1
                else:
                    self.i_init = 1

                self.defineTotalNbIter_beta_rho(method,config[method],task)



                # Initialize variables
                suffixes = []
                replicates = []

                PSNR_recon = []
                PSNR_norm_recon = []
                MSE_recon = []
                SSIM_recon = []
                MA_cold_recon = []
                AR_hot_recon = []
                AR_hot_TEP_recon = []
                AR_hot_TEP_match_square_recon = []
                AR_hot_perfect_match_recon = []
                AR_bkg_recon = []
                IR_bkg_recon = []
                IR_whole_recon = []

                IR_final = []
                metrics_final = []
                
                with open(root + '/data/Algo' + '/suffixes_for_last_run_' + method + '.txt') as f:
                    suffixes.append(f.readlines())
                with open(root + '/data/Algo' + '/replicates_for_last_run_' + method + '.txt') as f:
                    replicates.append(f.readlines())
                
                # Retrieve number of rhos and replicates and other dimension
                rho_name = "beta"
                nb_rho[method] = len(config[method]["rho"])
                if ("APGMAP" in method or method == "AML"):
                    config_other_dim[method] = config[method]["A_AML"]
                    other_dim_name = "A"
                elif (method == "MLEM" or method == "OSEM"):
                    config_other_dim[method] = config[method]["post_smoothing"]
                    rho_name = "smoothing"
                    other_dim_name = ""
                elif ("nested" in method or "DIPRecon" in method):
                    config_other_dim[method] = config[method]["lr"]
                    other_dim_name = "lr"
                else:
                    config_other_dim[method] = [""]
                    other_dim_name = ""
                    
                nb_other_dim[method] = len(config_other_dim[method])
                nb_replicates[method] = int(len(replicates[0]) / (nb_rho[method] * nb_other_dim[method]))

                # Sort rho and other dim like suffix
                config[method]["rho"] = sorted(config[method]["rho"])
                config_other_dim[method] = sorted(config_other_dim[method])


                # Settings in following curves
                variance_plot = False
                plot_all_replicates_curves = False
                
                if (self.phantom == "image2_0"):
                    color_dict = {
                        "nested" : ['red','pink'],
                        "DIPRecon" : ['cyan','blue','teal','blueviolet'],
                        "APGMAP" : ['darkgreen','lime','gold'],
                        "ADMMLim" : ['fuchsia'],
                        "OSEM" : ['darkorange'],
                        "BSREM" : ['grey']
                    }
                    color_dict_supp = {
                        "nested_BSREM_stand" : [color_dict["nested"][0]],
                        "nested_ADMMLim_stand" : [color_dict["nested"][1]],
                        "DIPRecon_BSREM_stand" : [color_dict["DIPRecon"][0]],
                        "DIPRecon_ADMMLim_stand" : [color_dict["DIPRecon"][1]],
                        "DIPRecon_ADMMLim_norm" : [color_dict["DIPRecon"][2]],
                        "DIPRecon_MLEM_norm" : [color_dict["DIPRecon"][3]],
                    }

                    color_dict = {**color_dict, **color_dict_supp} # Comparison between reconstruction methods
    
                elif(self.phantom == "image4_0" or self.phantom == "image4000_0" or self.phantom == "image40_0"):
                    color_dict_after_MIC = {
                        "nested_ADMMLim" : ['cyan','blue','teal','blueviolet'],
                        #"nested_APPGML_it" : ['darkgreen','lime','gold','darkseagreen'],
                        #"nested_APPGML_subsets" : ['darkgreen','lime','gold','darkseagreen'],
                        "nested_APPGML" : ['darkgreen','lime','gold','darkseagreen'],
                        "nested_CT_skip" : ['red','saddlebrown','lightsalmon','peru'],
                        "BSREM" : ['grey'],
                        "OSEM" : ['orange'],
                        #"APGMAP" : ['darkgreen','lime','gold'],
                        "APGMAP" : ['darkgreen','lime'],
                    }
                    color_dict_add_tests = {
                        "nested" : ['black'], # 3 it
                        "nested_ADMMLim_more_ADMMLim_it_10" : [color_dict_after_MIC["nested_ADMMLim"][0]],
                        "nested_ADMMLim_more_ADMMLim_it_30" : [color_dict_after_MIC["nested_ADMMLim"][1]],
                        "nested_ADMMLim_more_ADMMLim_it_80" : [color_dict_after_MIC["nested_ADMMLim"][2]],
                        "nested_ADMMLim_u_v" : [color_dict_after_MIC["nested_ADMMLim"][3]],
                        "nested_APPGML_1subset" : [color_dict_after_MIC["nested_APPGML"][0]],
                        "nested_APPGML_4subsets" : [color_dict_after_MIC["nested_APPGML"][1]],
                        "nested_APPGML_14subsets" : [color_dict_after_MIC["nested_APPGML"][2]],
                        "nested_APPGML_28subsets" : [color_dict_after_MIC["nested_APPGML"][3]],
                        "nested_APPGML_1it" : [color_dict_after_MIC["nested_APPGML"][0]],
                        "nested_APPGML_4it" : [color_dict_after_MIC["nested_APPGML"][1]],
                        "nested_APPGML_14it" : [color_dict_after_MIC["nested_APPGML"][2]],
                        "nested_APPGML_28it" : [color_dict_after_MIC["nested_APPGML"][3]],
                        "nested_CT_2_skip_3it" : [color_dict_after_MIC["nested_CT_skip"][0]],
                        "nested_CT_3_skip_3it" : [color_dict_after_MIC["nested_CT_skip"][1]],
                        "nested_CT_2_skip_10it" : [color_dict_after_MIC["nested_CT_skip"][2]],
                        "nested_CT_3_skip_10it" : [color_dict_after_MIC["nested_CT_skip"][3]],
                    }
    
                    color_dict = {**color_dict_after_MIC, **color_dict_add_tests} # Comparison between APPGML and ADMMLim in nested (varying subsets and iterations)

                if (self.phantom == "image2_0"):                    
                    marker_dict = {
                        "nested" : ['-','--'],
                        "DIPRecon" : ['-','--','loosely dotted','dashdot'],
                        "APGMAP" : ['-','--','loosely dotted'],
                        "ADMMLim" : ['-'],
                        "OSEM" : ['-'],
                        "BSREM" : ['-']
                    }
                    marker_dict_supp = {
                        "nested_BSREM_stand" : [marker_dict["nested"][0]],
                        "nested_ADMMLim_stand" : [marker_dict["nested"][1]],
                        "DIPRecon_BSREM_stand" : [marker_dict["DIPRecon"][0]],
                        "DIPRecon_ADMMLim_stand" : [marker_dict["DIPRecon"][1]],
                        "DIPRecon_ADMMLim_norm" : [marker_dict["DIPRecon"][2]],
                        "DIPRecon_MLEM_norm" : [marker_dict["DIPRecon"][3]],
                    }

                    marker_dict = {**marker_dict, **marker_dict_supp}
                elif(self.phantom == "image4_0" or self.phantom == "image4000_0" or self.phantom == "image40_0"):
                    marker_dict = {
                        "APPGML_it" : [':'],
                        "APPGML_subsets" : ['-'],
                        "ADMMLim" : ['--'],
                        "skip" : ['dashdot'],
                        "APGMAP" : ['-','-'],
                        "BSREM" : ['-'],
                        "OSEM" : ['-']
                    }
                    marker_dict_supp = {
                        "nested" : [marker_dict["ADMMLim"][0]], # 3 it
                        "nested_ADMMLim_more_ADMMLim_it_10" : [marker_dict["ADMMLim"][0]],
                        "nested_ADMMLim_more_ADMMLim_it_30" : [marker_dict["ADMMLim"][0]],
                        "nested_ADMMLim_more_ADMMLim_it_80" : [marker_dict["ADMMLim"][0]],
                        "nested_ADMMLim_u_v" : [marker_dict["ADMMLim"][0]],
                        "nested_APPGML_1subset" : [marker_dict["APPGML_subsets"][0]],
                        "nested_APPGML_4subsets" : [marker_dict["APPGML_subsets"][0]],
                        "nested_APPGML_14subsets" : [marker_dict["APPGML_subsets"][0]],
                        "nested_APPGML_28subsets" : [marker_dict["APPGML_subsets"][0]],
                        "nested_APPGML_1it" : [marker_dict["APPGML_it"][0]],
                        "nested_APPGML_4it" : [marker_dict["APPGML_it"][0]],
                        "nested_APPGML_14it" : [marker_dict["APPGML_it"][0]],
                        "nested_APPGML_28it" : [marker_dict["APPGML_it"][0]],
                        "nested_CT_2_skip_3it" : [marker_dict["skip"][0]],
                        "nested_CT_3_skip_3it" : [marker_dict["skip"][0]],
                        "nested_CT_2_skip_10it" : [marker_dict["skip"][0]],
                        "nested_CT_3_skip_10it" : [marker_dict["skip"][0]],
                    }
                    marker_dict = {**marker_dict, **marker_dict_supp}

                if plot_all_replicates_curves:
                    color_avg = 'black'
                    for key in color_dict.keys():
                        #for i in range(len(color_dict[key])):
                        color_dict[key] = len(color_dict[key]) * ['black']
                else:
                    color_avg = None
                    if (self.phantom == "image4_0" or self.phantom == "image4000_0" or self.phantom == "image40_0"):
                        color_avg = color_dict[method][0]    
                    

                # Wanted list of replicates
                idx_wanted = []
                for i in range(nb_rho[method]):
                    for p in range(nb_other_dim[method]):
                        idx_wanted += range(0,nb_replicates[method])

                # Check replicates from results are compatible with this script
                replicate_idx = [int(re.findall(r'(\w+?)(\d+)', replicates[0][idx].rstrip())[0][-1]) for idx in range(len(replicates[0]))]
                if list(np.sort(replicate_idx).astype(int)-1) != list(np.sort(idx_wanted)):
                    print(np.sort(idx_wanted))
                    print(np.sort(replicate_idx).astype(int)-1)
                    raise ValueError("Replicates are not the same for each case !")

                if method == method_list[-1]:
                    if not all(x == list(nb_replicates.values())[0] for x in list(nb_replicates.values())):
                        print(nb_replicates)
                        raise ValueError("Replicates are not the same for each method !")
                
                # Sort suffixes from file by rho and other dim values 
                sorted_suffixes = list(suffixes[0])
                if (method != "ADMMLim" and "nested" not in method and "APGMAP" not in method):
                    sorted_suffixes.sort(key=self.natural_keys)
                else:
                    sorted_suffixes.sort(key=self.natural_keys_ADMMLim)

                # Load metrics from last runs to merge them in one figure
                for i in range(len(sorted_suffixes)):
                    i_replicate = idx_wanted[i] # Loop over rhos and replicates, for each sorted rho, take sorted replicate
                    suffix = sorted_suffixes[i].rstrip("\n")
                    replicate = "replicate_" + str(i_replicate + 1)
                    metrics_file = root + '/data/Algo' + '/metrics/' + config[method]["image"] + '/' + str(replicate) + '/' + config[method]["method"] + '/' + suffix + '/' + 'metrics.csv'
                    
                    try:
                        with open(metrics_file, 'r') as myfile:
                            spamreader = reader_csv(myfile,delimiter=';')
                            rows_csv = list(spamreader)

                            '''
                            # if 1 out of i_init iterations was saved
                            rows_csv[0] = [float(rows_csv[0][i]) for i in range(int(self.i_init/self.i_init) - 1,min(len(rows_csv[0]),self.total_nb_iter))]
                            rows_csv[1] = [float(rows_csv[1][i]) for i in range(int(self.i_init/self.i_init) - 1, min(len(rows_csv[1]),self.total_nb_iter))]
                            rows_csv[2] = [float(rows_csv[2][i]) for i in range(int(self.i_init/self.i_init) - 1, min(len(rows_csv[2]),self.total_nb_iter))]
                            rows_csv[3] = [float(rows_csv[3][i]) for i in range(int(self.i_init/self.i_init) - 1, min(len(rows_csv[3]),self.total_nb_iter))]
                            rows_csv[4] = [float(rows_csv[4][i]) for i in range(int(self.i_init/self.i_init) - 1, min(len(rows_csv[4]),self.total_nb_iter))]
                            rows_csv[5] = [float(rows_csv[5][i]) for i in range(int(self.i_init/self.i_init) - 1, min(len(rows_csv[5]),self.total_nb_iter))]
                            rows_csv[6] = [float(rows_csv[6][i]) for i in range(int(self.i_init/self.i_init) - 1, min(len(rows_csv[6]),self.total_nb_iter))]
                            rows_csv[7] = [float(rows_csv[7][i]) for i in range(int(self.i_init/self.i_init) - 1, min(len(rows_csv[7]),self.total_nb_iter))]
                            '''
                            
                            rows_csv[0] = [float(rows_csv[0][i]) for i in range(int(self.i_init) - 1,min(len(rows_csv[0]),self.total_nb_iter))]
                            rows_csv[1] = [float(rows_csv[1][i]) for i in range(int(self.i_init) - 1, min(len(rows_csv[1]),self.total_nb_iter))]
                            rows_csv[2] = [float(rows_csv[2][i]) for i in range(int(self.i_init) - 1, min(len(rows_csv[2]),self.total_nb_iter))]
                            rows_csv[3] = [float(rows_csv[3][i]) for i in range(int(self.i_init) - 1, min(len(rows_csv[3]),self.total_nb_iter))]
                            rows_csv[4] = [float(rows_csv[4][i]) for i in range(int(self.i_init) - 1, min(len(rows_csv[4]),self.total_nb_iter))]
                            rows_csv[5] = [float(rows_csv[5][i]) for i in range(int(self.i_init) - 1, min(len(rows_csv[5]),self.total_nb_iter))]
                            rows_csv[6] = [float(rows_csv[6][i]) for i in range(int(self.i_init) - 1, min(len(rows_csv[6]),self.total_nb_iter))]
                            rows_csv[7] = [float(rows_csv[7][i]) for i in range(int(self.i_init) - 1, min(len(rows_csv[7]),self.total_nb_iter))]
                            if (not csv_before_MIC):
                                rows_csv[8] = [float(rows_csv[8][i]) for i in range(int(self.i_init) - 1, min(len(rows_csv[8]),self.total_nb_iter))]
                                rows_csv[9] = [float(rows_csv[9][i]) for i in range(int(self.i_init) - 1, min(len(rows_csv[9]),self.total_nb_iter))]
                                rows_csv[10] = [float(rows_csv[10][i]) for i in range(int(self.i_init) - 1, min(len(rows_csv[10]),self.total_nb_iter))]
                            
                            PSNR_recon.append(np.array(rows_csv[0]))
                            PSNR_norm_recon.append(np.array(rows_csv[1]))
                            MSE_recon.append(np.array(rows_csv[2]))
                            SSIM_recon.append(np.array(rows_csv[3]))
                            MA_cold_recon.append(np.array(rows_csv[4]) / 10 * 100)
                            AR_hot_recon.append(np.array(rows_csv[5]) / 400 * 100)

                            if (not csv_before_MIC):
                                AR_hot_TEP_recon.append(np.array(rows_csv[6]) / 400 * 100)
                                AR_hot_TEP_match_square_recon.append(np.array(rows_csv[7]) / 400 * 100)
                                AR_hot_perfect_match_recon.append(np.array(rows_csv[8]) / 400 * 100)
                                AR_bkg_recon.append(np.array(rows_csv[9]))
                                IR_bkg_recon.append(np.array(rows_csv[10]))
                                IR_whole_recon.append(np.array(rows_csv[13]))
                            else:        
                                AR_bkg_recon.append(np.array(rows_csv[6]))
                                IR_bkg_recon.append(np.array(rows_csv[7]))
                        
                            try:
                                MA_cold = np.array(rows_csv[8])
                            except:
                                MA_cold = np.array(rows_csv[6])
                            try:
                                AR_hot = np.array(rows_csv[9])
                            except:
                                AR_hot = np.array(rows_csv[7])

                    except:
                        print("No such file : " + metrics_file)

                
                if (method == "MLEM" or method == "OSEM"):
                    nb_rho[method], nb_other_dim[method] = nb_other_dim[method], nb_rho[method]
                    config[method]["rho"], config_other_dim[method] = config_other_dim[method], config[method]["rho"]


                # Select metrics to plot according to ROI
                if (quantitative_tradeoff):
                    if ROI == 'hot_TEP':
                        metrics = AR_hot_TEP_recon
                    elif ROI == 'hot_TEP_match_square_recon':
                        metrics = AR_hot_TEP_match_square_recon
                    elif ROI == 'hot_perfect_match_recon':
                        metrics = AR_hot_perfect_match_recon
                    elif ROI == 'hot':
                        #metrics = [abs(hot) for hot in AR_hot_recon] # Take absolute value of AR hot for tradeoff curves
                        metrics = AR_hot_recon
                    elif ROI == 'cold':
                        #metrics = [abs(cold) for cold in MA_cold_recon] # Take absolute value of MA cold for tradeoff curves
                        metrics = MA_cold_recon
                else:
                    metrics = SSIM_recon
                # Keep useful information to plot from metrics                
                IR_final = IR_bkg_recon
                metrics_final = metrics

                # Compute number of displayable iterations for each rho and find case with smallest iterations (useful for ADMMLim)
                len_mini_list = np.zeros((nb_rho[method],nb_other_dim[method],nb_replicates[method]),dtype=int)
                len_mini = np.zeros((nb_rho[method]),dtype=int)
                case_mini = np.zeros((nb_rho[method]),dtype=int)
                for rho_idx in range(nb_rho[method]):
                    for other_dim_idx in range(nb_other_dim[method]):
                        for replicate_idx in range(nb_replicates[method]):
                            len_mini_list[rho_idx,other_dim_idx,replicate_idx] = len(metrics_final[replicate_idx + nb_replicates[method]*other_dim_idx + (nb_replicates[method]*nb_other_dim[method])*rho_idx])
                        len_mini[rho_idx] = int(np.min(len_mini_list[rho_idx]))
                        case_mini[rho_idx] = int(np.argmin(len_mini_list[rho_idx,:,:])) + nb_replicates[method]*rho_idx

                # Create numpy array with same number of iterations for each case
                IR_final_array = []
                metrics_final_array = []
                for rho_idx in range(nb_rho[method]):
                    for other_dim_idx in range(nb_other_dim[method]):
                        IR_final_array.append(np.zeros((nb_replicates[method],len_mini[rho_idx])))
                        metrics_final_array.append(np.zeros((nb_replicates[method],len_mini[rho_idx])))
                        for common_it in range(len_mini[rho_idx]):
                            for replicate_idx in range(nb_replicates[method]):
                                IR_final_array[other_dim_idx+nb_other_dim[method]*rho_idx][replicate_idx,common_it] = IR_final[replicate_idx + nb_replicates[method]*other_dim_idx + (nb_replicates[method]*nb_other_dim[method])*rho_idx][common_it]
                                metrics_final_array[other_dim_idx+nb_other_dim[method]*rho_idx][replicate_idx,common_it] = metrics_final[replicate_idx + nb_replicates[method]*other_dim_idx + (nb_replicates[method]*nb_other_dim[method])*rho_idx][common_it]             

                
                IR_final_final_array = np.zeros((nb_rho[method],nb_other_dim[method],nb_replicates[method],np.max(len_mini)))
                metrics_final_final_array = np.zeros((nb_rho[method],nb_other_dim[method],nb_replicates[method],np.max(len_mini)))
                for rho_idx in range(nb_rho[method]):
                    for other_dim_idx in range(nb_other_dim[method]):
                        for common_it in range(len_mini[rho_idx]):
                            for replicate_idx in range(nb_replicates[method]):
                                IR_final_final_array[rho_idx,other_dim_idx,replicate_idx,common_it] = IR_final_array[other_dim_idx+nb_other_dim[method]*rho_idx][replicate_idx,common_it]
                                metrics_final_final_array[rho_idx,other_dim_idx,replicate_idx,common_it] = metrics_final_array[other_dim_idx+nb_other_dim[method]*rho_idx][replicate_idx,common_it]


                # Plot 3 figures for each ROI : tradeoff curve with iteration (metric VS IR), bias with iterations, and tradeoff curve at convergence
                reg = [None] * 3
                for fig_nb in range(3):
                    if (fig_nb == 0):
                        reg[fig_nb] = np.zeros((nb_rho[method]*nb_other_dim[method],np.max(len_mini)))
                    elif (fig_nb == 2):
                        if ("nested" not in method and "DIPRecon" not in method):
                            reg[fig_nb] = np.zeros((nb_rho[method]*nb_other_dim[method]))
                        else:
                            reg[fig_nb] = np.zeros((nb_rho[method]*nb_other_dim[method],np.max(len_mini)))
                    for rho_idx in range(nb_rho[method]):
                        for other_dim_idx in range(nb_other_dim[method]):
                            for replicate_idx in range(nb_replicates[method]):
                                case = replicate_idx + nb_replicates[method]*other_dim_idx + (nb_replicates[method]*nb_other_dim[method])*rho_idx
                                if (fig_nb == 0): # Plot tradeoff curves with iterations
                                    it = np.arange(len(IR_final[case_mini[rho_idx]]))
                                    if (plot_all_replicates_curves):
                                        ax[fig_nb].plot(100*IR_final_final_array[rho_idx,other_dim_idx,replicate_idx,it],metrics_final_final_array[rho_idx,other_dim_idx,replicate_idx,it],label='_nolegend_') # IR in %
                            
                            #'''
                            if (fig_nb == 0):
                                for it in range(len(IR_final[case_mini[rho_idx]])):
                                    reg[fig_nb][other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]] = self.linear_regression(100*IR_final_array[other_dim_idx+nb_other_dim[method]*rho_idx][:,it],metrics_final_array[other_dim_idx+nb_other_dim[method]*rho_idx][:,it])
                            #'''
                            for replicate_idx in range(nb_replicates[method]):
                                if (fig_nb == 1): # Plot bias curves
                                    if (plot_all_replicates_curves):
                                        #ax[fig_nb].plot(np.arange(0,len_mini[rho_idx])*self.i_init,metrics_final_final_array[rho_idx,other_dim_idx,replicate_idx,:len_mini[rho_idx]],label='_nolegend_') # IR in % # if 1 out of i_init iterations was saved
                                        ax[fig_nb].plot(np.arange(0,len_mini[rho_idx]),metrics_final_final_array[rho_idx,other_dim_idx,replicate_idx,:len_mini[rho_idx]],label='_nolegend_') # IR in %

                    avg_metrics = np.zeros((nb_rho[method]*nb_other_dim[method],np.max(len_mini)))
                    avg_IR = np.zeros((nb_rho[method]*nb_other_dim[method],np.max(len_mini)))
                    std_metrics = np.zeros((nb_rho[method]*nb_other_dim[method],np.max(len_mini)))
                    std_IR = np.zeros((nb_rho[method]*nb_other_dim[method],np.max(len_mini)))


                    #'''
                    for replicate_idx in range(nb_replicates[method]):
                        for other_dim_idx in range(nb_other_dim[method]):
                            if (fig_nb == 2): # Plot tradeoff curves at convergence
                                if (plot_all_replicates_curves):
                                    ax[fig_nb].plot(100*IR_final_final_array[:,other_dim_idx,replicate_idx,:][(np.arange(nb_rho[method]),len_mini-1)],metrics_final_final_array[:,other_dim_idx,replicate_idx,:][(np.arange(nb_rho[method]),len_mini-1)],label='_nolegend_') # IR in %
                    #'''

                    #'''
                    if (fig_nb == 2): # Plot tradeoff curves at convergence
                        for rho_idx in range(nb_rho[method]):
                            for other_dim_idx in range(nb_other_dim[method]):
                                if ("nested" not in method and "DIPRecon" not in method):
                                    reg[fig_nb][other_dim_idx+nb_other_dim[method]*rho_idx] = self.linear_regression(100*IR_final_array[other_dim_idx+nb_other_dim[method]*rho_idx][:,-1],metrics_final_array[other_dim_idx+nb_other_dim[method]*rho_idx][:,-1])
                                else:
                                    for it in range(len(IR_final[case_mini[rho_idx]])):
                                        reg[fig_nb][other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]] = self.linear_regression(100*IR_final_array[other_dim_idx+nb_other_dim[method]*rho_idx][:,it],metrics_final_array[other_dim_idx+nb_other_dim[method]*rho_idx][:,it])

                    #'''
                    #'''

                    
                    for rho_idx in range(nb_rho[method]):
                        for other_dim_idx in range(nb_other_dim[method]):
                            '''
                            # Compute average of tradeoff and bias curves with iterations
                            avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]] = np.sum(metrics_final_final_array[rho_idx,other_dim_idx,:,:len_mini[rho_idx]],axis=0) / nb_replicates[method]
                            avg_IR[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]] = np.sum(IR_final_final_array[rho_idx,other_dim_idx,:,:len_mini[rho_idx]],axis=0) / nb_replicates[method]
                            # Compute std bias curves with iterations
                            std_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]] = np.sqrt(np.sum((metrics_final_final_array[rho_idx,other_dim_idx,:,:len_mini[rho_idx]] - np.array(avg_metrics)[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]])**2,axis=0) / nb_replicates[method])
                            std_IR[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]] = np.sqrt(np.sum((IR_final_final_array[rho_idx,other_dim_idx,:,:len_mini[rho_idx]]- np.array(avg_IR)[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]])**2,axis=0) / nb_replicates[method])
                            '''

                            # Compute average of tradeoff and bias curves with iterations
                            # Remove NaNs from computation
                            nb_usable_replicates = np.count_nonzero(~np.isnan(metrics_final_final_array[rho_idx,other_dim_idx,:,len_mini[rho_idx] - 5]))

                            avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]] = np.nansum(metrics_final_final_array[rho_idx,other_dim_idx,:,:len_mini[rho_idx]],axis=0) / nb_usable_replicates
                            avg_IR[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]] = np.nansum(IR_final_final_array[rho_idx,other_dim_idx,:,:len_mini[rho_idx]],axis=0) / nb_usable_replicates
                            # Compute std bias curves with iterations
                            std_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]] = np.sqrt(np.nansum((metrics_final_final_array[rho_idx,other_dim_idx,:,:len_mini[rho_idx]] - np.array(avg_metrics)[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]])**2,axis=0) / nb_usable_replicates)
                            std_IR[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]] = np.sqrt(np.nansum((IR_final_final_array[rho_idx,other_dim_idx,:,:len_mini[rho_idx]]- np.array(avg_IR)[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]])**2,axis=0) / nb_usable_replicates)


                            if (fig_nb == 0):
                                ax[fig_nb].plot(100*avg_IR[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]],avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]],'-o',color=color_avg)
                                if (variance_plot):
                                    ax[fig_nb].fill(np.concatenate((100*(avg_IR[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]] - np.sign(reg[fig_nb])[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]]*std_IR[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]]),100*(avg_IR[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]][::-1] + np.sign(reg[fig_nb][other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]][::-1])*std_IR[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]][::-1]))),np.concatenate((avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]]-std_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]],avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]][::-1]+std_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]][::-1])), alpha = 0.4, label='_nolegend_')
                                ax[fig_nb].set_xlabel('Image Roughness (IR) in the background (%)')
                                ax[fig_nb].set_ylabel('Activity Recovery (AR) (%) ')
                                #ax[fig_nb].set_ylabel(('Bias ')
                                #ax[fig_nb].set_title('AR ' + 'in ' + ROI + ' region vs IR in background (with iterations)')
                            #'''
                            if (fig_nb == 1):
                                # Plot average and std of bias curves with iterations
                                #ax[fig_nb].plot(np.arange(0,len_mini[rho_idx])*self.i_init,avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]],color=color_avg) # if 1 out of i_init iterations was saved
                                ax[fig_nb].plot(np.arange(0,len_mini[rho_idx]),avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]],color=color_avg)
                                # Plot dashed line for target value, according to ROI
                                if (quantitative_tradeoff):
                                    target_value = 100
                                else:
                                    target_value = 1
                                if ROI == ROI_list[-1]:
                                    #ax[fig_nb].hlines(100,xmin=0,xmax=(len(avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx])-1)*self.i_init,color='grey',linestyle='dashed',label='_nolegend_') # if 1 out of i_init iterations was saved
                                    ax[fig_nb].hlines(target_value,xmin=0,xmax=(len(avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx])-1),color='grey',linestyle='dashed',label='_nolegend_')
                                else:
                                    #ax[fig_nb].hlines(100,xmin=0,xmax=(len(avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx])-1)*self.i_init,color='grey',linestyle='dashed',label='_nolegend_') # if 1 out of i_init iterations was saved
                                    ax[fig_nb].hlines(target_value,xmin=0,xmax=(len(avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx])-1),color='grey',linestyle='dashed',label='_nolegend_')
                                if (variance_plot):
                                    #ax[fig_nb].fill_between(np.arange(0,len(avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx]))*self.i_init, avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx] - std_metrics[other_dim_idx+nb_other_dim[method]*rho_idx], avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx] + std_metrics[other_dim_idx+nb_other_dim[method]*rho_idx], alpha = 0.4, label='_nolegend_') # if 1 out of i_init iterations was saved
                                    ax[fig_nb].fill_between(np.arange(0,len(avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx])), avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx] - std_metrics[other_dim_idx+nb_other_dim[method]*rho_idx], avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx] + std_metrics[other_dim_idx+nb_other_dim[method]*rho_idx], alpha = 0.4, label='_nolegend_')
                                ax[fig_nb].set_xlabel('Iterations')
                                ax[fig_nb].set_ylabel(('Activity Recovery (AR) (%) '))
                                #ax[fig_nb].set_ylabel(('Bias ')
                                '''
                                if len(method_list) == 1:
                                    ax[fig_nb].set_title(method + " reconstruction averaged on " + str(nb_usable_replicates) + " replicates (" + ROI + " ROI)")
                                else:
                                    ax[fig_nb].set_title("Several methods" + " reconstruction averaged on " + str(nb_usable_replicates) + " replicates (" + ROI + " ROI)")
                                '''
                            #'''
                            if (fig_nb != 2):
                                if (self.phantom == "image2_0"):
                                    replicates_legend[fig_nb].append(method + " : " + rho_name + " = " + str(config[method]["rho"][rho_idx]) + (", " + other_dim_name + " = " + str(config_other_dim[method][other_dim_idx]))*(other_dim_name!=""))
                                elif(self.phantom == "image4_0" or self.phantom == "image4000_0" or self.phantom == "image40_0"):
                                    if ("nested" not in method and "DIPRecon" not in method):
                                        replicates_legend[fig_nb].append(method + " : " + rho_name + " = " + str(config[method]["rho"][rho_idx]) + (", " + other_dim_name + " = " + str(config_other_dim[method][other_dim_idx]))*(other_dim_name!=""))
                                    else:
                                        replicates_legend[fig_nb].append(method)
                        
                    #'''
                    if (fig_nb == 2):
                        for other_dim_idx in range(nb_other_dim[method]):
                            if ("nested" not in method and "DIPRecon" not in method):
                                cases = np.arange(0,nb_other_dim[method]*nb_rho[method],nb_other_dim[method]) + other_dim_idx
                                
                                if ((not APGMAP_vs_ADMMLim and (method == "APGMAP" and other_dim_idx == 1) or (method != "APGMAP" and other_dim_idx == 0)) or APGMAP_vs_ADMMLim):
                                #    nb_other_dim["APGMAP"] = 1
                                    ax[fig_nb].plot(100*avg_IR[(cases,len_mini-1)],avg_metrics[(cases,len_mini-1)],'-o',linewidth=3,color=color_dict[method][other_dim_idx],ls=marker_dict[method][other_dim_idx])#'-o',)
                                if (variance_plot):
                                    ax[fig_nb].fill(np.concatenate((100*(avg_IR[(cases,len_mini-1)] - np.sign(reg[fig_nb][cases])*std_IR[cases,-1]),100*(avg_IR[(cases,len_mini-1)][::-1] + np.sign(reg[fig_nb][cases][::-1])*std_IR[(cases,len_mini-1)][::-1]))),np.concatenate((avg_metrics[(cases,len_mini-1)]-std_metrics[(cases,len_mini-1)],avg_metrics[(cases,len_mini-1)][::-1]+std_metrics[(cases,len_mini-1)][::-1])), alpha = 0.4, label='_nolegend_', ls=marker_dict[method][other_dim_idx])
                                # BSREM beta 0.01 white circle
                                #'''
                                if ('BSREM' in method):
                                    idx = 0
                                    plt.plot(100*avg_IR[(cases[idx],len_mini[idx]-1)],avg_metrics[(cases[idx],len_mini[idx]-1)],'X', color='white', label='_nolegend_')
                                    plt.plot(100*avg_IR[(cases[idx],len_mini[idx]-1)],avg_metrics[(cases[idx],len_mini[idx]-1)],marker='X',markersize=10, color='black', label='_nolegend_')
                                #'''
                            else:
                                print(method,color_dict[method])
                                #ax[fig_nb].plot(100*avg_IR[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]],avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]],linewidth=4,color=color_dict[method][other_dim_idx])#'-o',)
                                ax[fig_nb].plot(100*avg_IR[other_dim_idx+nb_other_dim[method]*rho_idx,0:len_mini[rho_idx]:25],avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,0:len_mini[rho_idx]:25],'-o',linewidth=3,color=color_dict[method][other_dim_idx],ls=marker_dict[method][other_dim_idx])#'-o',)
                                # unnested
                                plt.plot(100*avg_IR[other_dim_idx+nb_other_dim[method]*rho_idx,0],avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,0],'D',markersize=10, mfc='none',color=color_dict[method][other_dim_idx],label='_nolegend_')
                                plt.plot(100*avg_IR[other_dim_idx+nb_other_dim[method]*rho_idx,0],avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,0],marker='D',markersize=9,color='white',label='_nolegend_')
                                # nested it 100 white circle
                                #'''
                                if ('nested_BSREM_stand' in method):
                                    idx = 100
                                elif ('nested_ADMMLim_stand' in method):
                                    idx = 75
                                else:
                                    idx = 75
                                if ('nested_BSREM_stand' in method or "DIPRecon_BSREM_stand" in method):
                                    #plt.plot(100*avg_IR[other_dim_idx+nb_other_dim[method]*rho_idx,idx],avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,idx],'o', color='white', label='_nolegend_')
                                    plt.plot(100*avg_IR[other_dim_idx+nb_other_dim[method]*rho_idx,idx],avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,idx],marker='X',markersize=10,color='black', label='_nolegend_')                                   
                                #'''
                                #if 'cold' in ROI:
                                #    plt.xlim([12,57])
                                #else:
                                #    plt.xlim([12,57])
                                if (variance_plot):
                                    ax[fig_nb].fill(np.concatenate((100*(avg_IR[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]] - np.sign(reg[fig_nb])[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]]*std_IR[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]]),100*(avg_IR[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]][::-1] + np.sign(reg[fig_nb][other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]][::-1])*std_IR[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]][::-1]))),np.concatenate((avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]]-std_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]],avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]][::-1]+std_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]][::-1])), alpha = 0.4, label='_nolegend_')
                            if (APGMAP_vs_ADMMLim):
                                replicates_legend[fig_nb].append(method + (": " + other_dim_name + " = " + str(config_other_dim[method][other_dim_idx]))*(other_dim_name!=""))
                            elif ((not APGMAP_vs_ADMMLim and other_dim_idx == 0) or APGMAP_vs_ADMMLim):
                                if ("MLEM_norm" in method):
                                    replicates_legend[fig_nb].append(r'DIPRecon$_{init~MLEM}^{scal~norm}$')
                                elif ("ADMMLim_norm" in method):
                                    replicates_legend[fig_nb].append(r'DIPRecon$_{init~ADMMLim}^{scal~norm}$')
                                elif ("nested_ADMMLim_stand" in method):
                                    replicates_legend[fig_nb].append(r'nested$_{init~ADMMLim}^{scal~stand}$')
                                elif ("DIPRecon_ADMMLim_stand" in method):
                                    replicates_legend[fig_nb].append(r'DIPRecon$_{init~ADMMLim}^{scal~stand}$')
                                elif ("nested_BSREM_stand" in method):
                                    replicates_legend[fig_nb].append('nested')
                                elif ("DIPRecon_BSREM_stand" in method):
                                    replicates_legend[fig_nb].append('DIPRecon')
                                else:
                                    replicates_legend[fig_nb].append(method)
                        ax[fig_nb].set_xlabel('Image Roughness (IR) in the background (%)')
                        ax[fig_nb].set_ylabel(('Activity Recovery (AR) (%) '))
                        #ax[fig_nb].set_ylabel(('Bias '))
                        #ax[fig_nb].set_title(('AR ') + 'in ' + ROI + ' region vs IR in background (at convergence)')
                    #'''

                    if (method == method_list[-1]):
                        #if ROI == ROI_list[-1]: # if legend is needed only in one ROI
                        ax[fig_nb].legend(replicates_legend[fig_nb])#, prop={'size': 15})

            # Saving figures locally in png
            for fig_nb in range(3):
                if len(method_list) == 1:
                    rho = config[method]["rho"]
                    pretitle = method + " : " + rho_name + " = " + str(rho) + (", " + other_dim_name + " = " + str(config_other_dim[method]))*(other_dim_name!="")
                else:
                    pretitle = str(method_list)
                if (quantitative_tradeoff):
                    metric_AR_or_SSIM = 'AR'
                else:
                    metric_AR_or_SSIM = 'SSIM'
                if (fig_nb == 0):
                    title = pretitle + ' : ' + metric_AR_or_SSIM + ' ' + ' in ' + ROI + ' region vs IR in background (with iterations)' + '.png'
                elif (fig_nb == 1):
                    title = pretitle + ' : ' + metric_AR_or_SSIM + ' ' + ' in ' + ROI + ' region for ' + str(nb_usable_replicates) + ' replicates' + '.png'
                elif (fig_nb == 2):
                    title = pretitle + ' : ' + metric_AR_or_SSIM + ' ' + ' in ' + ROI + ' region vs IR in background (at convergence)' + '.png'
                
                try:
                    fig[fig_nb].savefig(self.subroot_data + 'metrics/' + self.phantom + '/' + title)
                except OSError:
                    print("File name too long, setting a shorter one")
                    fig[fig_nb].savefig(self.subroot_data + 'metrics/' + self.phantom + '/' + title[-250:])


            for method in method_list: # Loop over methods
                # Swap rho and post smoothing because MLEM and OSEM do not have rho parameter
                if (method == "MLEM" or method == "OSEM"):
                    nb_rho[method], nb_other_dim[method] = nb_other_dim[method], nb_rho[method]
                    config[method]["rho"], config_other_dim[method] = config_other_dim[method], config[method]["rho"]

