## Python libraries
# Math
import numpy as np
import matplotlib.pyplot as plt

# Useful
from csv import reader as reader_csv
import re
from ray import tune
import os

# Local files to import
from vGeneral import vGeneral
from iResultsAlreadyComputed import iResultsAlreadyComputed

class iFinalCurves(vGeneral):
    def __init__(self,config, *args, **kwargs):
        print("__init__")

    def initializeSpecific(self,config,root, *args, **kwargs):
        print("init")
        # show the plots in python or not !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def runComputation(self,config_all_methods,root):
        method_list = config_all_methods["method"]

        MIC_config = True
        csv_before_MIC = False

        # Plot APGMAP vs ADMMLim (True)
        APGMAP_vs_ADMMLim = False
        # Rename my settings (MIC)
        rename_settings = True

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
        config_tmp = dict.fromkeys(method_list)
        nb_replicates = dict.fromkeys(method_list)

        for method in method_list: # Loop over methods
            if (MIC_config):
                # Read config dictionnary for this method in config file
                config[method] = self.choose_good_config_file(method,config,csv_before_MIC,DIPRecon)
            else:
                config[method]["method"] = method

            if (method.endswith("_configuration")):
                method_without_configuration = method[:-14]
            else:
                method_without_configuration = method


            # Initialize config files with good phantom        
            config[method]["image"] = {'grid_search': [config_all_methods["image"]]}
            # Initialize config files with good replicates
            config[method]["replicates"] = {'grid_search': config_all_methods["replicates"]}
            # # Initialize config files with number of iterations
            # config[method]["max_iter"] = {'grid_search': [config_all_methods["max_iter"]]}
            # Launch task to write suffix and replicate files
            config_tmp[method] = dict(config[method])
            config_tmp[method]["method"] = tune.grid_search([method]) # Put only 1 method to remove useless hyperparameters from settings_config and hyperparameters_config
            config_tmp[method]["ray"] = True # Activate ray
            os.system("rm -rf " + root + '/data/Algo/' + 'suffixes_for_last_run_' + method + '.txt')
            os.system("rm -rf " + root + '/data/Algo/' + 'replicates_for_last_run_' + method + '.txt')
            classTask = iResultsAlreadyComputed(config[method])
            task = 'show_metrics_results_already_computed'
            classTask.runRayTune(config_tmp[method],root,task,only_suffix_replicate_file=True) # Only to write suffix and replicate files

            # Remove keyword "grid search" in config
            config[method] = dict(config[method])
            
            for key,value in config[method].items():
                if (type(value) == type(config[method])):
                    if ("grid_search" in value):
                        config[method][key] = value['grid_search']
                        if key != 'rho' and key != 'replicates' and key != 'method':
                            if key != 'A_AML' and key != 'post_smoothing' and key != 'lr':
                                if key != 'tau_DIP':
                                    config[method][key] = config[method][key][0]

        # Show figures for each ROI, with all asked methods
        import matplotlib
        font = {'family' : 'normal',
        'size'   : 14}
        matplotlib.rc('font', **font)

        if (self.phantom == "image2_0"):
            ROI_list = ['cold','hot','phantom']
        elif (self.phantom == "image4_0" or self.phantom == "image400_0" or self.phantom == "image40_0"):
            # ROI_list = ['cold','hot_TEP','hot_perfect_match_recon','phantom']
            # ROI_list = ['cold','hot_TEP','hot_perfect_match_recon','phantom','whole']
            # ROI_list = ['whole']
            # ROI_list = ['cold','cold_inside','cold_edge']
            ROI_list = ['cold']

        for ROI in ROI_list:
            # Plot tradeoff with SSIM (set quantitative_tradeoff is False) or AR (set quantitative_tradeoff to True)
            if ROI == 'phantom' or ROI == 'whole':
                quantitative_tradeoff = False
            else:
                quantitative_tradeoff = True
            # Initialize 3 figures
            fig, ax = [None] * 3, [None] * 3
            for fig_nb in range(3):
                fig[fig_nb], ax[fig_nb] = plt.subplots()

            replicates_legend = [None] * 3
            replicates_legend = [[],[],[]]

            for method in method_list: # Compute 

                if (method.endswith("_configuration")):
                    method_without_configuration = method[:-14]
                else:
                    method_without_configuration = method

                if (method == 'ADMMLim' or 'DIPRecon' in method):
                    self.i_init = 30 # Remove first iterations
                    self.i_init = 20 # Remove first iterations
                    self.i_init = 1
                else:
                    self.i_init = 1

                # if ('Gong' in method or 'nested' in method):
                #     self.i_init = 1 # 0 will take last value as first...

                # Initialize variables
                suffixes = []
                replicates = []

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
                    rho_name = "rho"
                    # config_other_dim[method] = config[method]["lr"]
                    # other_dim_name = "lr"
                    config_other_dim[method] = config_tmp[method]["tau_DIP"]["grid_search"]
                    other_dim_name = "tau_DIP"
                else:
                    config_other_dim[method] = [""]
                    other_dim_name = ""
                    
                nb_other_dim[method] = len(config_other_dim[method])
                nb_replicates[method] = int(len(replicates[0]) / (nb_rho[method] * nb_other_dim[method]))

                # Affect color and marker according to method and settings
                marker_dict, color_dict = self.marker_color_dict_method()

                # Sort rho and other dim like suffix
                config[method]["rho"] = sorted(config[method]["rho"])
                config_other_dim[method] = sorted(config_other_dim[method])

                # Settings in following curves
                variance_plot = False
                plot_all_replicates_curves = False
                
                if plot_all_replicates_curves:
                    color_avg = 'black'
                    for key in color_dict.keys():
                        #for i in range(len(color_dict[key])):
                        color_dict[key] = len(color_dict[key]) * ['black']
                else:
                    color_avg = None
                    if (self.phantom == "image4_0" or self.phantom == "image400_0" or self.phantom == "image40_0"):
                        color_avg = color_dict[method_without_configuration][0]    
                    

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
                if (method != "ADMMLim" and "nested" not in method and "APGMAP" not in method and method != "BSREM"):
                    sorted_suffixes.sort(key=self.natural_keys)
                else:
                    sorted_suffixes.sort(key=self.natural_keys_ADMMLim)

                # Load metrics from last runs to merge them in one figure
                metrics_final, IR_final = self.load_metrics(sorted_suffixes, idx_wanted, root, config, method, task, csv_before_MIC, quantitative_tradeoff, ROI)
                
                if (method == "MLEM" or method == "OSEM"):
                    nb_rho[method], nb_other_dim[method] = nb_other_dim[method], nb_rho[method]
                    config[method]["rho"], config_tmp[method]["rho"], config_other_dim[method] = config_other_dim[method], config_tmp[method]["post_smoothing"], config[method]["rho"]



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


                # Plot 3 figures for each ROI : tradeoff curve with iteration (metric VS IR), metric with iterations, and tradeoff curve at convergence
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
                                if (fig_nb == 0): # Plot tradeoff curves with all iterations
                                    it = np.arange(len(IR_final[case_mini[rho_idx]]))
                                    if (plot_all_replicates_curves):
                                        ax[fig_nb].plot(100*IR_final_final_array[rho_idx,other_dim_idx,replicate_idx,it],metrics_final_final_array[rho_idx,other_dim_idx,replicate_idx,it],label='_nolegend_') # IR in %
                            
                            #'''
                            if (fig_nb == 0):
                                for it in range(len(IR_final[case_mini[rho_idx]])):
                                    reg[fig_nb][other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]] = self.linear_regression(100*IR_final_array[other_dim_idx+nb_other_dim[method]*rho_idx][:,it],metrics_final_array[other_dim_idx+nb_other_dim[method]*rho_idx][:,it])
                            #'''
                            for replicate_idx in range(nb_replicates[method]):
                                if (fig_nb == 1): # Plot metrics with iterations
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
                            # nb_usable_replicates = np.count_nonzero(~np.isnan(metrics_final_final_array[rho_idx,other_dim_idx,:,len_mini[rho_idx] - 5]))
                            nb_usable_replicates = np.count_nonzero(~np.isnan(metrics_final_final_array[rho_idx,other_dim_idx,:,len_mini[rho_idx] - 1]))

                            avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]] = np.nansum(metrics_final_final_array[rho_idx,other_dim_idx,:,:len_mini[rho_idx]],axis=0) / nb_usable_replicates
                            avg_IR[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]] = np.nansum(IR_final_final_array[rho_idx,other_dim_idx,:,:len_mini[rho_idx]],axis=0) / nb_usable_replicates
                            # Compute std bias curves with iterations
                            std_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]] = np.sqrt(np.nansum((metrics_final_final_array[rho_idx,other_dim_idx,:,:len_mini[rho_idx]] - np.array(avg_metrics)[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]])**2,axis=0) / nb_usable_replicates)
                            std_IR[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]] = np.sqrt(np.nansum((IR_final_final_array[rho_idx,other_dim_idx,:,:len_mini[rho_idx]]- np.array(avg_IR)[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]])**2,axis=0) / nb_usable_replicates)


                            if (fig_nb == 0):
                                # ax[fig_nb].plot(100*avg_IR[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]],avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]],'-o',color=color_avg)
                                if ((('nested' in method or 'DIPRecon' in method) and nb_other_dim[method] == 1) or nb_rho[method] > 1 or config_other_dim[method] == [""]):
                                    idx_good_rho_color = config_tmp[method]["rho"]["grid_search"].index(config[method]["rho"][rho_idx])
                                else:
                                    idx_good_rho_color = config_other_dim[method].index(config[method][other_dim_name][other_dim_idx])
                                ax[fig_nb].plot(100*avg_IR[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]],avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]],'-o',color=color_dict[method_without_configuration][idx_good_rho_color],ls=marker_dict[method][other_dim_idx])
                                if (variance_plot):
                                    # ax[fig_nb].fill(np.concatenate((100*(avg_IR[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]] - np.sign(reg[fig_nb])[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]]*std_IR[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]]),100*(avg_IR[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]][::-1] + np.sign(reg[fig_nb][other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]][::-1])*std_IR[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]][::-1]))),np.concatenate((avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]]-std_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]],avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]][::-1]+std_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]][::-1])), alpha = 0.4, label='_nolegend_')
                                    ax[fig_nb].fill(np.concatenate((100*(avg_IR[other_dim_idx+nb_other_dim[method]*rho_idx,np.linspace(0,len_mini[rho_idx]-1,20).astype(int)] - np.sign(reg[fig_nb])[other_dim_idx+nb_other_dim[method]*rho_idx,np.linspace(0,len_mini[rho_idx]-1,20).astype(int)]*std_IR[other_dim_idx+nb_other_dim[method]*rho_idx,np.linspace(0,len_mini[rho_idx]-1,20).astype(int)]),100*(avg_IR[other_dim_idx+nb_other_dim[method]*rho_idx,np.linspace(0,len_mini[rho_idx]-1,20).astype(int)][::-1] + np.sign(reg[fig_nb][other_dim_idx+nb_other_dim[method]*rho_idx,np.linspace(0,len_mini[rho_idx]-1,20).astype(int)][::-1])*std_IR[other_dim_idx+nb_other_dim[method]*rho_idx,np.linspace(0,len_mini[rho_idx]-1,20).astype(int)][::-1]))),np.concatenate((avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,np.linspace(0,len_mini[rho_idx]-1,20).astype(int)]-std_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,np.linspace(0,len_mini[rho_idx]-1,20).astype(int)],avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,np.linspace(0,len_mini[rho_idx]-1,20).astype(int)][::-1]+std_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,np.linspace(0,len_mini[rho_idx]-1,20).astype(int)][::-1])), alpha = 0.4, label='_nolegend_',color=color_dict[method_without_configuration][idx_good_rho_color],ls=marker_dict[method][other_dim_idx])                                
                                #ax[fig_nb].set_title('AR ' + 'in ' + ROI + ' region vs IR in background (with iterations)')
                            #'''
                            if (fig_nb == 1):
                                # Plot average and std of bias curves with iterations
                                #ax[fig_nb].plot(np.arange(0,len_mini[rho_idx])*self.i_init,avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]],color=color_avg) # if 1 out of i_init iterations was saved
                                # ax[fig_nb].plot(np.arange(0,len_mini[rho_idx]),avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]],color=color_avg)
                                if ((('nested' in method or 'DIPRecon' in method) and nb_other_dim[method] == 1) or nb_rho[method] > 1 or config_other_dim[method] == [""]):
                                    idx_good_rho_color = config_tmp[method]["rho"]["grid_search"].index(config[method]["rho"][rho_idx])
                                else:
                                    idx_good_rho_color = config_other_dim[method].index(config[method][other_dim_name][other_dim_idx])
                                ax[fig_nb].plot(np.arange(0,len_mini[rho_idx]),avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]],color=color_dict[method_without_configuration][idx_good_rho_color],ls=marker_dict[method][other_dim_idx])
                                # Plot dashed line for target value, according to ROI
                                if (ROI != "whole"):
                                    if (quantitative_tradeoff):
                                        if ("cold" in ROI):
                                            target_value = 0
                                        else:
                                            target_value = 100
                                    else:
                                        target_value = 1
                                    if ROI == ROI_list[-1]:
                                        #ax[fig_nb].hlines(100,xmin=0,xmax=(len(avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx])-1)*self.i_init,color='grey',linestyle='dashed',label='_nolegend_') # if 1 out of i_init iterations was saved
                                        ax[fig_nb].hlines(target_value,xmin=0,xmax=(len(avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx])-1),color='grey',linestyle='dashed',label='_nolegend_')
                                    else:
                                        #ax[fig_nb].hlines(100,xmin=0,xmax=(len(avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx])-1)*self.i_init,color='grey',linestyle='dashed',label='_nolegend_') # if 1 out of i_init iterations was saved
                                        ax[fig_nb].hlines(target_value,xmin=0,xmax=(len(avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx])-1),color='grey',linestyle='dashed',label='_nolegend_')
                                # Show variance shadow over average line if asked for
                                if (variance_plot):
                                    #ax[fig_nb].fill_between(np.arange(0,len(avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx]))*self.i_init, avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx] - std_metrics[other_dim_idx+nb_other_dim[method]*rho_idx], avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx] + std_metrics[other_dim_idx+nb_other_dim[method]*rho_idx], alpha = 0.4, label='_nolegend_') # if 1 out of i_init iterations was saved
                                    ax[fig_nb].fill_between(np.arange(0,len(avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx])), avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx] - std_metrics[other_dim_idx+nb_other_dim[method]*rho_idx], avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx] + std_metrics[other_dim_idx+nb_other_dim[method]*rho_idx], alpha = 0.4, label='_nolegend_',color=color_dict[method_without_configuration][idx_good_rho_color],ls=marker_dict[method][other_dim_idx])
                                '''
                                if len(method_list) == 1:
                                    ax[fig_nb].set_title(method + " reconstruction averaged on " + str(nb_usable_replicates) + " replicates (" + ROI + " ROI)")
                                else:
                                    ax[fig_nb].set_title("Several methods" + " reconstruction averaged on " + str(nb_usable_replicates) + " replicates (" + ROI + " ROI)")
                                '''
                            #'''                        
                    #'''
                    if (fig_nb == 2):
                        if ("nested" not in method and "DIPRecon" not in method):
                            # if ("APGMAP" in method):
                            #     for other_dim_idx in range(nb_other_dim[method]):
                            # else:
                            #     for rho_idx in range(nb_rho[method]):
                            for other_dim_idx in range(nb_other_dim[method]):
                                cases = np.arange(0,nb_other_dim[method]*nb_rho[method],nb_other_dim[method]) + other_dim_idx
                                
                                if ((not APGMAP_vs_ADMMLim and (method == "APGMAP" and other_dim_idx == 1) or (method != "APGMAP" and other_dim_idx == 0)) or APGMAP_vs_ADMMLim):
                                #    nb_other_dim["APGMAP"] = 1
                                    ax[fig_nb].plot(100*avg_IR[(cases,len_mini-1)],avg_metrics[(cases,len_mini-1)],'-o',linewidth=3,color=color_dict[method_without_configuration][other_dim_idx],ls=marker_dict[method][other_dim_idx])#'-o',)
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
                            for rho_idx in range(nb_rho[method]):
                                for other_dim_idx in range(nb_other_dim[method]):
                                    if (nb_other_dim[method] == 1):
                                        idx_good_rho_color = config_tmp[method]["rho"]["grid_search"].index(config[method]["rho"][rho_idx])
                                    else:
                                        idx_good_rho_color = config_other_dim[method].index(config[method][other_dim_name][other_dim_idx])
                                    ax[fig_nb].plot(100*avg_IR[other_dim_idx+nb_other_dim[method]*rho_idx,np.linspace(0,len_mini[rho_idx]-1,20).astype(int)],avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,np.linspace(0,len_mini[rho_idx]-1,20).astype(int)],marker='o'*('CT' in method) + '*'*('random' in method) + 'o'*('CT' not in method and 'random' not in method),linewidth=3,color=color_dict[method_without_configuration][idx_good_rho_color],ls=marker_dict[method][other_dim_idx])#'-o',)
                                    # ax[fig_nb].plot(100*avg_IR[other_dim_idx+nb_other_dim[method]*rho_idx,:],avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:],marker='o'*('CT' in method) + '*'*('random' in method) + '+'*('CT' not in method and 'random' not in method),linewidth=3,color=color_dict[method_without_configuration][other_dim_idx],ls=marker_dict[method][other_dim_idx])#'-o',)
                                    # unnested
                                    idx_good_rho_color = config_tmp[method]["rho"]["grid_search"].index(config[method]["rho"][rho_idx])
                                    plt.plot(100*avg_IR[other_dim_idx+nb_other_dim[method]*rho_idx,0],avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,0],'D',markersize=10, mfc='none',color=color_dict[method_without_configuration][idx_good_rho_color],label='_nolegend_')
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
                                    # plt.ylim([3.42e6,3.43e6])
                                    #'''
                                    #if 'cold' in ROI:
                                    #    plt.xlim([12,57])
                                    #else:
                                    #    plt.xlim([12,57])
                                    if (variance_plot):
                                        # ax[fig_nb].fill(np.concatenate((100*(avg_IR[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]] - np.sign(reg[fig_nb])[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]]*std_IR[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]]),100*(avg_IR[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]][::-1] + np.sign(reg[fig_nb][other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]][::-1])*std_IR[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]][::-1]))),np.concatenate((avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]]-std_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]],avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]][::-1]+std_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,:len_mini[rho_idx]][::-1])), alpha = 0.4, label='_nolegend_')
                                        ax[fig_nb].fill(np.concatenate((100*(avg_IR[other_dim_idx+nb_other_dim[method]*rho_idx,np.linspace(0,len_mini[rho_idx]-1,20).astype(int)] - np.sign(reg[fig_nb])[other_dim_idx+nb_other_dim[method]*rho_idx,np.linspace(0,len_mini[rho_idx]-1,20).astype(int)]*std_IR[other_dim_idx+nb_other_dim[method]*rho_idx,np.linspace(0,len_mini[rho_idx]-1,20).astype(int)]),100*(avg_IR[other_dim_idx+nb_other_dim[method]*rho_idx,np.linspace(0,len_mini[rho_idx]-1,20).astype(int)][::-1] + np.sign(reg[fig_nb][other_dim_idx+nb_other_dim[method]*rho_idx,np.linspace(0,len_mini[rho_idx]-1,20).astype(int)][::-1])*std_IR[other_dim_idx+nb_other_dim[method]*rho_idx,np.linspace(0,len_mini[rho_idx]-1,20).astype(int)][::-1]))),np.concatenate((avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,np.linspace(0,len_mini[rho_idx]-1,20).astype(int)]-std_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,np.linspace(0,len_mini[rho_idx]-1,20).astype(int)],avg_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,np.linspace(0,len_mini[rho_idx]-1,20).astype(int)][::-1]+std_metrics[other_dim_idx+nb_other_dim[method]*rho_idx,np.linspace(0,len_mini[rho_idx]-1,20).astype(int)][::-1])), alpha = 0.4, label='_nolegend_')
                    #'''
                    # Set labels for x and y axes
                    self.set_axes_labels(ax,fig_nb,ROI)
                    # Add label for each curve
                    if (fig_nb != 2): 
                        for rho_idx in range(nb_rho[method]):
                            for other_dim_idx in range(nb_other_dim[method]):
                                self.label_method_plot(replicates_legend,fig_nb,method,rho_name,nb_rho,nb_other_dim,rho_idx,other_dim_name,other_dim_idx,config,config_other_dim,APGMAP_vs_ADMMLim,rename_settings)
                    else: # Do not loop on rho because here is at convergence
                        self.label_method_plot(replicates_legend,fig_nb,method,rho_name,nb_rho,nb_other_dim,rho_idx,other_dim_name,other_dim_idx,config,config_other_dim,APGMAP_vs_ADMMLim,rename_settings)
                    if (method == method_list[-1]):
                        legend_this_ROI = False
                        if (quantitative_tradeoff): # AR
                            if (len(ROI_list) > 2):
                                if ROI == ROI_list[2] or ROI == ROI_list[-1]: # if legend is needed only in one ROI
                                    legend_this_ROI = True
                            else:
                                legend_this_ROI = True
                            if (legend_this_ROI):
                                ax[fig_nb].legend(replicates_legend[fig_nb], prop={'size': 12})
                        else: # SSIM
                            ax[fig_nb].legend(replicates_legend[fig_nb])#, prop={'size': 15})

            # Saving figures locally in png
            for fig_nb in range(3):
                if len(method_list) == 1:
                    rho = config[method]["rho"]
                    pretitle = method + " : " + rho_name + " = " + str(rho) + (", " + other_dim_name + " = " + str(config_other_dim[method]))*(other_dim_name!="")
                else:
                    pretitle = str(method_list)
                if (quantitative_tradeoff):
                    if ('cold' in ROI):
                        metric_AR_or_SSIM = 'bias'
                    else:
                        metric_AR_or_SSIM = 'AR'
                else:
                    if (ROI == "whole"):
                        metric_AR_or_SSIM = 'likelihood'
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
                    config[method]["rho"], config_tmp[method]["rho"], config_other_dim[method] = config_other_dim[method], config_tmp[method]["post_smoothing"], config[method]["rho"]


    def set_axes_labels(self,ax,fig_nb,ROI):
        if (fig_nb == 1):
            ax[fig_nb].set_xlabel('Iterations')
        else:
            ax[fig_nb].set_xlabel('Image Roughness (IR) in the background (%)')
        if 'cold' in ROI:
            ax[fig_nb].set_ylabel('Bias (%)')
        elif 'hot' in ROI:
            ax[fig_nb].set_ylabel('Activity Recovery (AR) (%) ')
        elif ROI == "whole":
            ax[fig_nb].set_ylabel('log-likelihood')
        else:
            ax[fig_nb].set_ylabel('SSIM')

    def label_method_plot(self,replicates_legend,fig_nb,method,rho_name,nb_rho,nb_other_dim,rho_idx,other_dim_name,other_dim_idx,config,config_other_dim,APGMAP_vs_ADMMLim,rename_settings):
        if (self.phantom == "image2_0"):
            replicates_legend[fig_nb].append(method + " : " + rho_name + " = " + str(config[method]["rho"][rho_idx]) + (", " + other_dim_name + " = " + str(config_other_dim[method][other_dim_idx]))*(other_dim_name!=""))
        elif(self.phantom == "image4_0" or self.phantom == "image400_0" or self.phantom == "image40_0"):
            if ("nested" not in method and "DIPRecon" not in method):
                if (fig_nb != 2):
                    replicates_legend[fig_nb].append(method + " : " + rho_name + " = " + str(config[method]["rho"][rho_idx]) + (", " + other_dim_name + " = " + str(config_other_dim[method][other_dim_idx]))*(other_dim_name!=""))
                else:
                    replicates_legend[fig_nb].append(method)
            else:
                # replicates_legend[fig_nb].append(method)
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
                if (rename_settings):
                    if ("random" in method):
                        replicates_legend[fig_nb].append("random input, " + str(config[method]["skip_connections"]) + " SC")
                    elif ("CT" in method):
                        replicates_legend[fig_nb].append("anatomical input, " + str(config[method]["skip_connections"]) + " SC")
                    elif ("DD" in method):
                        replicates_legend[fig_nb].append("random input, DD")
                    else:
                        for rho_idx in range(nb_rho[method]):
                            for other_dim_idx in range(nb_other_dim[method]):
                                label_name = "intermediate setting, " + str(config[method]["skip_connections"]) + " SC"
                                if (nb_other_dim[method] > 1): # Remove rho from label if other dim
                                    label_name += " : " + other_dim_name + " = " + str(config_other_dim[method][other_dim_idx])
                                else:
                                    label_name += " : " + rho_name + " = " + str(config[method]["rho"][rho_idx])
                                replicates_legend[fig_nb].append(label_name)
        

    def choose_good_config_file(self,method,config,csv_before_MIC,DIPRecon):
        # Gong reconstruction
        if (csv_before_MIC and 'DIPRecon' in method):
            #config[method] = np.load(root + 'config_DIP.npy',allow_pickle='TRUE').item()
            from all_config.Gong_configuration import config_func_MIC
            #config[method] = config_func()
            if (csv_before_MIC):
                if ('stand' in method):
                    config[method]["scaling"] = {'grid_search': ["standardization"]}
                elif ('norm' in method):
                    config[method]["scaling"] = {'grid_search': ["positive_normalization"]}
                else:
                    if (DIPRecon):
                        config[method]["scaling"] = {'grid_search': ["standardization"]}
                    else:
                        raise ValueError("stand norm DIPRecon")
            else:
                from all_config.Gong_configuration import config_func_MIC
                config[method] = config_func_MIC()
                # method_name = "DIPRecon"
                method_name = "Gong"

            #method_name = "DIPRecon"
            method_name = "Gong"
            
        # nested reconstruction
        if ('nested' in method):
            if(csv_before_MIC):
                from all_config.nested_configuration import config_func_MIC
                #config[method] = config_func()
                if ('ADMMLim' in method):
                    config[method]["max_iter"] = {'grid_search': [99]}
                elif ('BSREM' in method):
                    config[method]["max_iter"] = {'grid_search': [300]}

                method_name = "nested"

        # MLEM reconstruction
        if (method == 'MLEM'):
            from all_config.MLEM_configuration import config_func_MIC
            #config[method] = config_func()

        # OSEM reconstruction
        if (method == 'OSEM'):
            from all_config.OSEM_configuration import config_func_MIC
            #config[method] = config_func()

        # BSREM reconstruction
        if (method == 'BSREM'):
            from all_config.BSREM_configuration import config_func_MIC
            #config[method] = config_func()
            method_name = method

        # APGMAP reconstruction
        if ('APGMAP' in method):
            APGMAP_vs_ADMMLim = True
            from all_config.APGMAP_configuration import config_func_MIC
            #config[method] = config_func()
            #for method2 in method_list: # Loop over methods
            #    if ('APGMAP' not in method2 and 'ADMMLim' not in method2):
            #        APGMAP_vs_ADMMLim = False
                    #config[method]['A_AML'] = {'grid_search': [100]}
            method_name = method

        # ADMMLim reconstruction
        if (method == 'ADMMLim'):
            from all_config.ADMMLim_configuration import config_func_MIC
            #config[method] = config_func()



        # nested reconstruction
        if ('nested_ADMMLim_u_v' in method):
            from all_config.nested_ADMMLim_u_v_configuration import config_func_MIC
            method_name = "nested"
            
        # nested reconstruction
        if ('nested_ADMMLim_more_ADMMLim_it_10' in method):
            from all_config.nested_ADMMLim_more_ADMMLim_it_10_configuration import config_func_MIC
            method_name = "nested"

        # nested reconstruction
        if (method == 'nested_ADMMLim_more_ADMMLim_it_30'):
            from all_config.nested_ADMMLim_more_ADMMLim_it_30_configuration import config_func_MIC
            method_name = "nested"

        # nested reconstruction
        if (method == 'nested_ADMMLim_more_ADMMLim_it_80'):
            from all_config.nested_ADMMLim_more_ADMMLim_it_80_configuration import config_func_MIC
            method_name = "nested"

        # nested reconstruction
        if ('nested_APPGML_4subsets' in method):
            APGMAP_vs_ADMMLim = True
            from all_config.nested_APPGML_4subsets_configuration import config_func_MIC
            method_name = "nested"

        # nested reconstruction
        if (method == 'nested_APPGML_14subsets'):
            from all_config.nested_APPGML_14subsets_configuration import config_func_MIC
            method_name = "nested"
        
        # nested reconstruction
        if (method == 'nested_APPGML_28subsets'):
            from all_config.nested_APPGML_28subsets_configuration import config_func_MIC
            method_name = "nested"

        # nested reconstruction
        if (method == 'nested_APPGML_1it' or method == 'nested_APPGML_1subset'):
            from all_config.nested_APPGML_1it_configuration import config_func_MIC
            method_name = "nested"

        # nested reconstruction
        if ('nested_APPGML_4it' in method):
            APGMAP_vs_ADMMLim = True
            from all_config.nested_APPGML_4it_configuration import config_func_MIC
            method_name = "nested"

        # nested reconstruction
        if (method == 'nested_APPGML_14it'):
            from all_config.nested_APPGML_14it_configuration import config_func_MIC
            method_name = "nested"
        
        # nested reconstruction
        if (method == 'nested_APPGML_28it'):
            from all_config.nested_APPGML_28it_configuration import config_func_MIC
            method_name = "nested"

        # # nested reconstruction
        # if (method == 'nested_CT_0_skip_3it'):
        #     from all_config.nested_CT_0_skip_3it import config_func_MIC
        #     method_name = "nested"

        # # nested reconstruction
        # if (method == 'nested_CT_1_skip_3it'):
        #     from all_config.nested_CT_1_skip_3it import config_func_MIC
        #     method_name = "nested"

        # # nested reconstruction
        # if (method == 'nested_CT_2_skip_3it'):
        #     from all_config.nested_CT_2_skip_3it import config_func_MIC
        #     method_name = "nested"

        # # nested reconstruction
        # if (method == 'nested_CT_3_skip_3it'):
        #     from all_config.nested_CT_3_skip_3it import config_func_MIC
        #     method_name = "nested"

        # # nested reconstruction
        # if (method == 'nested_CT_0_skip_10it'):
        #     from all_config.nested_CT_0_skip_10it import config_func_MIC
        #     method_name = "nested"

        # # nested reconstruction
        # if (method == 'nested_CT_1_skip_10it'):
        #     from all_config.nested_CT_1_skip_10it import config_func_MIC
        #     method_name = "nested"

        # # nested reconstruction
        # if (method == 'nested_CT_2_skip_10it'):
        #     from all_config.nested_CT_2_skip_10it import config_func_MIC
        #     method_name = "nested"

        # # nested reconstruction
        # if (method == 'nested_CT_3_skip_10it'):
        #     from all_config.nested_CT_3_skip_10it import config_func_MIC
        #     method_name = "nested"

        # nested reconstruction
        if ('nested_random' in method or 'nested_CT' in method or 'nested_DD' in method):
            # from all_config.nested_random_3_skip_10it import config_func_MIC
            # import_str = "from all_config." + method + " import config_func_MIC"
            # exec(import_str,globals())
            import importlib
            globals().update(importlib.import_module('all_config.' + method).__dict__) 
            method_name = "nested"

        # # Gong reconstruction
        # if (method == 'DIPRecon_CT_1_skip'):
        #     from all_config.Gong_CT_1_skip import config_func_MIC
        #     method_name = "Gong"

        # # Gong reconstruction
        # if (method == 'DIPRecon_CT_2_skip'):
        #     from all_config.Gong_CT_2_skip import config_func_MIC
        #     method_name = "Gong"

        # # Gong reconstruction
        # if (method == 'DIPRecon_CT_3_skip'):
        #     from all_config.Gong_CT_3_skip import config_func_MIC
        #     method_name = "Gong"

        # DIPRecon reconstruction
        if ('DIPRecon_' in method):
            # from all_config.nested_random_3_skip_10it import config_func_MIC
            # import_str = "from all_config." + method + " import config_func_MIC"
            # exec(import_str,globals())
            import importlib
            globals().update(importlib.import_module('all_config.' + 'Gong' + method[8:]).__dict__) 
            method_name = "Gong"

        try:
            config[method] = config_func_MIC()
            method_name = method
            if ("OSEM" in method):
                method_name = "MLEM"
        except:
            import importlib
            if 'Gong' in method:
                globals().update(importlib.import_module('all_config.' + 'Gong' + method[8:]).__dict__)
                method_name = "Gong"
            elif 'nested' in method:
                globals().update(importlib.import_module('all_config.' + method).__dict__)
                method_name = "nested"
            else:
                globals().update(importlib.import_module('all_config.' + method).__dict__)
                method_name = method
            config[method] = config_MIC
        config[method]["method"] = method_name

        return config[method]

    def marker_color_dict_method(self):
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

        elif(self.phantom == "image4_0" or self.phantom == "image400_0" or self.phantom == "image40_0"):
            color_dict_after_MIC = {
                "nested_ADMMLim" : ['cyan','blue','teal','blueviolet','black'],
                #"nested_APPGML_it" : ['darkgreen','lime','gold','darkseagreen'],
                #"nested_APPGML_subsets" : ['darkgreen','lime','gold','darkseagreen'],
                "nested_APPGML" : ['darkgreen','lime','gold','darkseagreen'],
                "nested_CT_skip" : ['red','saddlebrown','blueviolet','lime','black','yellow','grey','peru'],
                "nested_random_skip" : ['fuchsia','orange','darkgreen','pink','black'],
                # "DIPRecon" : ['cyan','blue','teal','blueviolet'],
                "DIPRecon" : ['red','saddlebrown','blueviolet','lime','black','yellow','grey','peru'],
                "BSREM" : 5*['grey','cyan','blue','teal','blueviolet'],
                # "BSREM" : ['grey'],
                "OSEM" : ['orange'],
                #"APGMAP" : ['darkgreen','lime','gold'],
                "APGMAP" : ['darkgreen','lime'],
            }
            color_dict_add_tests = {
                "nested" : ['black'], # 3 it
                "nested_skip0_3_my_settings" : [color_dict_after_MIC["nested_ADMMLim"][3]],
                "nested_skip1_3_my_settings" : [color_dict_after_MIC["nested_ADMMLim"][1]],
                "nested_skip2_3_my_settings" : [color_dict_after_MIC["nested_ADMMLim"][2]],
                "nested_ADMMLim_more_ADMMLim_it_10" : [color_dict_after_MIC["nested_ADMMLim"][0],color_dict_after_MIC["nested_ADMMLim"][1],color_dict_after_MIC["nested_ADMMLim"][2],color_dict_after_MIC["nested_ADMMLim"][3],color_dict_after_MIC["nested_ADMMLim"][4]],
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
                "nested_CT_0_skip_3it" : [color_dict_after_MIC["nested_CT_skip"][4]],
                "nested_CT_1_skip_3it" : [color_dict_after_MIC["nested_CT_skip"][5]],
                "nested_CT_0_skip_10it" : [color_dict_after_MIC["nested_CT_skip"][6]],
                "nested_CT_1_skip_10it" : [color_dict_after_MIC["nested_CT_skip"][7]],
                "nested_random_3_skip_10it" : [color_dict_after_MIC["nested_random_skip"][0]],
                "nested_random_2_skip_10it" : [color_dict_after_MIC["nested_random_skip"][1]],
                "nested_random_1_skip_10it" : [color_dict_after_MIC["nested_random_skip"][2]],
                "nested_random_0_skip_10it" : [color_dict_after_MIC["nested_random_skip"][3]],
                "nested_DD" : [color_dict_after_MIC["nested_random_skip"][4]],
                "DIPRecon_BSREM_stand" : [color_dict_after_MIC["DIPRecon"][0]],
                "DIPRecon_CT_3_skip" : [color_dict_after_MIC["DIPRecon"][1]],
                "DIPRecon_CT_2_skip" : [color_dict_after_MIC["DIPRecon"][2]],
                "DIPRecon_CT_1_skip" : [color_dict_after_MIC["DIPRecon"][3]],
                "DIPRecon_skip3_3_my_settings" : [color_dict_after_MIC["DIPRecon"][7]],
                "DIPRecon_random_3_skip" : [color_dict_after_MIC["DIPRecon"][4]],
                "DIPRecon_random_2_skip" : [color_dict_after_MIC["DIPRecon"][5]],
                "DIPRecon_random_1_skip" : [color_dict_after_MIC["DIPRecon"][6]],
            }

            color_dict_TMI_DNA = {
                "nested" : ['red','pink'],
                "DIPRecon" : ['cyan','blue','teal','blueviolet'],
                "APGMAP" : ['darkgreen','lime','gold'] + 5*['cyan','blue','teal','blueviolet'],
                "ADMMLim" : ['fuchsia'] + ['cyan','blue','teal','blueviolet'],
                "OSEM" : ['darkorange'] + ['cyan','blue','teal','blueviolet'],
                "BSREM" : ['grey'] + ['cyan','blue','teal','blueviolet']
            }

            color_dict = {**color_dict_after_MIC, **color_dict_add_tests, **color_dict_TMI_DNA} # Comparison between APPGML and ADMMLim in nested (varying subsets and iterations)

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
        elif(self.phantom == "image4_0" or self.phantom == "image400_0" or self.phantom == "image40_0"):
            marker_dict = {
                "APPGML_it" : [':'],
                "APPGML_subsets" : ['-'],
                "ADMMLim" : ['--'],
                "CT" : ['dashdot'],
                "random" : ['dashdot'],
                "intermediate" : ['-'],
                "APGMAP" : ['-','-'],
                "BSREM" : ['-'],
                "OSEM" : ['-'],
                "DIPRecon" : ['-']
            }
            marker_dict_supp = {
                "nested" : [marker_dict["ADMMLim"][0]], # 3 it
                "nested_skip0_3_my_settings" : [marker_dict["intermediate"][0]],
                "nested_skip1_3_my_settings" : [marker_dict["intermediate"][0]],
                "nested_skip2_3_my_settings" : [marker_dict["intermediate"][0]],
                "nested_ADMMLim_more_ADMMLim_it_10" : [marker_dict["intermediate"][0],marker_dict["intermediate"][0],marker_dict["intermediate"][0],marker_dict["intermediate"][0]],
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
                "nested_CT_2_skip_3it" : [marker_dict["CT"][0]],
                "nested_CT_3_skip_3it" : [marker_dict["CT"][0]],
                "nested_CT_2_skip_10it" : [marker_dict["CT"][0]],
                "nested_CT_3_skip_10it" : [marker_dict["CT"][0]],
                "nested_CT_0_skip_3it" : [marker_dict["CT"][0]],
                "nested_CT_1_skip_3it" : [marker_dict["CT"][0]],
                "nested_CT_0_skip_10it" : [marker_dict["CT"][0]],
                "nested_CT_1_skip_10it" : [marker_dict["CT"][0]],
                "nested_random_3_skip_10it" : [marker_dict["random"][0]],
                "nested_random_2_skip_10it" : [marker_dict["random"][0]],
                "nested_random_1_skip_10it" : [marker_dict["random"][0]],
                "nested_random_0_skip_10it" : [marker_dict["random"][0]],
                "nested_DD" : [marker_dict["random"][0]],
                "DIPRecon_BSREM_stand" : [marker_dict["DIPRecon"][0]],
                "DIPRecon_CT_3_skip" : [marker_dict["DIPRecon"][0]],
                "DIPRecon_CT_2_skip" : [marker_dict["DIPRecon"][0]],
                "DIPRecon_CT_1_skip" : [marker_dict["DIPRecon"][0]],
                "DIPRecon_skip3_3_my_settings" : [marker_dict["DIPRecon"][0]],
                "DIPRecon_random_3_skip" : [marker_dict["DIPRecon"][0]],
                "DIPRecon_random_2_skip" : [marker_dict["DIPRecon"][0]],
                "DIPRecon_random_1_skip" : [marker_dict["DIPRecon"][0]],
            }
            marker_dict = {**marker_dict, **marker_dict_supp}

        return marker_dict, color_dict


    def load_metrics(self, sorted_suffixes, idx_wanted, root, config, method, task, csv_before_MIC, quantitative_tradeoff, ROI):
        # Initialize metrics arrays
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
        likelihoods = []
        MA_cold_inside_recon = []
        MA_cold_edge_recon = []

        IR_final = []
        metrics_final = []

        for i in range(len(sorted_suffixes)):
            i_replicate = idx_wanted[i] # Loop over rhos and replicates, for each sorted rho, take sorted replicate
            suffix = sorted_suffixes[i].rstrip("\n")
            replicate = "replicate_" + str(i_replicate + 1)

            
            self.subroot = self.subroot_data + 'debug/'*self.debug + self.phantom + '/'+ str(replicate) + '/' + config[method]["method"] + '/' # Directory root
            self.suffix = suffix[:-12] # Remove NNEPPS from suffix
            self.max_iter = config[method]["max_iter"]
            self.defineTotalNbIter_beta_rho(method,config[method],task)
            
            metrics_file = root + '/data/Algo' + '/metrics/' + config[method]["image"] + '/' + str(replicate) + '/' + config[method]["method"] + '/' + suffix + '/' + 'metrics.csv'
            with open(metrics_file, 'r') as myfile:
                spamreader = reader_csv(myfile,delimiter=';')
                rows_csv = list(spamreader)
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
                    # rows_csv[13] = [float(rows_csv[13][i]) for i in range(int(self.i_init) - 1, min(len(rows_csv[13]),self.total_nb_iter))] # IR whole recon is useless
                    if (ROI == "whole"):
                        if (len(rows_csv) > 14):
                            rows_csv[14] = [float(rows_csv[14][i]) for i in range(int(self.i_init) - 1, min(len(rows_csv[14]),self.total_nb_iter))]
                        else:
                            raise ValueError("likelihood is not in csv")

                PSNR_recon.append(np.array(rows_csv[0]))
                PSNR_norm_recon.append(np.array(rows_csv[1]))
                MSE_recon.append(np.array(rows_csv[2]))
                SSIM_recon.append(np.array(rows_csv[3]))
                MA_cold_recon.append(np.array(np.array(rows_csv[4]) - 10) / 10 * 100) # relative bias
                # MA_cold_recon.append(np.array(np.array(rows_csv[4]) - 10) * 100) # bias
                AR_hot_recon.append(np.array(rows_csv[5]) / 400 * 100)

                if (not csv_before_MIC):
                    AR_hot_TEP_recon.append(np.array(rows_csv[6]) / 400 * 100)
                    AR_hot_TEP_match_square_recon.append(np.array(rows_csv[7]) / 400 * 100)
                    AR_hot_perfect_match_recon.append(np.array(rows_csv[8]) / 400 * 100)
                    AR_bkg_recon.append(np.array(rows_csv[9]))
                    IR_bkg_recon.append(np.array(rows_csv[10]))
                    # IR_whole_recon.append(np.array(rows_csv[13])) # IR whole recon is useless
                    if (ROI == "whole"):
                        likelihoods.append(np.array(rows_csv[14]))
                else:        
                    AR_bkg_recon.append(np.array(rows_csv[6]))
                    IR_bkg_recon.append(np.array(rows_csv[7]))
            
            if ("cold_" in ROI):
                metrics_file = root + '/data/Algo' + '/metrics/' + config[method]["image"] + '/' + str(replicate) + '/' + config[method]["method"] + '/' + suffix + '/' + 'metrics_cold.csv'
                with open(metrics_file, 'r') as myfile:
                    spamreader = reader_csv(myfile,delimiter=';')
                    rows_csv = list(spamreader)
                    rows_csv[0] = [float(rows_csv[0][i]) for i in range(int(self.i_init) - 1,min(len(rows_csv[0]),self.total_nb_iter))]
                    rows_csv[1] = [float(rows_csv[1][i]) for i in range(int(self.i_init) - 1, min(len(rows_csv[1]),self.total_nb_iter))]
                    rows_csv[2] = [float(rows_csv[2][i]) for i in range(int(self.i_init) - 1, min(len(rows_csv[2]),self.total_nb_iter))]
                    MA_cold_recon.append(np.array(np.array(rows_csv[0]) - 10) / 10 * 100) # relative bias
                    MA_cold_inside_recon.append(np.array(np.array(rows_csv[1]) - 10) / 10 * 100) # relative bias
                    MA_cold_edge_recon.append(np.array(np.array(rows_csv[2]) - 10) / 10 * 100) # relative bias
                    # MA_cold_recon.append(np.array(np.array(rows_csv[0]) - 10) * 100) # bias
                    # MA_cold_inside_recon.append(np.array(np.array(rows_csv[1]) - 10) * 100) # bias
                    # MA_cold_edge_recon.append(np.array(np.array(rows_csv[2]) - 10) * 100) # bias
                # print("No such file : " + metrics_file)

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
            elif ROI == 'cold_inside':
                #metrics = [abs(cold) for cold in MA_cold_recon] # Take absolute value of MA cold for tradeoff curves
                metrics = MA_cold_inside_recon
            elif ROI == 'cold_edge':
                #metrics = [abs(cold) for cold in MA_cold_recon] # Take absolute value of MA cold for tradeoff curves
                metrics = MA_cold_edge_recon
        else:
            if (ROI == "whole"):
                metrics = likelihoods
            else:
                metrics = SSIM_recon
        # Keep useful information to plot from metrics                
        IR_final = IR_bkg_recon
        metrics_final = metrics

        return metrics_final, IR_final
