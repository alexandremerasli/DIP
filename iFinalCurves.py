## Python libraries
# Math
from turtle import goto
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Useful
from csv import reader as reader_csv
import re

# Local files to import
from vGeneral import vGeneral

class iFinalCurves(vGeneral):
    def __init__(self,config, *args, **kwargs):
        print("__init__")

    def initializeSpecific(self,config,root, *args, **kwargs):
        print("init")
        # show the plots in python or not !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def runComputation(self,config,root):

        plot_all_replicates_curves = True
        if plot_all_replicates_curves:
            color_avg = 'black'
        else:
            color_avg = None

        for ROI in ['hot','cold']:
            plt.figure()

            fig, ax = [None] * 2, [None] * 2
            for fig_nb in range(2):
                fig[fig_nb], ax[fig_nb] = plt.subplots()

            replicates_legend = []
            method_list = config["method"]

            for method in method_list: # Loop over methods
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
                
                # Retrieve number of rhos and replicates
                nb_rho = len(config["rho"])
                nb_replicates = int(len(replicates[0]) / nb_rho)

                # Wanted list of replicates
                idx_wanted = []
                for i in range(nb_rho):
                    idx_wanted += range(0,nb_replicates)

                # Check replicates from results are compatible with this script
                replicate_idx = [int(re.findall(r'(\w+?)(\d+)', replicates[0][idx].rstrip())[0][-1]) for idx in range(len(replicates[0]))]
                if list(np.sort(replicate_idx).astype(int)-1) != list(np.sort(idx_wanted)):
                    print(np.sort(idx_wanted))
                    print(np.sort(replicate_idx).astype(int)-1)
                    raise ValueError("Replicates are not the same for each case !")

                # Sort suffixes from file by rho values 
                sorted_suffixes = list(suffixes[0])
                sorted_suffixes.sort(key=self.natural_keys)

                # Load metrics from last runs to merge them in one figure
                for i in range(len(sorted_suffixes)):
                    i_replicate = idx_wanted[i] # Loop over rhos and replicates, for each sorted rho, take sorted replicate
                    suffix = sorted_suffixes[i].rstrip("\n")
                    replicate = "replicate_" + str(i_replicate + 1)
                    metrics_file = root + '/data/Algo' + '/metrics/' + config["image"] + '/' + str(replicate) + '/' + method + '/' + suffix + '/' + 'metrics.csv'
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

                # Compute number of displayable iterations for each rho
                len_mini_list = np.zeros((nb_rho,nb_replicates),dtype=int)
                len_mini = np.zeros((nb_rho),dtype=int)
                for rho_idx in range(nb_rho):
                    for replicate in range(nb_replicates):
                        len_mini_list[rho_idx][replicate] = len(metrics_final[replicate + nb_replicates*rho_idx])
                    len_mini[rho_idx] = int(np.min(len_mini_list[rho_idx]))

                # Plot 2 figures for each ROI : tradeoff curve (metric VS IR) and bias with iterations
                for fig_nb in range(2):
                    for rho_idx in range(nb_rho):
                        for replicate in range(nb_replicates):
                            case = replicate + nb_replicates*rho_idx
                            if (fig_nb == 0): # Plot tradeoff curves
                                idx_sort = np.argsort(IR_final[case])
                                if (plot_all_replicates_curves):
                                    if (method == "nested" or method == "Gong"):
                                        ax[fig_nb].plot(100*IR_final[case][0],metrics_final[case][0],'o', mfc='none',color='black',label='_nolegend_') # IR in %
                                    else:
                                        ax[fig_nb].plot(100*IR_final[case][idx_sort],metrics_final[case][idx_sort],'-o',label='_nolegend_') # IR in %                     
                        
                            if (fig_nb == 1): # Plot bias curves
                                if (plot_all_replicates_curves):
                                    ax[fig_nb].plot(np.arange(0,len(metrics_final[case])),metrics_final[case],label='_nolegend_') # Plot bias curves with iterations for each replicate

                        # Compute average of tradeoff and bias curves with iterations
                        avg_metrics = np.sum(np.array(metrics_final)[nb_replicates*rho_idx:nb_replicates*(rho_idx+1)][:len_mini[rho_idx]],axis=0) / nb_replicates
                        avg_IR = np.sum(np.array(IR_final)[nb_replicates*rho_idx:nb_replicates*(rho_idx+1)][:len_mini[rho_idx]],axis=0) / nb_replicates
                        # Compute std bias curves with iterations
                        std_metrics = np.sqrt(np.sum((np.array(metrics_final)[nb_replicates*rho_idx:nb_replicates*(rho_idx+1),:] - avg_metrics[:])**2,axis=0) / nb_replicates)
                        std_IR = np.sqrt(np.sum((np.array(IR_final)[nb_replicates*rho_idx:nb_replicates*(rho_idx+1),:] - avg_IR[:])**2,axis=0) / nb_replicates)

                        if (fig_nb == 0):
                            ax[fig_nb].plot(100*avg_IR,avg_metrics,'-o',color=color_avg)
                            ax[fig_nb].fill(np.concatenate((100*(avg_IR-std_IR),100*(avg_IR[::-1]+std_IR[::-1]))),np.concatenate((avg_metrics-std_metrics,avg_metrics[::-1]+std_metrics[::-1])), alpha = 0.4, label='_nolegend_')
                            ax[fig_nb].set_xlabel('Image Roughness in the background (%)', fontsize = 18)
                            ax[fig_nb].set_ylabel('Absolute bias (AU)', fontsize = 18)
                        if (fig_nb == 1):
                            # Plot average and std of bias curves with iterations
                            ax[fig_nb].plot(np.arange(0,len(avg_metrics)),avg_metrics,color=color_avg)
                            # Plot dashed line for target value, according to ROI
                            if ROI == 'hot':
                                ax[fig_nb].hlines(400,xmin=0,xmax=len(avg_metrics)-1,color='grey',linestyle='dashed',label='_nolegend_')
                            else:
                                ax[fig_nb].hlines(10,xmin=0,xmax=len(avg_metrics)-1,color='grey',linestyle='dashed',label='_nolegend_')
                            ax[fig_nb].fill_between(np.arange(0,len(avg_metrics)), avg_metrics - std_metrics, avg_metrics + std_metrics, alpha = 0.4, label='_nolegend_')
                            ax[fig_nb].set_xlabel('Iterations', fontsize = 18)
                            ax[fig_nb].set_ylabel('Bias (AU)', fontsize = 18)
                            ax[fig_nb].set_title(method + " reconstruction averaged on " + str(nb_replicates) + " replicates")

                        replicates_legend.append(method + " : rho = " + str(config["rho"][rho_idx]))
                    
                    ax[fig_nb].legend(replicates_legend)

            # Saving figures locally in png
            for fig_nb in range(2):
                rho = config["rho"]
                if (fig_nb == 0):
                    if ROI == 'hot':
                        title = method + " rho = " + str(rho) + 'AR in ' + ROI + ' region vs IR in background' + '.png'    
                    elif ROI == 'cold':
                        title = method + " rho = " + str(rho) + 'AR in ' + ROI + ' region vs IR in background' + '.png'    
                elif (fig_nb == 1):
                    if ROI == 'hot':
                        title = method + " rho = " + str(rho) + 'AR in ' + ROI + ' region for ' + str(nb_replicates) + ' replicates' + '.png'
                    elif ROI == 'cold':
                        title = method + " rho = " + str(rho) + 'AR in ' + ROI + ' region for ' + str(nb_replicates) + ' replicates' + '.png'
                fig[fig_nb].savefig(self.subroot_data + 'metrics' + '/' + title)