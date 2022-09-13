## Python libraries
# Math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Useful
from csv import reader as reader_csv
import re

# Local files to import
from vGeneral import vGeneral

class iFinalCurves(vGeneral):
    def __init__(self,config):
        print("__init__")

    def initializeSpecific(self,settings_config,fixed_config,hyperparameters_config,root):
        print("init")
        # show the plots in python or not !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def runComputation(self,config,settings_config,fixed_config,hyperparameters_config,root):
        for ROI in ['hot','cold']:
            plt.figure()

            suffixes_legend = []
            replicates_legend = []

            #'''
            if (self.debug or self.ray == False):
                method_list = [config["method"]]
            else:
                method_list = config["method"]['grid_search']
            #'''


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

                # Sort replicates from file
                replicate_idx = [replicates[0][idx].rstrip() for idx in range(len(replicates[0]))]
                idx_replicates_sort = np.argsort(replicate_idx)
                # Sort suffixes from file by rho values 
                sorted_suffixes = list(suffixes[0])
                sorted_suffixes.sort(key=self.natural_keys)

                # Load metrics from last runs to merge them in one figure
                for idx in idx_replicates_sort: # Loop over rhos and replicates, for each sorted rho, take sorted replicate
                    suffix = sorted_suffixes[idx].rstrip("\n")
                    replicate = replicates[0][idx].rstrip()
                    if (self.debug or self.ray == False):
                        metrics_file = root + '/data/Algo' + '/metrics/' + config["image"] + '/' + str(replicate) + '/' + method + '/' + suffix + '/' + 'metrics.csv'
                    else:
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
                if (self.debug or self.ray == False):
                    nb_rho = 1
                else:
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
                        avg = np.zeros((len_mini[rho_idx],),dtype=np.float32)
                        std = np.zeros((len_mini[rho_idx],),dtype=np.float32)
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
                        if (self.debug or self.ray == False):
                            rho = config["rho"]
                        else:
                            rho = config["rho"]['grid_search'][0]
                        if ROI == 'hot':
                            title = method + " rho = " + str() + 'AR in ' + ROI + ' region vs IR in background' + '.png'    
                        elif ROI == 'cold':
                            title = method + " rho = " + str(rho) + 'AR in ' + ROI + ' region vs IR in background' + '.png'    
                    elif (fig_nb == 1):
                        if ROI == 'hot':
                            title = method + " rho = " + str(rho) + 'AR in ' + ROI + ' region for ' + str(nb_replicates) + ' replicates' + '.png'
                        elif ROI == 'cold':
                            title = method + " rho = " + str(rho) + 'AR in ' + ROI + ' region for ' + str(nb_replicates) + ' replicates' + '.png'
                    fig.savefig(self.subroot_data + 'metrics' + '/' + title)
                    from textwrap import wrap
                    wrapped_title = "\n".join(wrap(suffix, 50))
                    #ax.set_title(wrapped_title,fontsize=12)