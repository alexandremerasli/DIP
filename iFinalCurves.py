## Python libraries
# Math
from audioop import avg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Useful
from csv import reader as reader_csv
import re

# Local files to import
from vGeneral import vGeneral
from iResults import iResults

class iFinalCurves(vGeneral):
    def __init__(self,config, *args, **kwargs):
        print("__init__")

    def initializeSpecific(self,config,root, *args, **kwargs):
        print("init")
        # show the plots in python or not !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def runComputation(self,config,root):



        # Initialize general variables
        self.initializeGeneralVariables(config,root)
        #vDenoising.initializeSpecific(self,config,root)

        if ('ADMMLim' in config["method"]):
            self.i_init = 20 # Remove first iterations
            #self.i_init = 1 # Remove first iterations
        else:
            self.i_init = 1

        if ('ADMMLim' in config["method"]):
            try:
                self.path_stopping_criterion = self.subroot + self.suffix + '/' + format(0) + '_adaptive_stopping_criteria.log'
                with open(self.path_stopping_criterion) as f:
                    first_line = f.readline() # Read first line to get second one
                    self.total_nb_iter = min(int(f.readline().rstrip()) - self.i_init, config["nb_outer_iteration"] - self.i_init + 1)
                    #self.total_nb_iter = config["nb_outer_iteration"] - self.i_init + 1 # Override value
            except:
                self.total_nb_iter = config["nb_outer_iteration"] - self.i_init + 1
            self.beta = config["alpha"]
        elif (config["method"] == 'nested' or config["method"] == 'Gong'):
            if ('post_reco' in config["task"]):
                self.total_nb_iter = config["sub_iter_DIP"]
            else:
                self.total_nb_iter = config["max_iter"]
        else:
            self.total_nb_iter = self.max_iter

            if (config["method"] == 'AML'):
                self.beta = config["A_AML"]
            if (config["method"] == 'BSREM' or config["method"] == 'nested' or config["method"] == 'Gong' or config["method"] == 'APGMAP'):
                self.rho = config["rho"]
                self.beta = self.rho



        # Remove when CRC are computed in iResults.py
        self.PETImage_shape_str = self.read_input_dim(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.hdr')
        self.PETImage_shape = self.input_dim_str_to_list(self.PETImage_shape_str)
        self.image_gt = self.fijii_np(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.raw',shape=(self.PETImage_shape),type='<f')
        if config["FLTNB"] == "double":
            self.image_gt = self.image_gt.astype(np.float64)
        self.phantom_ROI = self.get_phantom_ROI(self.phantom)
        image_gt_cropped = self.image_gt * self.phantom_ROI
        C_bkg = np.mean(image_gt_cropped)

        CRC_plot = False
        plot_all_replicates_curves = False
        if plot_all_replicates_curves:
            color_avg = 'black'
        else:
            color_avg = None

        for ROI in ['hot','cold']:
            # Initialize 3 figures
            fig, ax = [None] * 3, [None] * 3
            for fig_nb in range(3):
                fig[fig_nb], ax[fig_nb] = plt.subplots()

            replicates_legend = [None] * 3
            replicates_legend = [[],[],[]]

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
                CRC_cold_recon = []
                CRC_hot_recon = []
                AR_bkg_recon = []
                IR_bkg_recon = []

                IR_final = []
                metrics_final = []
                
                with open(root + '/data/Algo' + '/suffixes_for_last_run_' + method + '.txt') as f:
                    suffixes.append(f.readlines())
                with open(root + '/data/Algo' + '/replicates_for_last_run_' + method + '.txt') as f:
                    replicates.append(f.readlines())
                
                # Retrieve number of rhos and replicates and other dimension
                rho_name = "rho"
                nb_rho = len(config["rho"])
                if (method == "APGMAP" or method == "AML"):
                    config_other_dim = config["A_AML"]
                    other_dim_name = "A"
                elif (method == "MLEM" or method == "OSEM"):
                    config_other_dim = config["post_smoothing"]
                    rho_name = "smoothing"
                    other_dim_name = ""
                elif (method == "nested" or method == "Gong"):
                    config_other_dim = config["lr"]
                    other_dim_name = "lr"
                else:
                    config_other_dim = [""]
                    other_dim_name = ""
                nb_other_dim = len(config_other_dim)
                nb_replicates = int(len(replicates[0]) / (nb_rho * nb_other_dim))

                # Sort rho and other dim like suffix
                config["rho"] = sorted(config["rho"])
                config_other_dim = sorted(config_other_dim)

                # Wanted list of replicates
                idx_wanted = []
                for i in range(nb_rho):
                    for p in range(nb_other_dim):
                        idx_wanted += range(0,nb_replicates)

                # Check replicates from results are compatible with this script
                replicate_idx = [int(re.findall(r'(\w+?)(\d+)', replicates[0][idx].rstrip())[0][-1]) for idx in range(len(replicates[0]))]
                if list(np.sort(replicate_idx).astype(int)-1) != list(np.sort(idx_wanted)):
                    print(np.sort(idx_wanted))
                    print(np.sort(replicate_idx).astype(int)-1)
                    raise ValueError("Replicates are not the same for each case !")

                # Sort suffixes from file by rho and other dim values 
                sorted_suffixes = list(suffixes[0])
                if (method != "ADMMLim" and method != "nested"):
                    sorted_suffixes.sort(key=self.natural_keys)
                else:
                    sorted_suffixes.sort(key=self.natural_keys_ADMMLim)
                
                for i in range(len(sorted_suffixes)):
                    print(sorted_suffixes[i])

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
                            
                            rows_csv[0] = [float(rows_csv[0][i]) for i in range(self.i_init - 1,min(len(rows_csv[0]),self.total_nb_iter))]
                            rows_csv[1] = [float(rows_csv[1][i]) for i in range(self.i_init - 1, min(len(rows_csv[1]),self.total_nb_iter))]
                            rows_csv[2] = [float(rows_csv[2][i]) for i in range(self.i_init - 1, min(len(rows_csv[2]),self.total_nb_iter))]
                            rows_csv[3] = [float(rows_csv[3][i]) for i in range(self.i_init - 1, min(len(rows_csv[3]),self.total_nb_iter))]
                            rows_csv[4] = [float(rows_csv[4][i]) for i in range(self.i_init - 1, min(len(rows_csv[4]),self.total_nb_iter))]
                            rows_csv[5] = [float(rows_csv[5][i]) for i in range(self.i_init - 1, min(len(rows_csv[5]),self.total_nb_iter))]
                            rows_csv[6] = [float(rows_csv[6][i]) for i in range(self.i_init - 1, min(len(rows_csv[6]),self.total_nb_iter))]
                            rows_csv[7] = [float(rows_csv[7][i]) for i in range(self.i_init - 1, min(len(rows_csv[7]),self.total_nb_iter))]

                            PSNR_recon.append(np.array(rows_csv[0]))
                            PSNR_norm_recon.append(np.array(rows_csv[1]))
                            MSE_recon.append(np.array(rows_csv[2]))
                            SSIM_recon.append(np.array(rows_csv[3]))
                            #MA_cold_recon.append(np.array(rows_csv[4]))
                            MA_cold_recon.append(np.array(rows_csv[4]) / 10 * 100)
                            #AR_hot_recon.append(np.array(rows_csv[5]))
                            AR_hot_recon.append(np.array(rows_csv[5]) / 400 * 100)
                            AR_bkg_recon.append(np.array(rows_csv[6]))
                            IR_bkg_recon.append(np.array(rows_csv[7]))
                            
                            try:
                                MA_cold = np.array(rows_csv[8])
                            except:
                                MA_cold = np.array(rows_csv[6])
                            CRC_cold = 100*((MA_cold.astype(float)/C_bkg - 1) / (10/C_bkg - 1))
                            try:
                                AR_hot = np.array(rows_csv[9])
                            except:
                                AR_hot = np.array(rows_csv[7])
                            CRC_hot = 100*((AR_hot.astype(float)/C_bkg - 1) / (400/C_bkg - 1))
                            CRC_cold_recon.append(CRC_cold)
                            CRC_hot_recon.append(CRC_hot)

                    except:
                        print("No such file : " + metrics_file)

                
                if (method == "MLEM" or method == "OSEM"):
                    nb_rho, nb_other_dim = nb_other_dim, nb_rho
                    config["rho"], config_other_dim = config_other_dim, config["rho"]


                # Select metrics to plot according to ROI
                if ROI == 'hot':
                    if CRC_plot:
                        metrics = CRC_hot_recon
                    else:
                        #metrics = [abs(hot) for hot in AR_hot_recon] # Take absolute value of AR hot for tradeoff curves
                        metrics = AR_hot_recon # Take absolute value of AR hot for tradeoff curves
                else:
                    if CRC_plot:
                        metrics = CRC_cold_recon
                    else:
                        #metrics = [abs(cold) for cold in MA_cold_recon] # Take absolute value of MA cold for tradeoff curves
                        metrics = MA_cold_recon # Take absolute value of MA cold for tradeoff curves

                # Keep useful information to plot from metrics                
                IR_final = IR_bkg_recon
                metrics_final = metrics

                # Compute number of displayable iterations for each rho and find case with smallest iterations (useful for ADMMLim)
                len_mini_list = np.zeros((nb_rho,nb_other_dim,nb_replicates),dtype=int)
                len_mini = np.zeros((nb_rho),dtype=int)
                case_mini = np.zeros((nb_rho),dtype=int)
                for rho_idx in range(nb_rho):
                    for other_dim_idx in range(nb_other_dim):
                        for replicate_idx in range(nb_replicates):
                            len_mini_list[rho_idx,other_dim_idx,replicate_idx] = len(metrics_final[replicate_idx + nb_replicates*other_dim_idx + (nb_replicates*nb_other_dim)*rho_idx])
                        len_mini[rho_idx] = int(np.min(len_mini_list[rho_idx]))
                        case_mini[rho_idx] = int(np.argmin(len_mini_list[rho_idx,:,:])) + nb_replicates*rho_idx

                # Create numpy array with same number of iterations for each case
                IR_final_array = []
                metrics_final_array = []
                for rho_idx in range(nb_rho):
                    for other_dim_idx in range(nb_other_dim):
                        IR_final_array.append(np.zeros((nb_replicates,len_mini[rho_idx])))
                        metrics_final_array.append(np.zeros((nb_replicates,len_mini[rho_idx])))
                        for common_it in range(len_mini[rho_idx]):
                            for replicate_idx in range(nb_replicates):
                                IR_final_array[other_dim_idx+nb_other_dim*rho_idx][replicate_idx,common_it] = IR_final[replicate_idx + nb_replicates*other_dim_idx + (nb_replicates*nb_other_dim)*rho_idx][common_it]
                                metrics_final_array[other_dim_idx+nb_other_dim*rho_idx][replicate_idx,common_it] = metrics_final[replicate_idx + nb_replicates*other_dim_idx + (nb_replicates*nb_other_dim)*rho_idx][common_it]             

                
                IR_final_final_array = np.zeros((nb_rho,nb_other_dim,nb_replicates,np.max(len_mini)))
                metrics_final_final_array = np.zeros((nb_rho,nb_other_dim,nb_replicates,np.max(len_mini)))
                for rho_idx in range(nb_rho):
                    for other_dim_idx in range(nb_other_dim):
                        for common_it in range(len_mini[rho_idx]):
                            for replicate_idx in range(nb_replicates):
                                IR_final_final_array[rho_idx,other_dim_idx,replicate_idx,common_it] = IR_final_array[other_dim_idx+nb_other_dim*rho_idx][replicate_idx,common_it]
                                metrics_final_final_array[rho_idx,other_dim_idx,replicate_idx,common_it] = metrics_final_array[other_dim_idx+nb_other_dim*rho_idx][replicate_idx,common_it]


                # Plot 3 figures for each ROI : tradeoff curve with iteration (metric VS IR), bias with iterations, and tradeoff curve at convergence
                reg = [None] * 3
                for fig_nb in range(3):
                    if (fig_nb == 0):
                        reg[fig_nb] = np.zeros((nb_rho*nb_other_dim,np.max(len_mini)))
                    elif (fig_nb == 2):
                        #reg[fig_nb] = np.zeros((nb_rho*nb_other_dim,np.max(len_mini)))
                        reg[fig_nb] = np.zeros((nb_rho*nb_other_dim))
                    for rho_idx in range(nb_rho):
                        for other_dim_idx in range(nb_other_dim):
                            for replicate_idx in range(nb_replicates):
                                case = replicate_idx + nb_replicates*other_dim_idx + (nb_replicates*nb_other_dim)*rho_idx
                                if (fig_nb == 0): # Plot tradeoff curves with iterations
                                    it = np.arange(len(IR_final[case_mini[rho_idx]]))
                                    if (plot_all_replicates_curves):
                                        ax[fig_nb].plot(100*IR_final_final_array[rho_idx,other_dim_idx,replicate_idx,it],metrics_final_final_array[rho_idx,other_dim_idx,replicate_idx,it],label='_nolegend_') # IR in %
                            
                            #'''
                            if (fig_nb == 0):
                                for it in range(len(IR_final[case_mini[rho_idx]])):
                                    reg[fig_nb][other_dim_idx+nb_other_dim*rho_idx,:len_mini[rho_idx]] = self.linear_regression(100*IR_final_array[other_dim_idx+nb_other_dim*rho_idx][:,it],metrics_final_array[other_dim_idx+nb_other_dim*rho_idx][:,it])
                            #'''
                            for replicate_idx in range(nb_replicates):
                                if (fig_nb == 1): # Plot bias curves
                                    if (plot_all_replicates_curves):
                                        ax[fig_nb].plot(np.arange(0,len_mini[rho_idx]),metrics_final_final_array[rho_idx,other_dim_idx,replicate_idx,:len_mini[rho_idx]],label='_nolegend_') # IR in %

                    avg_metrics = np.zeros((nb_rho*nb_other_dim,np.max(len_mini)))
                    avg_IR = np.zeros((nb_rho*nb_other_dim,np.max(len_mini)))
                    std_metrics = np.zeros((nb_rho*nb_other_dim,np.max(len_mini)))
                    std_IR = np.zeros((nb_rho*nb_other_dim,np.max(len_mini)))


                    #'''
                    for replicate_idx in range(nb_replicates):
                        for other_dim_idx in range(nb_other_dim):
                            if (fig_nb == 2): # Plot tradeoff curves at convergence
                                if (plot_all_replicates_curves):
                                    ax[fig_nb].plot(100*IR_final_final_array[:,other_dim_idx,replicate_idx,:][(np.arange(nb_rho),len_mini-1)],metrics_final_final_array[:,other_dim_idx,replicate_idx,:][(np.arange(nb_rho),len_mini-1)],label='_nolegend_') # IR in %
                    #'''

                    #'''
                    if (fig_nb == 2): # Plot tradeoff curves at convergence
                        for rho_idx in range(nb_rho):
                            for other_dim_idx in range(nb_other_dim):
                                #reg[fig_nb].append(self.linear_regression(100*IR_final_array[other_dim_idx+nb_other_dim*rho_idx][:,-1],metrics_final_array[other_dim_idx+nb_other_dim*rho_idx][:,-1]))
                                reg[fig_nb][other_dim_idx+nb_other_dim*rho_idx] = self.linear_regression(100*IR_final_array[other_dim_idx+nb_other_dim*rho_idx][:,-1],metrics_final_array[other_dim_idx+nb_other_dim*rho_idx][:,-1])
                    #'''
                    #'''

                    
                    for rho_idx in range(nb_rho):
                        for other_dim_idx in range(nb_other_dim):
                            # Compute average of tradeoff and bias curves with iterations
                            avg_metrics[other_dim_idx+nb_other_dim*rho_idx,:len_mini[rho_idx]] = np.sum(metrics_final_final_array[rho_idx,other_dim_idx,:,:len_mini[rho_idx]],axis=0) / nb_replicates
                            avg_IR[other_dim_idx+nb_other_dim*rho_idx,:len_mini[rho_idx]] = np.sum(IR_final_final_array[rho_idx,other_dim_idx,:,:len_mini[rho_idx]],axis=0) / nb_replicates
                            # Compute std bias curves with iterations
                            std_metrics[other_dim_idx+nb_other_dim*rho_idx,:len_mini[rho_idx]] = np.sqrt(np.sum((metrics_final_final_array[rho_idx,other_dim_idx,:,:len_mini[rho_idx]] - np.array(avg_metrics)[other_dim_idx+nb_other_dim*rho_idx,:len_mini[rho_idx]])**2,axis=0) / nb_replicates)
                            std_IR[other_dim_idx+nb_other_dim*rho_idx,:len_mini[rho_idx]] = np.sqrt(np.sum((IR_final_final_array[rho_idx,other_dim_idx,:,:len_mini[rho_idx]]- np.array(avg_IR)[other_dim_idx+nb_other_dim*rho_idx,:len_mini[rho_idx]])**2,axis=0) / nb_replicates)

                            if (fig_nb == 0):
                                ax[fig_nb].plot(100*avg_IR[other_dim_idx+nb_other_dim*rho_idx,:len_mini[rho_idx]],avg_metrics[other_dim_idx+nb_other_dim*rho_idx,:len_mini[rho_idx]],'-o',color=color_avg)
                                ax[fig_nb].fill(np.concatenate((100*(avg_IR[other_dim_idx+nb_other_dim*rho_idx,:len_mini[rho_idx]] - np.sign(reg[fig_nb])[other_dim_idx+nb_other_dim*rho_idx,:len_mini[rho_idx]]*std_IR[other_dim_idx+nb_other_dim*rho_idx,:len_mini[rho_idx]]),100*(avg_IR[other_dim_idx+nb_other_dim*rho_idx,:len_mini[rho_idx]][::-1] + np.sign(reg[fig_nb][other_dim_idx+nb_other_dim*rho_idx,:len_mini[rho_idx]][::-1])*std_IR[other_dim_idx+nb_other_dim*rho_idx,:len_mini[rho_idx]][::-1]))),np.concatenate((avg_metrics[other_dim_idx+nb_other_dim*rho_idx,:len_mini[rho_idx]]-std_metrics[other_dim_idx+nb_other_dim*rho_idx,:len_mini[rho_idx]],avg_metrics[other_dim_idx+nb_other_dim*rho_idx,:len_mini[rho_idx]][::-1]+std_metrics[other_dim_idx+nb_other_dim*rho_idx,:len_mini[rho_idx]][::-1])), alpha = 0.4, label='_nolegend_')
                                ax[fig_nb].set_xlabel('Image Roughness (IR) in the background (%)', fontsize = 18)
                                #ax[fig_nb].set_ylabel(('Activity Recovery (AR) (%) '*(CRC_plot==False) + 'Contrast Recovery (CRC) (%) '*CRC_plot), fontsize = 18)
                                ax[fig_nb].set_ylabel(('Bias '*(CRC_plot==False) + 'Contrast Recovery (CRC) (%) '*CRC_plot), fontsize = 18)
                                ax[fig_nb].set_title(('AR '*(CRC_plot==False) + 'CRC '*CRC_plot) + 'in ' + ROI + ' region vs IR in background (with iterations)')
                            #'''
                            if (fig_nb == 1):
                                # Plot average and std of bias curves with iterations
                                ax[fig_nb].plot(np.arange(0,len_mini[rho_idx]),avg_metrics[other_dim_idx+nb_other_dim*rho_idx,:len_mini[rho_idx]],color=color_avg)
                                # Plot dashed line for target value, according to ROI
                                if ROI == 'hot':
                                    ax[fig_nb].hlines(100*(CRC_plot==False)+100*CRC_plot,xmin=0,xmax=len(avg_metrics[other_dim_idx+nb_other_dim*rho_idx])-1,color='grey',linestyle='dashed',label='_nolegend_')
                                else:
                                    ax[fig_nb].hlines(100*(CRC_plot==False)+100*CRC_plot,xmin=0,xmax=len(avg_metrics[other_dim_idx+nb_other_dim*rho_idx])-1,color='grey',linestyle='dashed',label='_nolegend_')
                                ax[fig_nb].fill_between(np.arange(0,len(avg_metrics[other_dim_idx+nb_other_dim*rho_idx])), avg_metrics[other_dim_idx+nb_other_dim*rho_idx] - std_metrics[other_dim_idx+nb_other_dim*rho_idx], avg_metrics[other_dim_idx+nb_other_dim*rho_idx] + std_metrics[other_dim_idx+nb_other_dim*rho_idx], alpha = 0.4, label='_nolegend_')
                                ax[fig_nb].set_xlabel('Iterations', fontsize = 18)
                                #ax[fig_nb].set_ylabel(('Activity Recovery (AR) (%) '*(CRC_plot==False) + 'Contrast Recovery (CRC) (%) '*CRC_plot), fontsize = 18)
                                ax[fig_nb].set_ylabel(('Bias '*(CRC_plot==False) + 'Contrast Recovery (CRC) (%) '*CRC_plot), fontsize = 18)
                                ax[fig_nb].set_title(method + " reconstruction averaged on " + str(nb_replicates) + " replicates (" + ROI + " ROI)")
                            #'''
                            if (fig_nb != 2):
                                replicates_legend[fig_nb].append(method + " : " + rho_name + " = " + str(config["rho"][rho_idx]) + (", " + other_dim_name + " = " + str(config_other_dim[other_dim_idx]))*(other_dim_name!=""))
                        
                    #'''
                    if (fig_nb == 2):
                        for other_dim_idx in range(nb_other_dim):
                            cases = np.arange(0,nb_other_dim*nb_rho,nb_other_dim) + other_dim_idx
                            ax[fig_nb].plot(100*avg_IR[(cases,len_mini-1)],avg_metrics[(cases,len_mini-1)],'-o',color=color_avg)
                            ax[fig_nb].fill(np.concatenate((100*(avg_IR[(cases,len_mini-1)] - np.sign(reg[fig_nb][cases])*std_IR[cases,-1]),100*(avg_IR[(cases,len_mini-1)][::-1] + np.sign(reg[fig_nb][cases][::-1])*std_IR[(cases,len_mini-1)][::-1]))),np.concatenate((avg_metrics[(cases,len_mini-1)]-std_metrics[(cases,len_mini-1)],avg_metrics[(cases,len_mini-1)][::-1]+std_metrics[(cases,len_mini-1)][::-1])), alpha = 0.4, label='_nolegend_')
                            replicates_legend[fig_nb].append(method + (": " + other_dim_name + " = " + str(config_other_dim[other_dim_idx]))*(other_dim_name!=""))
                        ax[fig_nb].set_xlabel('Image Roughness (IR) in the background (%)', fontsize = 18)
                        #ax[fig_nb].set_ylabel(('Activity Recovery (AR) (%) '*(CRC_plot==False) + 'Contrast Recovery (CRC) (%) '*CRC_plot), fontsize = 18)
                        ax[fig_nb].set_ylabel(('Bias '*(CRC_plot==False) + 'Contrast Recovery (CRC) (%) '*CRC_plot), fontsize = 18)
                        ax[fig_nb].set_title(('AR '*(CRC_plot==False) + 'CRC '*CRC_plot) + 'in ' + ROI + ' region vs IR in background (at convergence)')
                    #'''

                    if (method == method_list[-1]):
                        ax[fig_nb].legend(replicates_legend[fig_nb])

            # Saving figures locally in png
            for fig_nb in range(3):
                method = str(config["method"])
                rho = config["rho"]
                if (fig_nb == 0):
                    if ROI == 'hot':
                        title = method + " : " + rho_name + " = " + str(rho) + (", " + other_dim_name + " = " + str(config_other_dim))*(other_dim_name!="") + (' : AR'*(CRC_plot==False) + 'CRC '*CRC_plot) + ' in ' + ROI + ' region vs IR in background (with iterations)' + '.png'
                    elif ROI == 'cold':
                        title = method + " : " + rho_name + " = " + str(rho) + (", " + other_dim_name + " = " + str(config_other_dim))*(other_dim_name!="") + (' : AR'*(CRC_plot==False) + 'CRC '*CRC_plot) + ' in ' + ROI + ' region vs IR in background (with iterations)' + '.png'
                elif (fig_nb == 1):
                    if ROI == 'hot':
                        title = method + " : " + rho_name + " = " + str(rho) + (", " + other_dim_name + " = " + str(config_other_dim))*(other_dim_name!="") + (' : AR'*(CRC_plot==False) + 'CRC '*CRC_plot) + ' in ' + ROI + ' region for ' + str(nb_replicates) + ' replicates' + '.png'
                    elif ROI == 'cold':
                        title = method + " : " + rho_name + " = " + str(rho) + (", " + other_dim_name + " = " + str(config_other_dim))*(other_dim_name!="") + (' : AR'*(CRC_plot==False) + 'CRC '*CRC_plot) + ' in ' + ROI + ' region for ' + str(nb_replicates) + ' replicates' + '.png'
                elif (fig_nb == 2):
                    if ROI == 'hot':
                        title = method + " : " + rho_name + " = " + str(rho) + (", " + other_dim_name + " = " + str(config_other_dim))*(other_dim_name!="") + (' : AR'*(CRC_plot==False) + 'CRC '*CRC_plot) + ' in ' + ROI + ' region vs IR in background (at convergence)' + '.png'
                    elif ROI == 'cold':
                        title = method + " : " + rho_name + " = " + str(rho) + (", " + other_dim_name + " = " + str(config_other_dim))*(other_dim_name!="") + (' : AR'*(CRC_plot==False) + 'CRC '*CRC_plot) + ' in ' + ROI + ' region vs IR in background (at convergence)' + '.png'
                fig[fig_nb].savefig(self.subroot_data + 'metrics' + '/' + title)

            for method in method_list: # Loop over methods
                # Swap rho and post smoothing because MLEM and OSEM do not have rho parameter
                if (method == "MLEM" or method == "OSEM"):
                    nb_rho, nb_other_dim = nb_other_dim, nb_rho
                    config["rho"], config_other_dim = config_other_dim, config["rho"]

