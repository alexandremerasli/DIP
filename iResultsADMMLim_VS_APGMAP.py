## Python libraries

# Pytorch
from torch.utils.tensorboard import SummaryWriter

# Math
import numpy as np
import matplotlib.pyplot as plt

# Useful
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

# Local files to import
#from vGeneral import vGeneral
from vDenoising import vDenoising

class iResultsADMMLim_VS_APGMAP(vDenoising):
    def __init__(self,config, *args, **kwargs):
        print("__init__")

    def initializeSpecific(self,config,root, *args, **kwargs):
        # Initialize general variables
        self.initializeGeneralVariables(config,root)
        '''
        vDenoising.initializeSpecific(self,config,root)

        if ('ADMMLim' in config["method"]):
            try:
                self.path_stopping_criterion = self.subroot + self.suffix + '/' + format(0) + '_adaptive_stopping_criteria.log'
                with open(self.path_stopping_criterion) as f:
                    first_line = f.readline() # Read first line to get second one
                    self.total_nb_iter = min(int(f.readline().rstrip()) - 2, config["nb_outer_iteration"] - 1)
            except:
                self.total_nb_iter = config["nb_outer_iteration"] - 1
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
        # Create summary writer from tensorboard
        self.tensorboard = config["tensorboard"]
        self.writer = SummaryWriter()
        
        #Loading Ground Truth image to compute metrics
        self.image_gt = self.fijii_np(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.raw',shape=(self.PETImage_shape),type='<f')
        if config["FLTNB"] == "double":
            self.image_gt.astype(np.float64)

        # Defining ROIs
        self.bkg_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "background_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type='<f')
        self.hot_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "tumor_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type='<f')
        self.cold_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "cold_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type='<f')
        self.phantom_ROI = self.get_phantom_ROI(self.phantom)

        # Metrics arrays
        self.PSNR_recon = np.zeros(self.total_nb_iter)
        self.PSNR_norm_recon = np.zeros(self.total_nb_iter)
        self.MSE_recon = np.zeros(self.total_nb_iter)
        self.SSIM_recon = np.zeros(self.total_nb_iter)
        self.MA_cold_recon = np.zeros(self.total_nb_iter)
        self.AR_hot_recon = np.zeros(self.total_nb_iter)
        self.AR_bkg_recon = np.zeros(self.total_nb_iter)
        self.IR_bkg_recon = np.zeros(self.total_nb_iter)
        '''

    def imageNames(self):
        
        if (hasattr(self,'beta')):
            beta_string = ', beta = ' + str(self.beta)

        if (config["method"] == "nested" or config["method"] == "Gong"):
            self.writeBeginningImages(self.suffix,self.image_net_input) # Write GT and DIP input
            #self.writeCorruptedImage(0,self.total_nb_iter,self.image_corrupt,self.suffix,pet_algo="to fit",iteration_name="(post reconstruction)")
        else:
            self.writeBeginningImages(self.suffix) # Write GT

        if (self.FLTNB == 'float'):
            type = '<f'
        elif (self.FLTNB == 'double'):
            type = '<d'

        f = np.zeros(self.PETImage_shape,dtype=type)
        f_p = np.zeros(self.PETImage_shape,dtype=type)

        if ('ADMMLim' in config["method"]):
            i_init = 20
        else:
            i_init = 1

        #for i in range(i_init,self.total_nb_iter+1):
        for i in range(self.total_nb_iter,self.total_nb_iter+1):
            IR = 0
            for p in range(1,self.nb_replicates+1):
                if (config["average_replicates"] or (config["average_replicates"] == False and p == self.replicate)):
                    self.subroot_p = self.subroot_data + 'debug/'*self.debug + '/' + self.phantom + '/' + 'replicate_' + str(p) + '/' + self.method + '/' # Directory root

                    # Take NNEPPS images if NNEPPS is asked for this run
                    if (config["NNEPPS"]):
                        NNEPPS_string = "_NNEPPS"
                    else:
                        NNEPPS_string = ""
                    if (config["method"] == 'Gong' or config["method"] == 'nested'):
                        if ('post_reco' in config["task"]):
                            pet_algo=config["method"]+"to fit"
                            iteration_name="(post reconstruction)"
                        else:
                            pet_algo=config["method"]
                            iteration_name="iterations"
                        if ('post_reco' in config["task"]):
                            try:
                                f_p = self.fijii_np(self.subroot_p+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.net + '' + format(0) + '_epoch=' + format(i-i_init) + NNEPPS_string + '.img',shape=(self.PETImage_shape),type='<f') # loading DIP output
                            except: # ES point is reached
                                break
                        else:
                            f_p = self.fijii_np(self.subroot_p+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.net + '' + format(i-i_init) + "FINAL" + NNEPPS_string + '.img',shape=(self.PETImage_shape),type='<f') # loading DIP output
                        if config["FLTNB"] == "double":
                            f_p.astype(np.float64)
                    elif ('ADMMLim' in config["method"] or config["method"] == 'MLEM' or config["method"] == 'OSEM' or config["method"] == 'BSREM' or config["method"] == 'AML' or config["method"] == 'APGMAP'):
                        pet_algo=config["method"]
                        iteration_name = "iterations"
                        if (hasattr(self,'beta')):
                            iteration_name += beta_string
                        if ('ADMMLim' in config["method"]):
                            subdir = 'ADMM' + '_' + str(config["nb_threads"])
                            subdir = ''
                            #f_p = self.fijii_np(self.subroot_p + self.suffix + '/' + subdir + '/0_' + format(i) + '_it' + str(config["nb_inner_iteration"]) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                            #f_p = self.fijii_np(self.subroot_p + self.suffix + '/' + subdir + '/0_' + format(i) + '_it1' + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                            #f_p = self.fijii_np(self.subroot_p + self.suffix + '/' + subdir + '/0_1'  + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                            f_p = self.fijii_np(self.subroot_p + self.suffix + '/' + subdir + '/0'  + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                        #elif (config["method"] == 'BSREM'):
                        #    f_p = self.fijii_np(self.subroot_p + self.suffix + '/' +  config["method"] + '_beta_' + str(self.beta) + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                        else:
                            f_p = self.fijii_np(self.subroot_p + self.suffix + '/' +  config["method"] + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output

                    # Compute IR metric (different from others with several replicates)
                    self.compute_IR_bkg(self.PETImage_shape,f_p,i-i_init,self.IR_bkg_recon,self.phantom)

                    # Specific average for IR
                    if (config["average_replicates"] == False and p == self.replicate):
                        IR = self.IR_bkg_recon[i-i_init]
                    elif (config["average_replicates"]):
                        IR += self.IR_bkg_recon[i-i_init] / self.nb_replicates
                        
                    if (config["average_replicates"]): # Average images across replicates (for metrics except IR)
                        f += f_p / self.nb_replicates
                    elif (config["average_replicates"] == False and p == self.replicate):
                        f = f_p
        
    def compareImages(self,suffix):
        if (self.tensorboard):
            self.write_image_tensorboard(self.writer,self.image_method,self.method + " at convergence",suffix,self.image_gt,0,full_contrast=True) # ADMMLim at convergence in tensorboard
            #self.write_image_tensorboard(self.writer,self.image_APGMAP,"APGMAP at convergence",suffix,self.image_gt,0,full_contrast=True) # APGMAP at convergence in tensorboard
            
            
    def runComputation(self,config,root):

        plot_all_replicates_curves = True
        if plot_all_replicates_curves:
            color_avg = 'black'
        else:
            color_avg = None

        for ROI in ['hot','cold']:
            # Initialize 3 figures
            fig, ax = [None] * 3, [None] * 3
            for fig_nb in range(3):
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

                # Create numpy array with same number of iterations for each case
                IR_final_array = []
                metrics_final_array = []
                for rho_idx in range(nb_rho):
                    IR_final_array.append(np.zeros((nb_replicates,len_mini[rho_idx])))
                    metrics_final_array.append(np.zeros((nb_replicates,len_mini[rho_idx])))
                    for common_it in range(len_mini[rho_idx]):
                        for idx_replicate in range(nb_replicates):
                            IR_final_array[rho_idx][idx_replicate,common_it] = IR_final[idx_replicate + nb_replicates*rho_idx][common_it]
                            metrics_final_array[rho_idx][idx_replicate,common_it] = metrics_final[idx_replicate + nb_replicates*rho_idx][common_it]

                # Plot 3 figures for each ROI : tradeoff curve with iteration (metric VS IR), bias with iterations, and tradeoff curve at convergence
                reg = [None] * 3
                for fig_nb in range(3):
                    if (fig_nb == 0):
                        reg[fig_nb] = [None] * nb_rho
                    elif (fig_nb == 2):
                        reg[fig_nb] = []                
                    for rho_idx in range(nb_rho):
                        for replicate in range(nb_replicates):
                            case = replicate + nb_replicates*rho_idx
                            if (fig_nb == 0): # Plot tradeoff curves with iterations
                                idx_sort = np.argsort(IR_final[case])
                                idx_sort = np.arange(len(IR_final[case]))
                                if (plot_all_replicates_curves):
                                    if (method == "nested" or method == "Gong"):
                                        ax[fig_nb].plot(100*IR_final[case][0],metrics_final[case][0],'o', mfc='none',color='black',label='_nolegend_') # IR in %
                                    else:
                                        ax[fig_nb].plot(100*IR_final[case][idx_sort],metrics_final[case][idx_sort],'o',label='_nolegend_') # IR in %                     
                        
                        '''
                        if (fig_nb == 0):
                            reg[fig_nb][rho_idx] = []
                            idx_sort = np.argsort(IR_final[case])
                            for it in idx_sort:
                                cases = np.arange(nb_replicates*rho_idx,nb_replicates*(rho_idx+1))
                                print(np.array(IR_final[cases]))
                                print(IR_final[cases][100])
                                print(np.array(IR_final).shape)#[cases,it])
                                print(np.array(metrics_final)[cases,it])
                                reg[fig_nb][rho_idx].append(self.linear_regression(100*IR_final_array[rho_idx][:,it],metrics_final_array[rho_idx][:,it]))
                        '''
                        for replicate in range(nb_replicates):
                            case = replicate + nb_replicates*rho_idx
                            if (fig_nb == 1): # Plot bias curves
                                if (plot_all_replicates_curves):
                                    ax[fig_nb].plot(np.arange(0,len(metrics_final[case])),metrics_final[case],'o',label='_nolegend_') # Plot bias curves with iterations for each replicate

                    '''
                    for replicate in range(nb_replicates):
                        cases = replicate + nb_replicates*np.arange(nb_rho)
                        if (fig_nb == 2): # Plot tradeoff curves at convergence
                            if (plot_all_replicates_curves):
                                if (method == "nested" or method == "Gong"):
                                    ax[fig_nb].plot(100*IR_final[cases][0],metrics_final[cases][0],'o', mfc='none',color='black',label='_nolegend_') # IR in %
                                else:
                                    ax[fig_nb].plot(100*IR_final_array[:][:,-1],metrics_final_array[:][:,-1],label='_nolegend_') # IR in %                     
                                    ax[fig_nb].plot(100*np.array(IR_final)[cases,-1],np.array(metrics_final)[cases,-1],label='_nolegend_') # IR in %                     
                    '''

                    '''
                    if (fig_nb == 2): # Plot tradeoff curves at convergence
                        for rho_idx in range(nb_rho):
                            cases = np.arange(nb_replicates*rho_idx,nb_replicates*(rho_idx+1))
                            reg[fig_nb].append(self.linear_regression(100*IR_final_array[rho_idx][:,-1],metrics_final_array[rho_idx][:,-1]))
                    '''
                    #'''
                    avg_metrics = []
                    avg_IR = []
                    std_metrics = []
                    std_IR = []
                    
                    for rho_idx in range(nb_rho):
                        # Compute average of tradeoff and bias curves with iterations
                        avg_metrics.append(np.sum(metrics_final_array[rho_idx],axis=0) / nb_replicates)
                        avg_IR.append(np.sum(IR_final_array[rho_idx],axis=0) / nb_replicates)
                        # Compute std bias curves with iterations
                        #std_metrics.append(np.sqrt(np.sum((metrics_final_array[rho_idx,:] - np.array(avg_metrics)[rho_idx,:])**2,axis=0) / nb_replicates))
                        #std_IR.append(np.sqrt(np.sum((IR_final_array[rho_idx,:]- np.array(avg_IR)[rho_idx,:])**2,axis=0) / nb_replicates))

                        if (fig_nb == 0):
                            #ax[fig_nb].plot(100*avg_IR[rho_idx],avg_metrics[rho_idx],'-o',color=color_avg)
                            #ax[fig_nb].fill(np.concatenate((100*(avg_IR[rho_idx] - np.sign(np.array(reg[fig_nb][rho_idx]))*std_IR[rho_idx]),100*(avg_IR[rho_idx][::-1] + np.sign(np.array(reg[fig_nb][rho_idx][::-1]))*std_IR[rho_idx][::-1]))),np.concatenate((avg_metrics[rho_idx]-std_metrics[rho_idx],avg_metrics[rho_idx][::-1]+std_metrics[rho_idx][::-1])), alpha = 0.4, label='_nolegend_')
                            ax[fig_nb].set_xlabel('Image Roughness in the background (%)', fontsize = 18)
                            ax[fig_nb].set_ylabel('Absolute bias (AU)', fontsize = 18)
                        #'''
                        if (fig_nb == 1):
                            # Plot average and std of bias curves with iterations
                            ax[fig_nb].plot(np.arange(0,len(avg_metrics[rho_idx])),avg_metrics[rho_idx],color=color_avg)
                            # Plot dashed line for target value, according to ROI
                            if ROI == 'hot':
                                ax[fig_nb].hlines(400,xmin=0,xmax=len(avg_metrics[rho_idx])-1,color='grey',linestyle='dashed',label='_nolegend_')
                            else:
                                ax[fig_nb].hlines(10,xmin=0,xmax=len(avg_metrics[rho_idx])-1,color='grey',linestyle='dashed',label='_nolegend_')
                            #ax[fig_nb].fill_between(np.arange(0,len(avg_metrics[rho_idx])), avg_metrics[rho_idx] - std_metrics[rho_idx], avg_metrics[rho_idx] + std_metrics[rho_idx], alpha = 0.4, label='_nolegend_')
                            ax[fig_nb].set_xlabel('Iterations', fontsize = 18)
                            ax[fig_nb].set_ylabel('Bias (AU)', fontsize = 18)
                            ax[fig_nb].set_title(method + " reconstruction averaged on " + str(nb_replicates) + " replicates")
                        #'''
                        replicates_legend.append(method + " : rho = " + str(config["rho"][rho_idx]))

                    ax[fig_nb].legend(replicates_legend)
                    '''
                    if (fig_nb == 2):
                        ax[fig_nb].plot([100*np.array(avg_IR)[:,-1] for rho_idx in range(nb_rho)],np.array(avg_metrics)[:,-1],'-o',color=color_avg)
                        ax[fig_nb].fill(np.concatenate((100*(np.array(avg_IR)[:,-1] - np.sign(np.array(reg[fig_nb]))*np.array(std_IR)[:,-1]),100*(np.array(avg_IR[::-1])[:,-1] + np.sign(np.array(reg[fig_nb][::-1]))*np.array(std_IR[::-1])[:,-1]))),np.concatenate((np.array(avg_metrics)[:,-1]-np.array(std_metrics)[:,-1],np.array(avg_metrics[::-1])[:,-1]+np.array(std_metrics[::-1])[:,-1])), alpha = 0.4, label='_nolegend_')
                        ax[fig_nb].set_xlabel('Image Roughness in the background (%)', fontsize = 18)
                        ax[fig_nb].set_ylabel('Absolute bias (AU)', fontsize = 18)
                    '''

            # Saving figures locally in png
            for fig_nb in range(3):
                rho = config["rho"]
                if (fig_nb == 0):
                    if ROI == 'hot':
                        title = method + " rho = " + str(rho) + 'AR in ' + ROI + ' region vs IR in background (with iterations)' + '.png'
                    elif ROI == 'cold':
                        title = method + " rho = " + str(rho) + 'AR in ' + ROI + ' region vs IR in background (with iterations)' + '.png'
                elif (fig_nb == 1):
                    if ROI == 'hot':
                        title = method + " rho = " + str(rho) + 'AR in ' + ROI + ' region for ' + str(nb_replicates) + ' replicates' + '.png'
                    elif ROI == 'cold':
                        title = method + " rho = " + str(rho) + 'AR in ' + ROI + ' region for ' + str(nb_replicates) + ' replicates' + '.png'
                elif (fig_nb == 2):
                    if ROI == 'hot':
                        title = method + " rho = " + str(rho) + 'AR in ' + ROI + ' region vs IR in background (at convergence)' + '.png'
                    elif ROI == 'cold':
                        title = method + " rho = " + str(rho) + 'AR in ' + ROI + ' region vs IR in background (at convergence)' + '.png'
                fig[fig_nb].savefig(self.subroot_data + 'metrics' + '/' + title)