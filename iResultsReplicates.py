## Python libraries

# Pytorch
from torch.utils.tensorboard import SummaryWriter

# Math
import numpy as np
import matplotlib.pyplot as plt

# Useful
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio

# Local files to import
from iResults import iResults

class iResultsReplicates(iResults):
    def __init__(self,config):
        print("__init__")

    def initializeSpecific(self,fixed_config,hyperparameters_config,root):
        # Initialize general variables
        self.initializeGeneralVariables(fixed_config,hyperparameters_config,root)
        iResults.initializeSpecific(self,fixed_config,hyperparameters_config,root)

        # Create summary writer from tensorboard
        self.writer = SummaryWriter()
        
        #Loading Ground Truth image to compute metrics
        self.image_gt = self.fijii_np(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.raw',shape=(self.PETImage_shape))
        
        # Metrics arrays
        self.PSNR_recon = np.zeros(self.total_nb_iter)
        self.PSNR_norm_recon = np.zeros(self.total_nb_iter)
        self.MSE_recon = np.zeros(self.total_nb_iter)
        self.MA_cold_recon = np.zeros(self.total_nb_iter)
        self.CRC_hot_recon = np.zeros(self.total_nb_iter)
        self.CRC_bkg_recon = np.zeros(self.total_nb_iter)
        self.IR_bkg_recon = np.zeros(self.total_nb_iter)

    def writeEndImagesAndMetrics(self,i,max_iter,PETImage_shape,f,suffix,phantom,net,pet_algo,iteration_name='iterations'):
        # Metrics for NN output
        self.compute_metrics(PETImage_shape,f,self.image_gt,i,self.PSNR_recon,self.PSNR_norm_recon,self.MSE_recon,self.MA_cold_recon,self.CRC_hot_recon,self.CRC_bkg_recon,self.IR_bkg_recon,phantom,writer=self.writer,write_tensorboard=True)
    
        # Display CRC vs STD curve in tensorboard
        if (i>max_iter - min(max_iter,10)):
            # Creating matplotlib figure
            plt.plot(self.IR_bkg_recon,self.CRC_hot_recon,linestyle='None',marker='x')
            plt.xlabel('IR')
            plt.ylabel('CRC')
            # Adding this figure to tensorboard
            self.writer.flush()
            self.writer.add_figure('CRC in hot region vs IR in background', plt.gcf(),global_step=i,close=True)
            self.writer.close()

    def runComputation(self,config,fixed_config,hyperparameters_config,root): 
        # Compromise curves for last iteration with several replicates

        self.writeBeginningImages(self.image_net_input,self.suffix)
        #self.writeCorruptedImage(0,self.total_nb_iter,self.image_corrupt,self.suffix,pet_algo="to fit",iteration_name="(post reconstruction)")

        self.total_nb_iter = len(self.beta_list)
        for i in range(1,self.total_nb_iter+1):
            print("self.beta",self.beta_list[i-1])
            f = np.zeros(self.PETImage_shape,dtype='<f')
            for p in range(1,self.nb_replicates+1):
                self.subroot = self.subroot_data + 'replicate_' + str(p) + '/'
                beta_string = ', beta = ' + str(self.beta_list[i-1])

                # Take NNEPPS images for last iteration if NNEPPS was computed
                if (fixed_config["NNEPPS"]):
                    NNEPPS_string = "_NNEPPS"
                else:
                    NNEPPS_string = ""
                if (config["method"] == 'Gong' or config["method"] == 'nested'):
                    pet_algo=config["method"]+"to fit"
                    iteration_name="(post reconstruction)"
                    f += self.fijii_np(self.subroot+'Block2/out_cnn/'+ format(self.experiment)+'/out_' + self.net + '' + format(hyperparameters_config["max_iter"]) + self.suffix + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading DIP output
                elif (config["method"] == 'ADMMLim' or config["method"] == 'MLEM' or config["method"] == 'BSREM' or config["method"] == 'AML'):
                    pet_algo=config["method"]
                    iteration_name="iterations"+beta_string
                    if (config["method"] == 'ADMMLim'):
                        f += self.fijii_np(self.subroot+'Comparison/' + config["method"] + '/' + self.suffix + '/ADMM/0_' + format(hyperparameters_config["nb_iter_second_admm"]) + '_it' + str(hyperparameters_config["sub_iter_MAP"]) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                    else:
                        f += self.fijii_np(self.subroot+'Comparison/' + config["method"] + '_beta_' + str(self.beta_list[i-1]) + '/' +  config["method"] + '_beta_' + str(self.beta_list[i-1]) + '_it' + str(fixed_config["max_iter"]) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
            # Compute metrics after averaging images across replicates
            f = f / self.nb_replicates
            self.writeEndImagesAndMetrics(i,self.total_nb_iter,self.PETImage_shape,f,self.suffix,self.phantom,self.net,pet_algo,iteration_name)
