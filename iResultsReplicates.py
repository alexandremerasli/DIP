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
        self.image_gt = self.fijii_np(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.raw',shape=(self.PETImage_shape),type='<f')
        
        # Metrics arrays
        self.total_nb_iter = len(self.beta_list)
        self.PSNR_recon = np.zeros(self.total_nb_iter)
        self.PSNR_norm_recon = np.zeros(self.total_nb_iter)
        self.MSE_recon = np.zeros(self.total_nb_iter)
        self.MA_cold_recon = np.zeros(self.total_nb_iter)
        self.AR_hot_recon = np.zeros(self.total_nb_iter)
        self.AR_bkg_recon = np.zeros(self.total_nb_iter)
        self.IR_bkg_recon = np.zeros(self.total_nb_iter)

    def runComputation(self,config,fixed_config,hyperparameters_config,root): 
        self.writeBeginningImages(self.image_net_input,self.suffix)
        #self.writeCorruptedImage(0,self.total_nb_iter,self.image_corrupt,self.suffix,pet_algo="to fit",iteration_name="(post reconstruction)")

        for i in range(1,self.total_nb_iter+1):
            print("self.beta",self.beta_list[i-1])
            f = np.zeros(self.PETImage_shape,dtype='<f')
            for p in range(1,self.nb_replicates+1):
                self.subroot_p = self.subroot_data + 'replicate_' + str(p) + '/'
                beta_string = ', beta = ' + str(self.beta_list[i-1])

                # Take NNEPPS images for last iteration if NNEPPS was computed
                if (hyperparameters_config["NNEPPS"]):
                    NNEPPS_string = "_NNEPPS"
                else:
                    NNEPPS_string = ""
                if (config["method"] == 'Gong' or config["method"] == 'nested'):
                    pet_algo=config["method"]+"to fit"
                    iteration_name="(post reconstruction)"
                    f_p = self.fijii_np(self.subroot_p+'Block2/out_cnn/'+ format(self.experiment)+'/out_' + self.net + '' + format(hyperparameters_config["max_iter"]) + self.suffix + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading DIP output
                elif ('ADMMLim' in config["method"] or config["method"] == 'MLEM' or config["method"] == 'BSREM' or config["method"] == 'AML'):
                    pet_algo=config["method"]
                    iteration_name="iterations"+beta_string
                    if ('ADMMLim' in config["method"]):
                        subdir = 'ADMM' + '_' + str(fixed_config["nb_threads"])
                        subdir = ''
                        f_p = self.fijii_np(self.subroot_p+'Comparison/' + config["method"] + '/' + self.suffix + '/' + subdir + '/0_' + format(hyperparameters_config["nb_iter_second_admm"]) + '_it' + str(hyperparameters_config["sub_iter_PLL"]) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                    else:
                        f_p = self.fijii_np(self.subroot_p+'Comparison/' + config["method"] + '_beta_' + str(self.beta_list[i-1]) + '/' +  config["method"] + '_beta_' + str(self.beta_list[i-1]) + '_it' + str(fixed_config["max_iter"]) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                f += f_p
                # Metrics for NN output 
                self.compute_IR_bkg(self.PETImage_shape,f_p,self.image_gt,i,self.PSNR_recon,self.PSNR_norm_recon,self.MSE_recon,self.MA_cold_recon,self.AR_hot_recon,self.AR_bkg_recon,self.IR_bkg_recon,self.phantom,writer=self.writer,write_tensorboard=True)
    
            print("Metrics saved in tensorboard")
            self.writer.add_scalar('Image roughness in the background (best : 0)', self.IR_bkg_recon[i-1], i)

            # Compute metrics after averaging images across replicates
            f = f / self.nb_replicates
            self.writeEndImagesAndMetrics(i,self.total_nb_iter,self.PETImage_shape,f,self.suffix,self.phantom,self.net,pet_algo,iteration_name)