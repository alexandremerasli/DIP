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
#from vGeneral import vGeneral
from vDenoising import vDenoising

class iResultsAlreadyComputed(vDenoising):
    def __init__(self,config):
        print("__init__")

    def initializeSpecific(self,settings_config,fixed_config,hyperparameters_config,root):
        # Initialize general variables
        self.initializeGeneralVariables(settings_config,fixed_config,hyperparameters_config,root)
        vDenoising.initializeSpecific(self,settings_config,fixed_config,hyperparameters_config,root)
        
        if ('ADMMLim' in settings_config["method"]):
            self.total_nb_iter = hyperparameters_config["nb_outer_iteration"]
            self.beta = hyperparameters_config["alpha"]
        elif (settings_config["method"] == 'nested' or settings_config["method"] == 'Gong'):
            if (settings_config["task"] == 'post_reco'):
                self.total_nb_iter = hyperparameters_config["sub_iter_DIP"]
            else:
                self.total_nb_iter = fixed_config["max_iter"]
        else:
            self.total_nb_iter = self.max_iter

            if (settings_config["method"] == 'AML'):
                self.beta = hyperparameters_config["A_AML"]
            if (settings_config["method"] == 'BSREM' or settings_config["method"] == 'nested' or settings_config["method"] == 'Gong'):
                self.rho = hyperparameters_config["rho"]
                self.beta = self.rho
        # Create summary writer from tensorboard
        self.writer = SummaryWriter()
        
        #Loading Ground Truth image to compute metrics
        self.image_gt = self.fijii_np(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.raw',shape=(self.PETImage_shape),type='<f')
        if settings_config["FLTNB"] == "double":
            self.image_gt.astype(np.float64)
            
        # Metrics arrays
        self.PSNR_recon = np.zeros(self.total_nb_iter)
        self.PSNR_norm_recon = np.zeros(self.total_nb_iter)
        self.MSE_recon = np.zeros(self.total_nb_iter)
        self.SSIM_recon = np.zeros(self.total_nb_iter)
        self.MA_cold_recon = np.zeros(self.total_nb_iter)
        self.AR_hot_recon = np.zeros(self.total_nb_iter)
        self.AR_bkg_recon = np.zeros(self.total_nb_iter)
        self.IR_bkg_recon = np.zeros(self.total_nb_iter)
        
    def runComputation(self,config,settings_config,fixed_config,hyperparameters_config,root):
        print('run computation')