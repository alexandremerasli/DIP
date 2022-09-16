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

    def initializeSpecific(self,config3,config4,config2,root):
        # Initialize general variables
        self.initializeGeneralVariables(config3,config4,config2,root)
        vDenoising.initializeSpecific(self,config3,config4,config2,root)
        
        if ('ADMMLim' in config3["method"]):
            self.total_nb_iter = config2["nb_outer_iteration"]
            self.beta = config2["alpha"]
        elif (config3["method"] == 'nested' or config3["method"] == 'Gong'):
            if (config3["task"] == 'post_reco'):
                self.total_nb_iter = config2["sub_iter_DIP"]
            else:
                self.total_nb_iter = config4["max_iter"]
        else:
            self.total_nb_iter = self.max_iter

            if (config3["method"] == 'AML'):
                self.beta = config2["A_AML"]
            if (config3["method"] == 'BSREM' or config3["method"] == 'nested' or config3["method"] == 'Gong'):
                self.rho = config2["rho"]
                self.beta = self.rho
        # Create summary writer from tensorboard
        self.writer = SummaryWriter()
        
        #Loading Ground Truth image to compute metrics
        self.image_gt = self.fijii_np(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.raw',shape=(self.PETImage_shape),type='<f')
        if config3["FLTNB"] == "double":
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
        
    def runComputation(self,config,config3,config4,config2,root):
        print('run computation')