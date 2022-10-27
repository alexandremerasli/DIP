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
    def __init__(self,config, *args, **kwargs):
        print("__init__")

    def initializeSpecific(self,config,root, *args, **kwargs):
        # Initialize general variables
        self.initializeGeneralVariables(config,root)
        #vDenoising.initializeSpecific(self,config,root)
        
        if ('ADMMLim' in config["method"]):
            self.i_init = 30 # Remove first iterations
            self.i_init = 1 # Remove first iterations
        else:
            self.i_init = 1

        self.defineTotalNbIter_beta_rho(config["method"], config, config["task"])

        # Create summary writer from tensorboard
        self.writer = SummaryWriter()
        
        #Loading Ground Truth image to compute metrics
        self.image_gt = self.fijii_np(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.raw',shape=(self.PETImage_shape),type='<f')
        if config["FLTNB"] == "double":
            self.image_gt = self.image_gt.astype(np.float64)
            
        # Metrics arrays
        self.PSNR_recon = np.zeros(self.total_nb_iter)
        self.PSNR_norm_recon = np.zeros(self.total_nb_iter)
        self.MSE_recon = np.zeros(self.total_nb_iter)
        self.SSIM_recon = np.zeros(self.total_nb_iter)
        self.MA_cold_recon = np.zeros(self.total_nb_iter)
        self.AR_hot_recon = np.zeros(self.total_nb_iter)
        self.AR_bkg_recon = np.zeros(self.total_nb_iter)
        self.IR_bkg_recon = np.zeros(self.total_nb_iter)
        
    def runComputation(self,config,root):
        print('run computation')