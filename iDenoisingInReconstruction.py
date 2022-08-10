## Python libraries

# Pytorch
import torch
from torchsummary import summary

# Useful
import os
from datetime import datetime

# Math
import numpy as np
import matplotlib.pyplot as plt

# Local files to import
from vDenoising import vDenoising

class iDenoisingInReconstruction(vDenoising):
    def __init__(self,config,global_it):
        self.global_it = global_it
    
    def initializeSpecific(self,fixed_config,hyperparameters_config,root):
        print("Denoising in reconstruction")
        vDenoising.initializeSpecific(self,fixed_config,hyperparameters_config,root)
        # Loading DIP x_label (corrupted image) from block1
        print(self.subroot+'Block2/x_label/' + format(self.experiment)+'/'+ format(self.global_it) +'_x_label' + self.suffix + '.img')
        try:
            self.image_corrupt = self.fijii_np(self.subroot+'Block2/x_label/' + format(self.experiment)+'/'+ format(self.global_it) +'_x_label' + self.suffix + '.img',shape=(self.PETImage_shape))
        except:
            self.image_corrupt = self.fijii_np(self.subroot+'Block2/x_label/' + format(self.experiment)+'/'+ format(self.global_it) +'_x_label' + self.suffix + '.img',shape=(self.PETImage_shape),type='<f')
        print(self.image_corrupt)
        self.net_outputs_path = self.subroot+'Block2/out_cnn/' + format(self.experiment) + '/out_' + self.net + '' + format(self.global_it) + self.suffix + '.img'
        self.checkpoint_simple_path = self.subroot+'Block2/checkpoint/'
        self.name_run = ""
        self.sub_iter_DIP = hyperparameters_config["sub_iter_DIP"]
        self.sub_iter_DIP_initial = hyperparameters_config["sub_iter_DIP_initial"]
        print(self.sub_iter_DIP)