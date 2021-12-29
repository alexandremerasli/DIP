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
from utils.utils_func import *
from vDenoising import vDenoising
#from vReconstruction import vReconstruction

class iDenoisingInReconstruction(vDenoising):
    def __init__(self,config,args,root,admm_it):
        self.admm_it = admm_it
        vDenoising.__init__(self,config,args,root)
    
    def initializeSpecific(self, config, args, root):
        print("Denoising in reconstruction")
        vDenoising.initializeSpecific(self,config,args,root)
        # Loading DIP x_label (corrupted image) from block1
        self.image_corrupt = fijii_np(self.subroot+'Block2/x_label/' + format(self.test)+'/'+ format(self.admm_it) +'_x_label' + self.suffix + '.img',shape=(self.PETImage_shape))
        self.net_outputs_path = self.subroot+'Block2/out_cnn/' + format(self.test) + '/out_' + self.net + '' + format(self.admm_it) + suffix_func(config) + '.img'
        self.checkpoint_simple_path = self.subroot+'Block2/checkpoint/'
        self.name_run = ""
        self.sub_iter_DIP = config["sub_iter_DIP"]