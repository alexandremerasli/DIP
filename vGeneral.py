## Python libraries

# Pytorch
import torch
from torch.utils.tensorboard import SummaryWriter

# Useful
import os
from pathlib import Path
import argparse
import time
import subprocess
from functools import partial
from ray import tune

# Math
import numpy as np
import matplotlib.pyplot as plt

# Local files to import
from utils.utils_func import *

import abc
class vGeneral(abc.ABC):
    @abc.abstractmethod
    def __init__(self,config,args,root):
        print("__init__")
        self.test = "not updated"
        self.args = args

    def initializeSpecific(self, config, args, root):
        """General variables"""
        # Initialize useful variables
        self.subroot = root + '/data/Algo/'  # Directory root
        # Retrieving arguments in this function
        self.max_iter = self.args.max_iter # Outer iterations
        self.processing_unit = self.args.proc
        self.phantom = config["image"]
        self.test = 24  # Label of the experiment

        # Define PET input dimensions according to input data dimensions
        self.PETImage_shape_str = read_input_dim(self.subroot+'Data/database_v2/' + self.phantom + '/' + self.phantom + '.hdr')
        self.PETImage_shape = input_dim_str_to_list(self.PETImage_shape_str)



        # Do it here to have raytune config hyperparameters selection
        # config dictionnary for hyperparameters
        self.config = config
        if config["input"] == 'uniform': # Do not standardize or normalize if uniform, otherwise NaNs
            config["scaling"] = "nothing"
        self.scaling_input = config["scaling"]

        if config["method"] == 'Gong':
            # config["scaling"] = 'positive_normalization' # Will still introduce bias as mu can be negative
            config["scaling"] = "normalization" # Will still introduce bias as mu can be negative, but this is Gong's method

        self.phantom = config["image"]
        # Define ROIs for image0 phantom, otherwise it is already done in the database
        if (self.phantom == "image0"):
            define_ROI_image0(self.PETImage_shape)

        self.test = 24  # Label of the experiment

        self.suffix = suffix_func(self.config) # self.suffix to make difference between raytune runs (different hyperparameters)

        # Define PET input dimensions according to input data dimensions
        self.PETImage_shape_str = read_input_dim(self.subroot+'Data/database_v2/' + self.phantom + '/' + self.phantom + '.hdr')
        self.PETImage_shape = input_dim_str_to_list(self.PETImage_shape_str)

        # Save this configuration of hyperparameters, and reload with suffix
        np.save(self.subroot + 'Config/' + self.suffix + '.npy', self.config) # Save this self.configuration of hyperparameters, and reload it at the beginning of block 2 thanks to self.suffix (passed in subprocess call arguments


    def createDirectory(self):
        Path(self.subroot+'Block1/' + self.suffix + '/before_eq22').mkdir(parents=True, exist_ok=True) # CASToR path
        Path(self.subroot+'Block1/' + self.suffix + '/during_eq22').mkdir(parents=True, exist_ok=True) # CASToR path
        Path(self.subroot+'Block1/' + self.suffix + '/out_eq22').mkdir(parents=True, exist_ok=True) # CASToR path

        Path(self.subroot+'Images/out_final/'+format(self.test)+'/').mkdir(parents=True, exist_ok=True) # Output of the framework (Last output of the DIP)

        Path(self.subroot+'Block2/checkpoint/'+format(self.test)+'/').mkdir(parents=True, exist_ok=True)
        Path(self.subroot+'Block2/out_cnn/'+ format(self.test)+'/').mkdir(parents=True, exist_ok=True) # Output of the DIP block every outer iteration
        Path(self.subroot+'Block2/out_cnn/vae').mkdir(parents=True, exist_ok=True) # Output of the DIP block every outer iteration
        Path(self.subroot+'Block2/out_cnn/cnn_metrics/'+ format(self.test)+'/').mkdir(parents=True, exist_ok=True) # DIP block metrics
        Path(self.subroot+'Block2/out_cnn/'+ format(self.test)+'/like/').mkdir(parents=True, exist_ok=True) # folder for Likelihood calculation (using CASTOR)
        Path(self.subroot+'Block2/x_label/'+format(self.test) + '/').mkdir(parents=True, exist_ok=True) # x corrupted - folder

        Path(self.subroot+'Block2/checkpoint/'+format(self.test)+'/').mkdir(parents=True, exist_ok=True)
        Path(self.subroot+'Block2/out_cnn/'+ format(self.test)+'/').mkdir(parents=True, exist_ok=True)
        Path(self.subroot+'Block2/mu/'+ format(self.test)+'/').mkdir(parents=True, exist_ok=True)
        Path(self.subroot+'Block2/out_cnn/cnn_metrics/'+ format(self.test)+'/').mkdir(parents=True, exist_ok=True)

        Path(self.subroot+'Comparison/MLEM/').mkdir(parents=True, exist_ok=True) # CASTor path
        Path(self.subroot+'Comparison/BSREM/').mkdir(parents=True, exist_ok=True) # CASTor path

        Path(self.subroot+'Config/').mkdir(parents=True, exist_ok=True) # CASTor path

        Path(self.subroot+'Data/initialization').mkdir(parents=True, exist_ok=True)