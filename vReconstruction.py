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
from vGeneral import vGeneral

import abc
class vReconstruction(abc.ABC):
    @abc.abstractmethod
    def __init__(self,config,args,root):
        print("__init__")
        self.test = "not updated"
        self.args = args 

    def runReconstruction(self):
        """ Implement me! """
        pass

    def runRayTune(self,config,args,root):
        config_combination = 1
        for i in range(len(config)):
            config_combination *= len(list(list(config.values())[i].values())[0])

        if args.proc == 'CPU':
            resources_per_trial = {"cpu": 1, "gpu": 0}
        elif args.proc == 'GPU':
            resources_per_trial = {"cpu": 0, "gpu": 0.1} # "gpu": 1 / config_combination
            #resources_per_trial = {"cpu": 0, "gpu": 1} # "gpu": 1 / config_combination
        elif args.proc == 'both':
            resources_per_trial = {"cpu": 10, "gpu": 1} # not efficient

        #reporter = CLIReporter(
        #    parameter_columns=['lr'],
        #    metric_columns=['mse'])

        # Start tuning of hyperparameters = start each admm computation in parallel
        #try: # resume previous run (if it exists)
        #    anaysis_raytune = tune.run(partial(admm_loop,args=args,root=os.getcwd()), config=config, local_dir = os.getcwd() + '/runs', name=suffix_func(config) + str(args.max_iter), resources_per_trial = resources_per_trial, resume = "ERRORED_ONLY")#, progress_reporter = reporter)
        #except: # do not resume previous run because there is no previous one
        #   anaysis_raytune = tune.run(partial(admm_loop,args=args,root=os.getcwd()), config=config, local_dir = os.getcwd() + '/runs', name=suffix_func(config) + "_max_iter=" + str(args.max_iter), resources_per_trial = resources_per_trial)#, progress_reporter = reporter)



        tune.run(partial(self.do_everything,args=args,root=root), config=config,local_dir = os.getcwd() + '/runs', resources_per_trial = resources_per_trial)#, progress_reporter = reporter)
    
    def do_everything(self,config,args,root):
        self.initializeSpecific(config,args,root)
        self.runReconstruction(config,args,root)

    def initializeSpecific(self, config, args, root):
        vGeneral.initializeSpecific(self,config,args,root)
        vGeneral.createDirectory(self)

        # Specific hyperparameters for reconstruction module (Do it here to have raytune config hyperparameters selection)
        self.config = config
        self.rho = config["rho"]
        self.alpha = config["alpha"]
        self.sub_iter_MAP = config["sub_iter_MAP"]

        # Initialize and save mu variable from ADMM
        self.mu = 0* np.ones((self.PETImage_shape[0], self.PETImage_shape[1]), dtype='<f')
        save_img(self.mu,self.subroot+'Block2/mu/'+ format(self.test)+'/mu_' + format(-1) + self.suffix + '.img')

        # Ininitializing DIP output and first image x with f_init and image_init
        if (self.config["method"] == "nested"): # Nested needs 1 to not add any prior information at the beginning, and to initialize x computation to uniform with 1
            self.f_init = np.ones((self.PETImage_shape[0],self.PETImage_shape[1]), dtype='<f')
            self.image_init_path_without_extension = '1_im_value_cropped'
            # self.image_init_path_without_extension = 'BSREM_it30_REF_cropped'
        elif (self.config["method"] == "Gong"): # Gong initialization with 60th iteration of MLEM (normally, DIP trained with this image as label...)
            self.f_init = fijii_np(self.subroot + 'Data/initialization/' + 'BSREM_it30_REF_cropped.img',shape=(self.PETImage_shape))
            #self.f_init = fijii_np(self.subroot + 'Data/initialization/' + 'MLEM_it60_REF_cropped.img',shape=(self.PETImage_shape))
            self.image_init_path_without_extension = '1_im_value_cropped' # OPTITR initialization, so firts in MLEM (1 iteration) computation. Not written in Gong, but perhaps not giving any prior information

        # Launch short MLEM reconstruction
        path_mlem_init = self.subroot + 'Data/MLEM_reco_for_init/' + self.phantom
        my_file = Path(path_mlem_init + '/' + self.phantom + '/' + self.phantom + '_it1.img')
        if (~my_file.is_file()):
            header_file = ' -df ' + self.subroot + 'Data/database_v2/' + self.phantom + '/data' + self.phantom[-1] + '/data' + self.phantom[-1]  + '.cdh' # PET data path
            executable = 'castor-recon'
            optimizer = 'MLEM'
            output_path = ' -dout ' + path_mlem_init # Output path for CASTOR framework
            dim = ' -dim ' + self.PETImage_shape_str
            vox = ' -vox 4,4,4'
            vb = ' -vb 0'
            it = ' -it 1:1'
            opti = ' -opti ' + optimizer
            os.system(executable + dim + vox + output_path + header_file + vb + it + opti) # + ' -fov-out 95')