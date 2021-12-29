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
class vReconstruction(abc.ABC):
    @abc.abstractmethod
    def __init__(self,config,args,root):
        print("__init__")
        self.test = "not updated"
        self.args = args

        # Initialize useful variables
        self.subroot = root + '/data/Algo/'  # Directory root

        # Retrieving arguments in this function
        self.processing_unit = self.args.proc
        self.finetuning = self.args.finetuning # Finetuning or not for the DIP optimizations (block 2)
        self.max_iter = self.args.max_iter # Outer iterations 

        # Metrics arrays
        self.PSNR_recon = np.zeros(self.max_iter)
        self.PSNR_norm_recon = np.zeros(self.max_iter)
        self.MSE_recon = np.zeros(self.max_iter)
        self.MA_cold_recon = np.zeros(self.max_iter)
        self.CRC_hot_recon = np.zeros(self.max_iter)
        self.CRC_bkg_recon = np.zeros(self.max_iter)
        self.IR_bkg_recon = np.zeros(self.max_iter)

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
        # Do it here to have raytune config hyperparameters selection
        # config dictionnary for hyperparameters
        self.config = config
        self.rho = config["rho"]
        self.alpha = config["alpha"]
        self.sub_iter_MAP = config["sub_iter_MAP"]
        self.net = config["net"]
        if config["input"] == 'uniform': # Do not standardize or normalize if uniform, otherwise NaNs
            config["scaling"] = "nothing"
        self.scaling_input = config["scaling"]

        if config["method"] == 'Gong':
            # config["scaling"] = 'positive_normalization' # Will still introduce bias as mu can be negative
            config["scaling"] = "normalization" # Will still introduce bias as mu can be negative, but this is Gong's method

        self.phantom = config["image"]
        self.test = 24  # Label of the experiment

        self.suffix = suffix_func(self.config) # self.suffix to make difference between raytune runs (different hyperparameters)

        '''
        Creation of directory 
        '''
        
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

        np.save(self.subroot + 'Config/' + self.suffix + '.npy', self.config) # Save this self.configuration of hyperparameters, and reload it at the beginning of block 2 thanks to self.suffix (passed in subprocess call arguments

        # Define PET input dimensions according to input data dimensions
        self.PETImage_shape_str = read_input_dim(self.subroot+'Data/database_v2/' + self.phantom + '/' + self.phantom + '.hdr')
        self.PETImage_shape = input_dim_str_to_list(self.PETImage_shape_str)

        """
        Initialization : variables
        """

        # ADMM variables
        self.mu = 0* np.ones((self.PETImage_shape[0], self.PETImage_shape[1]), dtype='<f')
        save_img(self.mu,self.subroot+'Block2/mu/'+ format(self.test)+'/mu_' + format(-1) + self.suffix + '.img') # saving mu

        self.x_out = np.zeros((self.PETImage_shape[0], self.PETImage_shape[1]), dtype='<f') # output of DIP

        self.writer = SummaryWriter()

        # Define ROIs for image0 phantom, otherwise it is already done in the database
        if (self.phantom == "image0"):
            define_ROI_image0(self.PETImage_shape)
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

        """
        CASTOR framework
        """

        ## Loading images (NN input, first DIP output, GT)
        # Loading DIP input
        # Creating random image input for DIP while we do not have CT, but need to be removed after
        create_input(self.net,self.PETImage_shape,self.config) # to be removed when CT will be used instead of random input. DO NOT PUT IT IN BLOCK 2 !!!
        # Loading DIP input (we do not have CT-map, so random image created in block 1)
        self.image_net_input = load_input(self.net,self.PETImage_shape,self.config) # Scaling of network input. DO NOT CREATE RANDOM INPUT IN BLOCK 2 !!! ONLY AT THE BEGINNING, IN BLOCK 1    
        #image_atn = fijii_np(self.subroot+'Data/database_v2/' + self.phantom + '/' + self.phantom + '_atn.raw',shape=(self.PETImage_shape))
        #write_image_tensorboard(writer,image_atn,"Attenuation map (FULL CONTRAST)",self.suffix,image_gt,0,full_contrast=True) # Attenuation map in tensorboard
        image_net_input_scale = rescale_imag(self.image_net_input,self.scaling_input)[0] # Rescale of network input
        # DIP input image, numpy --> torch
        self.image_net_input_torch = torch.Tensor(image_net_input_scale)
        # Adding dimensions to fit network architecture
        if (self.net == 'DIP' or self.net == 'DIP_VAE'):
            self.image_net_input_torch = self.image_net_input_torch.view(1,1,self.PETImage_shape[0],self.PETImage_shape[1]) # For DIP
        else:
            if (self.net == 'DD'):
                input_size_DD = int(self.PETImage_shape[0] / (2**self.config["d_DD"])) # if original Deep Decoder (i.e. only with decoder part)
                self.image_net_input_torch = self.image_net_input_torch.view(1,self.config["k_DD"],input_size_DD,input_size_DD) # For Deep Decoder, if original Deep Decoder (i.e. only with decoder part)
            elif (self.net == 'DD_AE'):
                input_size_DD = self.PETImage_shape[0] # if auto encoder based on Deep Decoder
                self.image_net_input_torch = self.image_net_input_torch.view(1,1,input_size_DD,input_size_DD) # For Deep Decoder, if auto encoder based on Deep Decoder
        torch.save(self.image_net_input_torch,self.subroot + 'Data/initialization/image_' + self.net + '_input_torch.pt')

        # Ininitializing DIP output and first image x with f_init and image_init
        #self.f_init = fijii_np(self.subroot + 'Data/initialization/' + 'BSREM_it30_REF_cropped.img',shape=(self.PETImage_shape))
        #self.f_init = fijii_np(self.subroot+'Comparison/MLEM/MLEM_converge_avec_post_filtre.img',shape=(self.PETImage_shape))
        if (self.config["method"] == "nested"): # Nested needs 1 to not add any prior information at the beginning, and to initialize x computation to uniform with 1
            self.f_init = np.ones((self.PETImage_shape[0],self.PETImage_shape[1]), dtype='<f')
            self.image_init_path_without_extension = '1_im_value_cropped'
            # self.image_init_path_without_extension = 'BSREM_it30_REF_cropped' # Just for self.testing
        elif (self.config["method"] == "Gong"): # Gong initialization with 60th iteration of MLEM (normally, DIP trained with this image as label...)
            self.f_init = fijii_np(self.subroot + 'Data/initialization/' + 'BSREM_it30_REF_cropped.img',shape=(self.PETImage_shape))
            #self.f_init = fijii_np(self.subroot + 'Data/initialization/' + 'MLEM_it60_REF_cropped.img',shape=(self.PETImage_shape))
            self.image_init_path_without_extension = '1_im_value_cropped' # OPTITR initialization, so firts in MLEM (1 iteration) computation. Not written in Gong, but perhaps not giving any prior information

        #save_img(self.f_init,self.subroot+'Block2/out_cnn/'+ format(self.test)+'/out_' + self.net + '' + format(-1) + self.suffix + '.img') # saving DIP output

        #Loading Ground Truth image to compute metrics
        self.image_gt = fijii_np(self.subroot+'Data/database_v2/' + self.phantom + '/' + self.phantom + '.raw',shape=(self.PETImage_shape))

