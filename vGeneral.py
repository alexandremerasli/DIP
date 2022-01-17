## Python libraries

# Useful
from pathlib import Path
from functools import partial
from ray import tune

# Math
import numpy as np

# Local files to import
from utils.utils_func import *

import abc
class vGeneral(abc.ABC):
    @abc.abstractmethod
    def __init__(self,config,root):
        print("__init__")
        self.test = "not updated"

    def split_config(self,config):
        fixed_config = dict(config)
        hyperparameters_config = dict(config)
        print(self.hyperparameters_list)
        for key in config.keys():
            if key in self.hyperparameters_list:
                fixed_config.pop(key, None)
            else:
                hyperparameters_config.pop(key, None)

        print("fixed_config", fixed_config)
        print("hyperparameters_config", hyperparameters_config)
        return fixed_config, hyperparameters_config

    def initializeGeneralVariables(self, fixed_config, hyperparameters_config, root):
        """General variables"""

        # Initialize some parameters from fixed_config
        self.finetuning = fixed_config["finetuning"]
        print(fixed_config["image"])
        self.phantom = fixed_config["image"]
        self.net = fixed_config["net"]
        self.method = fixed_config["method"]
        self.processing_unit = fixed_config["processing_unit"]
        self.max_iter = fixed_config["max_iter"] # Outer iterations
        self.test = fixed_config["test"] # Label of the experiment

        # Initialize useful variables
        self.subroot = root + '/data/Algo/'  # Directory root
        self.suffix = suffix_func(hyperparameters_config) # self.suffix to make difference between raytune runs (different hyperparameters)

        # Define PET input dimensions according to input data dimensions
        self.PETImage_shape_str = read_input_dim(self.subroot+'Data/database_v2/' + self.phantom + '/' + self.phantom + '.hdr')
        self.PETImage_shape = input_dim_str_to_list(self.PETImage_shape_str)

        # Define ROIs for image0 phantom, otherwise it is already done in the database
        if (self.phantom == "image0"):
            define_ROI_image0(self.PETImage_shape)

        return hyperparameters_config

    def createDirectoryAndConfigFile(self,hyperparameters_config):
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

        # Save this configuration of hyperparameters, and reload with suffix
        np.save(self.subroot + 'Config/' + self.suffix + '.npy', hyperparameters_config) # Save this configuration of hyperparameters, and reload it at the beginning of block 2 thanks to self.suffix (passed in subprocess call arguments

    def runRayTune(self,config,root):

        self.hyperparameters_list = config["hyperparameters"]
        config.pop("hyperparameters", None)

        config_combination = 1
        for i in range(len(config)): # List of hyperparameters keys is still in config dictionary
            config_combination *= len(list(list(config.values())[i].values())[0])

        self.processing_unit = config["processing_unit"]
        resources_per_trial = {"cpu": 1, "gpu": 0}
        if self.processing_unit == 'CPU':
            resources_per_trial = {"cpu": 1, "gpu": 0}
        elif self.processing_unit == 'GPU':
            resources_per_trial = {"cpu": 0, "gpu": 0.1} # "gpu": 1 / config_combination
            #resources_per_trial = {"cpu": 0, "gpu": 1} # "gpu": 1 / config_combination
        elif self.processing_unit == 'both':
            resources_per_trial = {"cpu": 10, "gpu": 1} # not efficient

        #reporter = CLIReporter(
        #    parameter_columns=['lr'],
        #    metric_columns=['mse'])

        # Start tuning of hyperparameters = start each admm computation in parallel
        #try: # resume previous run (if it exists)
        #    anaysis_raytune = tune.run(partial(self.do_everything,root=root), config=config,local_dir = os.getcwd() + '/runs', name=suffix_func(hyperparameters_config) + str(config["max_iter"]), resources_per_trial = resources_per_trial, resume = "ERRORED_ONLY")#, progress_reporter = reporter)
        #except: # do not resume previous run because there is no previous one
        #    anaysis_raytune = tune.run(partial(self.do_everything,root=root), config=config,local_dir = os.getcwd() + '/runs', name=suffix_func(hyperparameters_config) + "_max_iter=" + str(config["max_iter"], resources_per_trial = resources_per_trial)#, progress_reporter = reporter)



        tune.run(partial(self.do_everything,root=root), config=config,local_dir = os.getcwd() + '/runs', resources_per_trial = resources_per_trial)#, progress_reporter = reporter)

    def do_everything(self,config,root):

        # Retrieve fixed parameters and hyperparameters from config dictionnary
        fixed_config, hyperparameters_config = self.split_config(config)
        self.initializeGeneralVariables(fixed_config, hyperparameters_config,root)
        self.initializeSpecific(hyperparameters_config,root)
        self.runComputation(config,hyperparameters_config,root)




""""""""""""""""""""" Useful functions """""""""""""""""""""

def load_input(self,net,PETImage_shape):
    if self.input == "random":
        file_path = (subroot+'Data/initialization/random_input_' + net + '.img')
    elif self.input == "CT":
        file_path = (subroot+'Data/database_v2/' + self.image + '/' + self.image + '_atn.raw') #CT map, but not CT yet, attenuation for now...
    elif self.input == "BSREM":
        file_path = (subroot+'Data/initialization/BSREM_it30_REF_cropped.img') #
    elif self.input == "uniform":
        file_path = (subroot+'Data/initialization/uniform_input_' + net + '.img')
    if (net == 'DD'):
        input_size_DD = int(PETImage_shape[0] / (2**self.d_DD)) # if original Deep Decoder (i.e. only with decoder part)
        PETImage_shape = (self.k_DD,input_size_DD,input_size_DD) # if original Deep Decoder (i.e. only with decoder part)
    elif (net == 'DD_AE'):   
        input_size_DD = PETImage_shape[0] # if auto encoder based on Deep Decoder
        PETImage_shape = (input_size_DD,input_size_DD) # if auto encoder based on Deep Decoder

    im_input = fijii_np(file_path, shape=(PETImage_shape)) # Load input of the DNN (CT image)
    return im_input