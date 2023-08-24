## Python libraries

# Useful
from pathlib import Path
from os import getcwd, makedirs
from os.path import exists, isfile
from functools import partial
from ray import tune
from numpy import dtype, fromfile, argwhere, isnan, zeros, squeeze, ones_like, mean, std, sum, array, column_stack
from numpy import max as max_np
from numpy import min as min_np
from pandas import read_table
from matplotlib.pyplot import imshow, figure, colorbar, savefig, title, gcf, axis, show
from re import split, findall, compile

import abc

from ray.tune import CLIReporter
class ExperimentTerminationReporter(CLIReporter):
    def should_report(self, trials, done=False):
        """Reports only on experiment termination."""
        return False

class vGeneral(abc.ABC):
    @abc.abstractmethod
    def __init__(self,config, *args, **kwargs):
        print("__init__")
        self.experiment = "not updated"

    def split_config(self,config):
        config = dict(config)
        config = dict(config)
        config = dict(config)
        for key in config.keys():
            if key in self.hyperparameters_list:
                config.pop(key, None)
                config.pop(key, None)
            elif key in self.fixed_hyperparameters_list:
                config.pop(key, None)
                config.pop(key, None)
            else:
                config.pop(key, None)
                config.pop(key, None)

        return config

    def initializeGeneralVariables(self,config,root):
        """General variables"""

        # Initialize some parameters from config
        self.finetuning = config["finetuning"]
        self.all_images_DIP = config["all_images_DIP"]
        self.phantom = config["image"]
        self.net = config["net"]
        self.method = config["method"]
        self.processing_unit = config["processing_unit"]
        self.nb_threads = config["nb_threads"]
        self.max_iter = config["max_iter"] # Outer iterations
        self.experiment = config["experiment"] # Label of the experiment
        self.replicate = config["replicates"] # Label of the replicate
        self.penalty = config["penalty"]
        self.castor_foms = config["castor_foms"]
        self.FLTNB = config["FLTNB"]

        self.subroot_data = root + '/data/Algo/' # Directory root
        
        if (config["task"] != "show_metrics_results_already_computed_following_step"):
            # Initialize useful variables
            self.subroot = self.subroot_data + 'debug/'*self.debug + self.phantom + '/'+ 'replicate_' + str(self.replicate) + '/' + self.method + '/' # Directory root
            self.subroot_metrics = self.subroot_data + 'debug/'*self.debug + 'metrics/' + self.phantom + '/'+ 'replicate_' + str(self.replicate) + '/' # Directory root for metrics
            self.suffix = self.suffix_func(config) # self.suffix to make difference between raytune runs (different hyperparameters)
            self.suffix_metrics = self.suffix_func(config,NNEPPS=True) # self.suffix with NNEPPS information
            if ("post_reco" in config["task"] and "post_reco" not in self.suffix):
                self.suffix = "post_reco" + ' ' + self.suffix
                self.suffix_metrics = config["task"] + ' ' + self.suffix_metrics


            # Define PET input dimensions according to input data dimensions
            self.PETImage_shape_str = self.read_input_dim(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.hdr')
            self.PETImage_shape = self.input_dim_str_to_list(self.PETImage_shape_str)

            # Define ROIs for image0 phantom, otherwise it is already done in the database
            if (self.phantom == "image0" or self.phantom == "image2_0" and config["task"] != "show_metrics_results_already_computed"):
                self.define_ROI_image0(self.PETImage_shape,self.subroot_data)
            if (self.phantom == "image2_3D" and config["task"] != "show_metrics_results_already_computed"):
                self.define_ROI_image2_3D(self.PETImage_shape,self.subroot_data)
            if ((self.phantom == "image4_0" or self.phantom == "image400_0" or self.phantom == "image40_0" or self.phantom == "image40_1") and config["task"] != "show_metrics_results_already_computed"):
                self.define_ROI_new_phantom(self.PETImage_shape,self.subroot_data)
        return config

    def createDirectoryAndConfigFile(self,config):
        if (self.method == 'nested' or self.method == 'Gong'):
            Path(self.subroot+'Block1/' + self.suffix + '/before_eq22').mkdir(parents=True, exist_ok=True) # CASToR path
            Path(self.subroot+'Block1/' + self.suffix + '/during_eq22').mkdir(parents=True, exist_ok=True) # CASToR path
            Path(self.subroot+'Block1/' + self.suffix + '/out_eq22').mkdir(parents=True, exist_ok=True) # CASToR path

            Path(self.subroot+'Images/out_final/'+format(self.experiment)+'/').mkdir(parents=True, exist_ok=True) # Output of the framework (Last output of the DIP)

            Path(self.subroot+'Block2/' + self.suffix + '/checkpoint/'+format(self.experiment)+'/').mkdir(parents=True, exist_ok=True)
            Path(self.subroot+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/').mkdir(parents=True, exist_ok=True) # Output of the DIP block every outer iteration
            Path(self.subroot+'Block2/' + self.suffix + '/out_cnn/vae').mkdir(parents=True, exist_ok=True) # Output of the DIP block every outer iteration
            Path(self.subroot+'Block2/' + self.suffix + '/out_cnn/cnn_metrics/'+ format(self.experiment)+'/').mkdir(parents=True, exist_ok=True) # DIP block metrics
            Path(self.subroot+'Block2/' + self.suffix + '/x_label/'+format(self.experiment) + '/').mkdir(parents=True, exist_ok=True) # x corrupted - folder
            Path(self.subroot+'Block2/' + self.suffix + '/mu/'+ format(self.experiment)+'/').mkdir(parents=True, exist_ok=True)

        Path(self.subroot_data + 'Data/initialization').mkdir(parents=True, exist_ok=True)
        Path(self.subroot_data + 'Data/initialization/pytorch/replicate_' + str(self.replicate)).mkdir(parents=True, exist_ok=True)
                
    def runRayTune(self,config,root,task,only_suffix_replicate_file=False):
        # Check parameters incompatibility
        if (task != "show_metrics_results_already_computed_following_step"): # there is no grid_search in config for this task
            self.parametersIncompatibility(config,task)
        # Remove debug and ray keys from config, and ask task
        self.debug = config["debug"]
        self.ray = config["ray"]
        # Remove hyperparameters lists
        self.fixed_hyperparameters_list = config["fixed_hyperparameters"]
        config.pop("fixed_hyperparameters", None)
        self.hyperparameters_list = config["hyperparameters"]
        config.pop("hyperparameters", None)
        config.pop("debug",None)
        # config.pop("ray",None)
        # Convert tensorboard to ray
        config["tensorboard"] = tune.grid_search([config["tensorboard"]])

        config["task"] = {'grid_search': [task]}

        if (self.ray): # Launch raytune
            # Initialize ROIs before ray
            self.initializeBeforeRay(config,root)
            # Launch computation
            tune.run(partial(self.do_everything,root=root,only_suffix_replicate_file = only_suffix_replicate_file), config=config,local_dir = getcwd() + '/runs', progress_reporter = ExperimentTerminationReporter())
        else: # Without raytune
            # Remove grid search if not using ray and choose first element of each config key.
            config = self.removeGridSearch(config,task)            
            # Initialize ROIs
            self.initializeBeforeRay(config,root)            
            # Launch computation
            self.do_everything(config,root,only_suffix_replicate_file = only_suffix_replicate_file)

    def removeGridSearch(self,config,task):
        if (task != "show_metrics_results_already_computed_following_step"):
            for key, value in config.items():
                if key != "hyperparameters" and key != "fixed_hyperparameters" and key != "ray":
                    if (len(value["grid_search"]) == 1 or self.debug):
                        config[key] = value["grid_search"][0]
                    else:
                        raise ValueError("Please put one value for " + key + " in config variable in main.py if ray is deactivated.")
                    
                    if (self.debug):
                        # Set every iteration values to 1 to be quicker
                        if key in ["max_iter","nb_subsets","sub_iter_DIP","nb_inner_iteration","nb_outer_iteration"]:
                            config[key] = 1
                        elif key == "mlem_sequence":
                            config["mlem_sequence"] = False
        else:
            config["task"] = task
        return config

    def initializeBeforeRay(self,config,root):
        self.subroot_data = root + '/data/Algo/' # Directory root
        if (config["ray"]):
            self.phantom = config["image"]["grid_search"][0]
        else:
            self.phantom = config["image"]

        # Define scanner
        if (self.phantom == "image010_3D"):
            self.scanner = "mMR_3D"
        elif (self.phantom == "image012_3D" or self.phantom == "image013_3D"):
            self.scanner = "mCT_3D"
        else:
            self.scanner = "mMR_2D"

        # Define PET input dimensions according to input data dimensions
        self.PETImage_shape_str = self.read_input_dim(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.hdr')
        self.PETImage_shape = self.input_dim_str_to_list(self.PETImage_shape_str)

        # # Loading Ground Truth image to compute metrics
        # self.image_gt = self.fijii_np(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.raw',shape=(self.PETImage_shape),type_im='<f')            

        # Defining ROIs
        self.phantom_ROI = self.get_phantom_ROI(self.phantom)
        if ("3D" not in self.phantom):
            bkg_ROI_path = self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "background_mask" + self.phantom[5:] + '.raw'
            if (isfile(bkg_ROI_path)):
                self.bkg_ROI = self.fijii_np(bkg_ROI_path, shape=(self.PETImage_shape),type_im='<f')
                if (self.phantom == "image4_0" or self.phantom == "image400_0" or self.phantom == "image40_0" or self.phantom == "image40_1"):
                    self.hot_TEP_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "tumor_TEP_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
                    self.hot_TEP_match_square_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "tumor_TEP_match_square_ROI_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
                    self.hot_perfect_match_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "tumor_perfect_match_ROI_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
                    # This ROIs has already been defined, but is computed for the sake of simplicity
                    self.hot_ROI = self.hot_TEP_ROI
                else:
                    self.hot_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "tumor_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
                    # These ROIs do not exist, so put them equal to hot ROI for the sake of simplicity
                    self.hot_TEP_ROI = array(self.hot_ROI)
                    self.hot_TEP_match_square_ROI = array(self.hot_ROI)
                    self.hot_perfect_match_ROI = array(self.hot_ROI)
                self.cold_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "cold_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
                self.cold_inside_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "cold_inside_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
                self.cold_edge_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "cold_edge_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
            else:
                self.initializeGeneralVariables(config,root)

        
    def parametersIncompatibility(self,config,task):
        # Additional variables needing every values in config
        # Number of replicates         
        self.nb_replicates = config["replicates"]['grid_search'][-1]
        #if (task == "show_results_replicates" or task == "show_results"):
        #if (task == "compare_2_methods"):
        #    config["replicates"] = tune.grid_search([0]) # Only put 1 value to avoid running same run several times (only for results with several replicates)

        # By default use ADMMLim in nested, not APGMAP
        if "recoInNested" not in config:
            config["recoInNested"] = tune.grid_search(["ADMMLim"])

        # Do not scale images if network input is uniform of if Gong's method
        if config["input"]['grid_search'] == 'uniform': # Do not standardize or normalize if uniform, otherwise NaNs
            config["scaling"] = "nothing"
        if (len(config["method"]['grid_search']) == 1):
            if config["method"]['grid_search'][0] == 'Gong':
                #print("Goooooooooooooooooooong_normalization_enforced")
                #config["scaling"]['grid_search'] = ["normalization"]
                #config["scaling"]['grid_search'] = ["positive_normalization"]
                print("Gooooooooong")

        # If ADMMLim (not nested), begin with CASToR default value, which is uniform image of 1
        if (len(config["method"]['grid_search']) == 1):
            if config["method"]['grid_search'][0] == 'ADMMLim':
                config["unnested_1st_global_iter"]['grid_search'] = [True]
        
        # Remove NNEPPS=False if True is selected for computation
        if (len(config["NNEPPS"]['grid_search']) > 1 and False in config["NNEPPS"]['grid_search'] and 'results' not in task):
            print("No need for computation without NNEPPS")
            config["NNEPPS"]['grid_search'] = [True]

        # Delete hyperparameters specific to others optimizer 
        if (len(config["method"]['grid_search']) == 1):
            if (config["method"]['grid_search'][0] != "AML" and "APGMAP" not in config["method"]['grid_search'][0] and "APGMAP" not in config["recoInNested"]['grid_search'][0]):
                config.pop("A_AML", None)
            if (config["method"]['grid_search'][0] == 'BSREM' or 'nested' in config["method"]['grid_search'][0] or 'Gong' in config["method"]['grid_search'][0] or 'DIPRecon' in config["method"]['grid_search'][0] or 'APGMAP' in config["method"]['grid_search'][0]):
                config.pop("post_smoothing", None)
            if ((config["method"]['grid_search'][0] != 'ADMMLim' and "nested" not in config["method"]['grid_search'][0]) or "APGMAP" in config["recoInNested"]['grid_search'][0]):
                #config.pop("nb_inner_iteration", None)
                config.pop("alpha", None)
                config.pop("adaptive_parameters", None)
                config.pop("mu_adaptive", None)
                config.pop("tau", None)
                config.pop("tau_max", None)
                config.pop("stoppingCriterionValue", None)
                config.pop("saveSinogramsUAndV", None)
                #config.pop("xi", None)
            elif ((config["method"]['grid_search'][0] == 'ADMMLim' or "nested" in config["method"]['grid_search'][0]) and config["adaptive_parameters"]['grid_search'][0] == "nothing"):
                config.pop("mu_adaptive", None)
                config.pop("tau", None)
                config.pop("tau_max", None)
                config.pop("xi", None)
            if ('ADMMLim' not in config["method"]['grid_search'][0] and "nested" not in config["method"]['grid_search'][0] and "Gong" not in config["method"]['grid_search'][0]  and "DIPRecon" not in config["method"]['grid_search'][0]):
                config.pop("nb_outer_iteration", None)
            if ("nested" not in config["method"]['grid_search'][0] and "Gong" not in config["method"]['grid_search'][0]  and "DIPRecon" not in config["method"]['grid_search'][0] and task != "post_reco"):
                config.pop("lr", None)
                config.pop("sub_iter_DIP", None)
                config.pop("opti_DIP", None)
                config.pop("skip_connections", None)
                config.pop("scaling", None)
                config.pop("input", None)
                config.pop("d_DD", None)
                config.pop("adaptive_parameters_DIP", None)
                config.pop("mu_DIP", None)
                config.pop("tau_DIP", None)
                config.pop("xi_DIP", None)
                #config.pop("unnested_1st_global_iter",None)
            if (config["net"]['grid_search'][0] == "DD"):
                config.pop("skip_connections", None)
            elif (config["net"]['grid_search'][0] != "DD_AE"): # not a Deep Decoder based architecture, so remove k and d
                config.pop("d_DD", None)
                config.pop("k_DD", None)
            if (config["method"]['grid_search'][0] == 'MLEM' or config["method"] == 'OPTITR' or config["method"]['grid_search'][0] == 'OSEM' or config["method"]['grid_search'][0] == 'AML'):
                config.pop("rho", None)
            # Do not use subsets so do not use mlem sequence for ADMM Lim, because of stepsize computation in ADMMLim in CASToR
            if ('ADMMLim' in config["method"]['grid_search'][0] or "nested" in config["method"]['grid_search'][0]):
                config["mlem_sequence"]['grid_search'] = [False]
        else:
            if ('results' not in task):
                raise ValueError("Please do not put several methods at the same time for computation.")
        
        '''
        if (task == "show_results" or task == "show_results_replicates"):
            if (len(config["replicates"]['grid_search']) > 1):
            # Compute once results because for loop over replicates
                config["replicates"]['grid_search'] = [1]
        '''
        
        if (task == "show_results_replicates"):
            # List of beta values
            if (len(config["method"]['grid_search']) == 1):
                if ('ADMMLim' in config["method"]['grid_search'][0]):
                    self.beta_list = config["alpha"]['grid_search']
                    config["alpha"] = tune.grid_search([0]) # Only put 1 value to avoid running same run several times (only for results with several replicates)
                else:
                    if (config["method"]['grid_search'][0] == 'AML'):
                        self.beta_list = config["A_AML"]['grid_search']
                        config["A_AML"] = tune.grid_search([0]) # Only put 1 value to avoid running same run several times (only for results with several replicates)
                    else:                
                        self.beta_list = config["rho"]['grid_search']
                        config["rho"] = tune.grid_search([0]) # Only put 1 value to avoid running same run several times (only for results with several replicates)
            else:
                raise ValueError("There must be only one method to average over replicates")

    def do_everything(self,config,root,only_suffix_replicate_file = False):
        if (not only_suffix_replicate_file):
            # Initialize variables
            self.config = config
            self.root = root
            self.subroot_data = root + '/data/Algo/' # Directory root
            #if (config["task"] != "show_metrics_results_already_computed_following_step"):
            self.initializeGeneralVariables(config,root)
            self.initializeSpecific(config,root)
            # Run task computation
            self.runComputation(config,root)
        if (only_suffix_replicate_file and config["task"] != "show_metrics_results_already_computed_following_step"):
            # Initialize general variables
            self.replicate = config["replicates"] # Label of the replicate
            self.subroot_data = root + '/data/Algo/' # Directory root
            self.suffix_metrics = self.suffix_func(config,NNEPPS=True) # self.suffix with NNEPPS information

            # Store suffix to retrieve all suffixes in main.py for metrics
            text_file = open(self.subroot_data + 'suffixes_for_last_run_' + config["method"] + '.txt', "a")
            text_file.write(self.suffix_metrics + "\n")
            text_file.close()
            # Store replicate to retrieve all replicates in main.py for metrics
            text_file = open(self.subroot_data + 'replicates_for_last_run_' + config["method"] + '.txt', "a")
            text_file.write("replicate_" + str(self.replicate) + "\n")
            text_file.close()


    """"""""""""""""""""" Useful functions """""""""""""""""""""
    def write_hdr(self,subroot,L,subpath,phantom,variable_name='',subroot_output_path='',matrix_type='img',additional_name=''):
        """ write a header for the optimization transfer solution (it's use as CASTOR input)"""
        if (len(L) == 1):
            i = L[0]
            if variable_name != '':
                ref_numbers = format(i) + '_' + variable_name
            else:
                ref_numbers = format(i)
        elif (len(L) == 2):
            i = L[0]
            k = L[1]
            if variable_name != '':
                ref_numbers = format(i) + '_' + format(k) + '_' + variable_name
            else:
                ref_numbers = format(i)
        elif (len(L) == 3):
            i = L[0]
            k = L[1]
            inner_it = L[2]
            if variable_name != '':
                ref_numbers = format(i) + '_' + format(k) + '_' + format(inner_it) + '_' + variable_name
            else:
                ref_numbers = format(i)
        ref_numbers = additional_name + ref_numbers
        filename = subroot_output_path + '/'+ subpath + '/' + ref_numbers +'.hdr'
        with open(self.subroot_data + 'Data/MLEM_reco_for_init_hdr/' + phantom + '/' + phantom + '_it1.hdr') as f:
            with open(filename, "w") as f1:
                for line in f:
                    if line.strip() == ('!name of data file := ' + phantom + '_it1.img'):
                        f1.write('!name of data file := '+ ref_numbers +'.img')
                        f1.write('\n') 
                    elif line.strip() == ('patient name := ' + phantom + '_it1'):
                        f1.write('patient name := ' + ref_numbers)
                        f1.write('\n')
                    elif line.startswith("!number format"):
                        f1.write('!number format := ' + (self.FLTNB == "float")*"short" + (self.FLTNB == "double")*"long" +  ' float')
                        f1.write('\n')
                    elif line.startswith("!number of bytes per pixel"):
                        f1.write('!number of bytes per pixel := ' + (self.FLTNB == "float")*"4" + (self.FLTNB == "double")*"8")
                        f1.write('\n')
                    else:
                        if (matrix_type == 'sino'): # There are 68516=2447*28 events, but not really useful for computation
                            if line.strip().startswith('!matrix size [1]'):
                                f1.write('matrix size [1] := 2447')
                                f1.write('\n') 
                            elif line.strip().startswith('!matrix size [2]'):
                                f1.write('matrix size [2] := 28')
                                f1.write('\n')
                            else:
                                f1.write(line) 
                        else:
                            f1.write(line)

    def suffix_func(self,config,NNEPPS=False,hyperparameters_list=False):
        config_copy = dict(config)
        if (NNEPPS==False):
            config_copy.pop('NNEPPS',None)
        if config["method"] == "ADMMLim":
            config_copy.pop('nb_outer_iteration',None)
        elif ("post_reco" in config_copy["task"]):
            config_copy.pop("sub_iter_DIP", None)
        suffix = "config"
        if hyperparameters_list == False:
            for key, value in config_copy.items():
                if key in self.hyperparameters_list:
                    suffix +=  "_" + key[:min(len(key),5)] + "=" + str(value)
        else:   
            #'''
            #hyperparameters_list = ["lr", "optimizer"]
            for key, value in config_copy.items():
                if key in hyperparameters_list:
                    suffix +=  "_" + key[:min(len(key),5)] + "=" + str(value)
            #'''
        return suffix

    def read_input_dim(self,file_path):
        # Read CASToR header file to retrieve image dimension """
        try:
            with open(file_path) as f:
                for line in f:
                    if 'matrix size [1]' in line.strip():
                        dim1 = [int(s) for s in line.split() if s.isdigit()][-1]
                    if 'matrix size [2]' in line.strip():
                        dim2 = [int(s) for s in line.split() if s.isdigit()][-1]
                    if 'matrix size [3]' in line.strip():
                        dim3 = [int(s) for s in line.split() if s.isdigit()][-1]
        except:
            raise ValueError("Please put the header file from CASToR with name of phantom")
        # Create variables to store dimensions
        PETImage_shape = (dim1,dim2,dim3)
        # if (self.scanner == "mMR_3D"):
        #     PETImage_shape = (int(dim1/2),int(dim2/2),dim3)
        PETImage_shape_str = str(dim1) + ','+ str(dim2) + ',' + str(dim3)
        print('image shape :', PETImage_shape)
        return PETImage_shape_str

    def input_dim_str_to_list(self,PETImage_shape_str):
        return [int(e.strip()) for e in PETImage_shape_str.split(',')]#[:-1]

    def fijii_np(self,path,shape,type_im=None):
        """"Transforming raw data to numpy array"""
        if (type_im is None):
            if (self.FLTNB == 'float'):
                type_im = '<f'
            elif (self.FLTNB == 'double'):
                type_im = '<d'

        attempts = 0

        while attempts < 1000:
            attempts += 1
            try:
                type_im = ('<f')*(type_im=='<f') + ('<d')*(type_im=='<d')
                file_path=(path)
                dtype_np = dtype(type_im)
                with open(file_path, 'rb') as fid:
                    data = fromfile(fid,dtype_np)
                    if (1 in shape): # 2D
                        #shape = (shape[0],shape[1])
                        image = data.reshape(shape)
                    else: # 3D
                        image = data.reshape(shape[::-1])
                attempts = 1000
                break
            except:
                # fid.close()
                type_im = ('<f')*(type_im=='<d') + ('<d')*(type_im=='<f')
                file_path=(path)
                dtype_np = dtype(type_im)
                with open(file_path, 'rb') as fid:
                    data = fromfile(fid,dtype_np)
                    if (1 in shape): # 2D
                        #shape = (shape[0],shape[1])
                        try:
                            image = data.reshape(shape)
                        except Exception as e:
                            # print(data.shape)
                            # print(type_im)
                            # print(dtype_np)
                            # print(fid)
                            # '''
                            # import numpy as np
                            # data = fromfile(fid,dtype('<f'))
                            # np.save('data' + str(self.replicate) + '_' + str(attempts) + '_f.npy', data)
                            # '''
                            # print('Failed: '+ str(e) + '_' + str(attempts))
                            pass
                    else: # 3D
                        image = data.reshape(shape[::-1])
                
                fid.close()
            '''
            image = data.reshape(shape)
            #image = transpose(image,axes=(1,2,0)) # imshow ok
            #image = transpose(image,axes=(1,0,2)) # imshow ok
            #image = transpose(image,axes=(0,1,2)) # imshow ok
            #image = transpose(image,axes=(0,2,1)) # imshow ok
            #image = transpose(image,axes=(2,0,1)) # imshow ok
            #image = transpose(image,axes=(2,1,0)) # imshow ok
            '''
            
        #'''
        #image = data.reshape(shape)
        '''
        try:
            print(image[0,0])
        except Exception as e:
            print('exception image: '+ str(e))
        '''
        # print("read from ", path)
        return image

    def norm_imag(self,img):
        print("nooooooooorm")
        """ Normalization of input - output [0..1] and the normalization value for each slide"""
        if (max_np(img) - min_np(img)) != 0:
            return (img - min_np(img)) / (max_np(img) - min_np(img)), min_np(img), max_np(img)
        else:
            return img, min_np(img), max_np(img)

    def norm_init_imag(self,img,param1,param2):
        print("nooooooooorm with other image")
        if (param1-param2) != 0:
            return (img - param1) / (param2 - param1), param1, param2
        else:
            return img, param1, param2

    def denorm_imag(self,image, mini, maxi):
        """ Denormalization of input - output [0..1] and the normalization value for each slide"""
        image_np = image.detach().numpy()
        return self.denorm_numpy_imag(image_np, mini, maxi)

    def denorm_numpy_imag(self,img, mini, maxi):
        if (maxi - mini) != 0:
            return img * (maxi - mini) + mini
        else:
            return img


    def norm_positive_imag(self,img):
        """ Positive normalization of input - output [0..1] and the normalization value for each slide"""
        if (max_np(img) - min_np(img)) != 0:
            print(max_np(img))
            print(min_np(img))
            return img / max_np(img), 0, max_np(img)
        else:
            return img, 0, max_np(img)

    def denorm_positive_imag(self,image, mini, maxi):
        """ Positive normalization of input - output [0..1] and the normalization value for each slide"""
        image_np = image.detach().numpy()
        return self.denorm_numpy_imag(image_np, mini, maxi)

    def denorm_numpy_positive_imag(self, img, mini, maxi):
        if (maxi - mini) != 0:
            return img * maxi 
        else:
            return img

    def stand_imag(self,image_corrupt):
        print("staaaaaaaaaaand")
        """ Standardization of input - output with mean 0 and std 1 for each slide"""
        mean_im=mean(image_corrupt)
        std_im=std(image_corrupt)
        image_center = image_corrupt - mean_im
        if (std_im == 0.):
            raise ValueError("std 0")
        image_corrupt_std = image_center / std_im
        return image_corrupt_std,mean_im,std_im

    def destand_numpy_imag(self,image, mean_im, std_im):
        """ Destandardization of input - output with mean 0 and std 1 for each slide"""
        return image * std_im + mean_im

    def destand_imag(self,image, mean_im, std_im):
        image_np = image.detach().numpy()
        return self.destand_numpy_imag(image_np, mean_im, std_im)

    def rescale_imag(self,image_corrupt, scaling,param1=1e+40,param2=1e+40):
        """ Scaling of input """
        if (scaling == 'standardization'):
            return self.stand_imag(image_corrupt)
        elif (scaling == 'normalization'):
            return self.norm_imag(image_corrupt)
        elif (scaling == 'normalization_init'):
            return self.norm_init_imag(image_corrupt,param1,param2)
        elif (scaling == 'positive_normalization'):
            return self.norm_positive_imag(image_corrupt)
        else: # No scaling required
            return image_corrupt, 0, 0

    def descale_imag(self,image, param_scale1, param_scale2, scaling='standardization'):
        """ Descaling of input """
        try:
            image_np = image.detach().numpy()
        except:
            image_np = image
        if (scaling == 'standardization'):
            return self.destand_numpy_imag(image_np, param_scale1, param_scale2)
        elif (scaling == 'normalization'):
            return self.denorm_numpy_imag(image_np, param_scale1, param_scale2)
        elif (scaling == 'normalization_2'):
            return self.denorm_numpy_imag(image_np, param_scale1, param_scale2)
        elif (scaling == 'positive_normalization'):
            return self.denorm_numpy_positive_imag(image_np, param_scale1, param_scale2)
        else: # No scaling required
            return image_np

    def save_img(self,img,name):
        fp=open(name,'wb')
        img.tofile(fp)
        #print('Succesfully save in:', name)

    def find_nan(self,image):
        """ find NaN values on the image"""
        idx = argwhere(isnan(image))
        print('index with NaN value:',len(idx))
        for i in range(len(idx)):
            image[idx[i,0],idx[i,1]] = 0
        print('index with NaN value:',len(argwhere(isnan(image))))
        return image

    def points_in_circle(self,center_y,center_x,radius,PETImage_shape,inner_circle=True): # x and y are inverted in an array compared to coordinates
        liste = [] 

        center_x += int(PETImage_shape[0]/2)
        center_y += int(PETImage_shape[1]/2)
        for x in range(0,PETImage_shape[0]):
            for y in range(0,PETImage_shape[1]):
                if (x+0.5-center_x)**2 + (y+0.5-center_y)**2 <= radius**2:
                    liste.append((x,y))

        return liste

    def define_ROI_image0(self,PETImage_shape,subroot):
        phantom_ROI = self.points_in_circle(0/4,0/4,150/4,PETImage_shape)
        cold_ROI = self.points_in_circle(-40/4,-40/4,40/4-1,PETImage_shape)
        hot_ROI = self.points_in_circle(50/4,10/4,20/4-1,PETImage_shape)
            
        cold_ROI_bkg = self.points_in_circle(-40/4,-40/4,40/4+1,PETImage_shape)
        hot_ROI_bkg = self.points_in_circle(50/4,10/4,20/4+1,PETImage_shape)
        phantom_ROI_bkg = self.points_in_circle(0/4,0/4,150/4-1,PETImage_shape)
        bkg_ROI = list(set(phantom_ROI_bkg) - set(cold_ROI_bkg) - set(hot_ROI_bkg))

        cold_mask = zeros(PETImage_shape, dtype='<f')
        tumor_mask = zeros(PETImage_shape, dtype='<f')
        phantom_mask = zeros(PETImage_shape, dtype='<f')
        bkg_mask = zeros(PETImage_shape, dtype='<f')

        ROI_list = [cold_ROI, hot_ROI, phantom_ROI, bkg_ROI]
        mask_list = [cold_mask, tumor_mask, phantom_mask, bkg_mask]
        for i in range(len(ROI_list)):
            ROI = ROI_list[i]
            mask = mask_list[i]
            for couple in ROI:
                #mask[int(couple[0] - PETImage_shape[0]/2)][int(couple[1] - PETImage_shape[1]/2)] = 1
                mask[couple] = 1

        # Storing into file instead of defining them at each metrics computation
        self.save_img(cold_mask, subroot+'Data/database_v2/' + "image0" + '/' + "cold_mask0" + '.raw')
        self.save_img(tumor_mask, subroot+'Data/database_v2/' + "image0" + '/' + "tumor_mask0" + '.raw')
        self.save_img(phantom_mask, subroot+'Data/database_v2/' + "image0" + '/' + "phantom_mask0" + '.raw')
        self.save_img(bkg_mask, subroot+'Data/database_v2/' + "image0" + '/' + "background_mask0" + '.raw')

        '''
        # Storing into file instead of defining them at each metrics computation
        self.save_img(cold_mask, subroot+'Data/database_v2/' + "image2_0" + '/' + "cold_mask2_0" + '.raw')
        self.save_img(tumor_mask, subroot+'Data/database_v2/' + "image2_0" + '/' + "tumor_mask2_0" + '.raw')
        self.save_img(phantom_mask, subroot+'Data/database_v2/' + "image2_0" + '/' + "phantom_mask2_0" + '.raw')
        self.save_img(bkg_mask, subroot+'Data/database_v2/' + "image2_0" + '/' + "background_mask2_0" + '.raw')
        '''

    def define_ROI_image2_3D(self,PETImage_shape,subroot):

        phantom_ROI = self.points_in_circle(0/4,0/4,150/4,PETImage_shape)
        cold_ROI = self.points_in_circle(-40/4,-40/4,40/4-1,PETImage_shape)
        hot_ROI = self.points_in_circle(50/4,10/4,20/4-1,PETImage_shape)
            
        cold_ROI_bkg = self.points_in_circle(-40/4,-40/4,40/4+1,PETImage_shape)
        hot_ROI_bkg = self.points_in_circle(50/4,10/4,20/4+1,PETImage_shape)
        phantom_ROI_bkg = self.points_in_circle(0/4,0/4,150/4-1,PETImage_shape)
        bkg_ROI = list(set(phantom_ROI_bkg) - set(cold_ROI_bkg) - set(hot_ROI_bkg))

        # Reverse shape for 3D
        PETImage_shape = PETImage_shape[::-1]

        cold_mask = zeros(PETImage_shape, dtype='<f')
        tumor_mask = zeros(PETImage_shape, dtype='<f')
        phantom_mask = zeros(PETImage_shape, dtype='<f')
        bkg_mask = zeros(PETImage_shape, dtype='<f')

        ROI_list = [cold_ROI, hot_ROI, phantom_ROI, bkg_ROI]
        mask_list = [cold_mask, tumor_mask, phantom_mask, bkg_mask]
        for i in range(len(ROI_list)):
            ROI = ROI_list[i]
            mask = mask_list[i]
            for z in range(mask.shape[0]):
                for couple in ROI:
                    mask[z,couple[0],couple[1]] = 1


        # Storing into file instead of defining them at each metrics computation
        self.save_img(cold_mask, subroot+'Data/database_v2/' + "image2_3D" + '/' + "cold_mask2_3D" + '.raw')
        self.save_img(tumor_mask, subroot+'Data/database_v2/' + "image2_3D" + '/' + "tumor_mask2_3D" + '.raw')
        self.save_img(phantom_mask, subroot+'Data/database_v2/' + "image2_3D" + '/' + "phantom_mask2_3D" + '.raw')
        self.save_img(bkg_mask, subroot+'Data/database_v2/' + "image2_3D" + '/' + "background_mask2_3D" + '.raw')

    def define_ROI_new_phantom(self,PETImage_shape,subroot):

        # Define external radius to not take into account in ROI definition
        remove_external_radius = 1 # MIC abstract 2022, 2023
        # remove_external_radius = 3 # ROIs further away from true edges

        # Define ROIs
        phantom_ROI = self.points_in_circle(0/4,0/4,150/4,PETImage_shape)
        cold_ROI = self.points_in_circle(-40/4,-40/4,40/4-1,PETImage_shape)
        cold_inside_ROI = self.points_in_circle(-40/4,-40/4,40/4-3,PETImage_shape)
        cold_edge_ROI = list(set(cold_ROI) - set(cold_inside_ROI))
        hot_TEP_ROI = self.points_in_circle(50/4,10/4,20/4-1,PETImage_shape)
        hot_TEP_match_square_ROI = self.points_in_circle(-20/4,70/4,20/4-1,PETImage_shape)
        hot_perfect_match_ROI = self.points_in_circle(50/4,90/4,20/4-1,PETImage_shape)
            
        cold_ROI_bkg = self.points_in_circle(-40/4,-40/4,40/4+1,PETImage_shape)
        hot_TEP_ROI_bkg = self.points_in_circle(50/4,10/4,20/4+1,PETImage_shape)
        hot_TEP_match_square_ROI_bkg = self.points_in_circle(-20/4,70/4,20/4+1,PETImage_shape)
        hot_perfect_match_ROI_bkg = self.points_in_circle(50/4,90/4,20/4+1,PETImage_shape)
        phantom_ROI_bkg = self.points_in_circle(0/4,0/4,150/4-1,PETImage_shape)
        bkg_ROI = list(set(phantom_ROI_bkg) - set(cold_ROI_bkg) - set(hot_TEP_ROI_bkg) - set(hot_TEP_match_square_ROI_bkg) - set(hot_perfect_match_ROI_bkg))

        cold_mask = zeros(PETImage_shape, dtype='<f')
        cold_inside_mask = zeros(PETImage_shape, dtype='<f')
        cold_edge_mask = zeros(PETImage_shape, dtype='<f')
        tumor_TEP_mask = zeros(PETImage_shape, dtype='<f')
        tumor_TEP_match_square_ROI_mask = zeros(PETImage_shape, dtype='<f')
        tumor_perfect_match_ROI_mask = zeros(PETImage_shape, dtype='<f')
        phantom_mask = zeros(PETImage_shape, dtype='<f')
        bkg_mask = zeros(PETImage_shape, dtype='<f')

        ROI_list = [cold_ROI, cold_inside_ROI, cold_edge_ROI, hot_TEP_ROI, hot_TEP_match_square_ROI, hot_perfect_match_ROI, phantom_ROI, bkg_ROI]
        mask_list = [cold_mask, cold_inside_mask, cold_edge_mask, tumor_TEP_mask, tumor_TEP_match_square_ROI_mask, tumor_perfect_match_ROI_mask, phantom_mask, bkg_mask]
        for i in range(len(ROI_list)):
            ROI = ROI_list[i]
            mask = mask_list[i]
            for couple in ROI:
                mask[couple] = 1

        # Storing into file instead of defining them at each metrics computation
        self.save_img(cold_mask, subroot+'Data/database_v2/' + self.phantom + '/' + "cold_mask" + self.phantom[5:] + '.raw')
        self.save_img(cold_inside_mask, subroot+'Data/database_v2/' + self.phantom + '/' + "cold_inside_mask" + self.phantom[5:] + '.raw')
        self.save_img(cold_edge_mask, subroot+'Data/database_v2/' + self.phantom + '/' + "cold_edge_mask" + self.phantom[5:] + '.raw')
        self.save_img(tumor_TEP_mask, subroot+'Data/database_v2/' + self.phantom + '/' + "tumor_TEP_mask" + self.phantom[5:] + '.raw')
        self.save_img(tumor_TEP_match_square_ROI_mask, subroot+'Data/database_v2/' + self.phantom + '/' + "tumor_TEP_match_square_ROI_mask" + self.phantom[5:] + '.raw')
        self.save_img(tumor_perfect_match_ROI_mask, subroot+'Data/database_v2/' + self.phantom + '/' + "tumor_perfect_match_ROI_mask" + self.phantom[5:] + '.raw')
        self.save_img(phantom_mask, subroot+'Data/database_v2/' + self.phantom + '/' + "phantom_mask" + self.phantom[5:] + '.raw')
        self.save_img(bkg_mask, subroot+'Data/database_v2/' + self.phantom + '/' + "background_mask" + self.phantom[5:] + '.raw')

    def write_image_tensorboard(self,writer,image,name,suffix,image_gt,i=0,full_contrast=False):
        # Creating matplotlib figure with colorbar
        figure()
        if (len(squeeze(image).shape) != 2):
            print('image is ' + str(len(image.shape)) + 'D, plotting only 2D slice')
            image = image[int(image.shape[0] / 2.),:,:]
            #image = image[:,:,int(image.shape[0] / 2.)]
        MIC_show = False
        if (MIC_show):
            nb_crop = 10
            image = array(image[nb_crop:len(image) - nb_crop,nb_crop:len(image) - nb_crop])
        if ("DIP input" in name):
            cmap_ = "gray"
            # cmap_ = "gray_r" # already reversed in iResults.py in writeBeginningImages()
        else:
            cmap_ = "gray_r"
        if (full_contrast):
            imshow(image, cmap=cmap_,vmin=min_np(image),vmax=max_np(image)) # Showing each image with maximum contrast and white is zero (gray_r) 
        else:
            imshow(image, cmap=cmap_,vmin=min_np(image_gt),vmax=1.25*max_np(image_gt)) # Showing all images with same contrast and white is zero (gray_r)
        # colorbar()
        axis('off')
        #show()

        # if (isnan(sum(image))):
        #     raise ValueError("NaNs detected in image. Stopping computation (" + "replicate_" + str(i) + "/" + suffix + ")")

        # Saving this figure locally
        Path(self.subroot + 'Images/tmp/' + suffix).mkdir(parents=True, exist_ok=True)
        #system('rm -rf' + self.subroot + 'Images/tmp/' + suffix + '/*')
        from textwrap import wrap
        if (MIC_show):
            import matplotlib.pyplot as plt
            plt.tight_layout()
            plt.rcParams['figure.figsize'] = 10, 10
        else:
            colorbar()
            axis('off')
        savefig(self.subroot + 'Images/tmp/' + suffix + '/' + name + '_' + str(i) + '.png')

        # added line for small title of interest
        suffix = self.suffix_func(self.config,hyperparameters_list = ["lr", "opti_DIP"])
        
        wrapped_title = "\n".join(wrap(suffix, 80))

        #title(wrapped_title + "\n" + name,fontsize=8)
        #title(wrapped_title,fontsize=10)
        title(wrapped_title,fontsize=16)
        # Adding this figure to tensorboard
        writer.add_figure(name,gcf(),global_step=i,close=True)# for videos, using slider to change image with global_step

    def castor_common_command_line(self, subroot, PETImage_shape_str, phantom, replicates, post_smoothing=0):
        executable = 'castor-recon'
        # if (self.nb_replicates == 1):
        #     header_file = ' -df ' + subroot + 'Data/database_v2/' + phantom + '/data' + phantom[5:] + '/data' + phantom[5:] + '.cdh' # PET data path
        # else:
        #     header_file = ' -df ' + subroot + 'Data/database_v2/' + phantom + '/data' + phantom[5:] + '_' + str(replicates) + '/data' + phantom[5:] + '_' + str(replicates) + '.cdh' # PET data path
        header_file = ' -df ' + subroot + 'Data/database_v2/' + phantom + '/data' + phantom[5:] + '_' + str(replicates) + '/data' + phantom[5:] + '_' + str(replicates) + '.cdh' # PET data pat
        dim = ' -dim ' + PETImage_shape_str
        if (self.scanner != "mMR_3D"):
            if (self.phantom != "image50_0"):
                vox = ' -vox 4,4,4'
            else:
                vox = ' -vox 2,2,2'
        else:
            vox = ' -vox 1.04313,1.04313,2.03125'
            vox = ' -vox 2.08626,2.08626,2.03125'
        vb = ' -vb 3'
        th = ' -th ' + str(self.nb_threads) # must be set to 1 for ADMMLim, as multithreading does not work for now with ADMMLim optimizer
        proj = ' -proj incrementalSiddon'
        if ("1" in PETImage_shape_str.split(',')): # 2D
            psf = ' -conv gaussian,4,1,3.5::psf'
        else: # 3D
            if (self.scanner == "mMR_3D"):
                psf = ' -conv gaussian,4.5,4.5,3.5::psf' # isotropic psf in simulated phantoms
            else:
                psf = ' -conv gaussian,4,4,3.5::psf' # isotropic psf in simulated phantoms

        if (post_smoothing != 0):
            if ("1" in PETImage_shape_str.split(',')): # 2D
                conv = ' -conv gaussian,' + str(post_smoothing) + ',1,3.5::post'
            else: # 3D
                conv = ' -conv gaussian,' + str(post_smoothing) + ',' + str(post_smoothing) + ',3.5::post' # isotropic post smoothing
        else:
            conv = ''
        # Computing likelihood
        if (self.castor_foms):
            opti_like = ' -opti-fom'
        else:
            opti_like = ''

        return executable + dim + vox + header_file + vb + th + proj + opti_like + psf + conv

    def castor_opti_and_penalty(self, method, penalty, rho, i=None, unnested_1st_global_iter=None):
        if (method == 'MLEM'):
            opti = ' -opti ' + method
            pnlt = ''
            penaltyStrength = ''
        if (method == 'OPTITR'):
            opti = ' -opti ' + method
            pnlt = ''
            penaltyStrength = ''
        elif (method == 'OSEM'):
            opti = ' -opti ' + 'MLEM'
            pnlt = ''
            penaltyStrength = ''
        elif (method == 'AML'):
            opti = ' -opti ' + method + ',1,1e-10,' + str(self.A_AML)
            pnlt = ''
            penaltyStrength = ''
        elif (method == 'APGMAP'):
            #opti = ' -opti ' + "APPGML" + ',1,1e-10,0.01,-1,' + str(self.A_AML) + ',0' # Multimodal image is only used by APPGML
            opti = ' -opti ' + "APPGML" + ':' + self.subroot + '/' + self.suffix  + '/' + 'APPGML.conf'
            pnlt = ' -pnlt ' + penalty + ':' + self.subroot_data + method + '_MRF.conf'
            penaltyStrength = ' -pnlt-beta ' + str(self.beta)
        elif (method == 'BSREM'):
            opti = ' -opti ' + method + ':' + self.subroot_data + method + '.conf'
            pnlt = ' -pnlt ' + penalty + ':' + self.subroot_data + method + '_MRF.conf'
            penaltyStrength = ' -pnlt-beta ' + str(self.beta)
        elif ('nested' in method or 'ADMMLim' in method):
            if (self.recoInNested == "ADMMLim"):
                opti = ' -opti ' + 'ADMMLim' + ',' + str(self.alpha) + ',' + str(self.castor_adaptive_to_int(self.adaptive_parameters)) + ',' + str(self.mu_adaptive) + ',' + str(self.tau) + ',' + str(self.xi) + ',' + str(self.tau_max) + ',' + str(self.stoppingCriterionValue) + ',' + str(self.saveSinogramsUAndV)
                if ('nested' in method):
                    # if ((i==0 and unnested_1st_global_iter) or (i==-1 and not unnested_1st_global_iter)): # For first iteration, put rho to zero
                    if ((i==-1 and not unnested_1st_global_iter)): # For first iteration, put rho to zero
                        rho = 0
                        #self.rho = 0
                    method = 'ADMMLim' + method[6:]
                    #pnlt = ' -pnlt QUAD' # Multimodal image is only used by quadratic penalty
                    pnlt = ' -pnlt ' + "QUAD" + ':' + self.subroot + 'Block1/' + self.suffix  + '/' + 'QUAD.conf'
                elif ('ADMMLim' in method):
                    pnlt = ' -pnlt ' + penalty
                    if penalty == "MRF":
                        pnlt += ':' + self.subroot_data + method + '_MRF.conf'
                penaltyStrength = ' -pnlt-beta ' + str(rho)
            elif (self.recoInNested == "APGMAP"):
                if ((i==0 and unnested_1st_global_iter) or (i==-1 and not unnested_1st_global_iter)): # For first iteration, put rho to zero
                    rho = 0
                    #self.rho = 0
                #opti = ' -opti APPGML' + ',1,1e-10,0.01,-1,' + str(self.A_AML) + ',-1' # Do not use a multimodal image for APPGML, so let default multimodal index (-1)
                opti = ' -opti ' + "APPGML" + ':' + self.subroot + 'Block1/' + self.suffix  + '/' + 'APPGML.conf'
                #pnlt = ' -pnlt QUAD,0' # Multimodal image is used only for quadratic penalty, so put multimodal index to 0
                pnlt = ' -pnlt ' + "QUAD" + ':' + self.subroot + 'Block1/' + self.suffix  + '/' + 'QUAD.conf'
                penaltyStrength = ' -pnlt-beta ' + str(rho)
            
            # For all optimizers, remove penalty if rho == 0
            if (rho == 0):
                pnlt = ''
                penaltyStrength = ''
        elif (method == 'Gong'):
            if ((i==0 and unnested_1st_global_iter) or (i==-1 and not unnested_1st_global_iter)): # For first iteration, put rho to zero
                rho = 0
                #self.rho = 0
            opti = ' -opti OPTITR'
            pnlt = ' -pnlt ' + "QUAD" + ':' + self.subroot + 'Block1/' + self.suffix  + '/' + 'QUAD.conf'            
            penaltyStrength = ' -pnlt-beta ' + str(rho)
        
        # For all optimizers, remove penalty if rho == 0
        if (rho == 0):
            pnlt = ''
            penaltyStrength = ''
        
        return opti + pnlt + penaltyStrength

    def castor_adaptive_to_int(self,adaptive_parameters):
        if (adaptive_parameters == "nothing"): # not adaptive
            return 0
        if (adaptive_parameters == "alpha"): # only adative alpha
            return 1
        if (adaptive_parameters == "both"): # both adaptive alpha and tau
            return 2

    def get_phantom_ROI(self,image='image0'):
        # Select only phantom ROI, not whole reconstructed image
        path_phantom_ROI = self.subroot_data+'Data/database_v2/' + image + '/' + "phantom_mask" + str(image[5:]) + '.raw'
        my_file = Path(path_phantom_ROI)
        if (my_file.is_file()):
            phantom_ROI = self.fijii_np(path_phantom_ROI, shape=(self.PETImage_shape),type_im='<f')
        else:
            print("No phantom file for this phantom")
            # Loading Ground Truth image to compute metrics
            try:
                image_gt = self.fijii_np(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.raw',shape=(self.PETImage_shape),type_im='<f')
            except:
                raise ValueError("Please put the header file from CASToR with name of phantom")
            phantom_ROI = ones_like(image_gt)
            #raise ValueError("No phantom file for this phantom")
            #phantom_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + image + '/' + "background_mask" + image[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
            
        return phantom_ROI
    
    def mkdir(self,path):
        # check path exists or no before saving files
        folder = exists(path)

        if not folder:
            makedirs(path)

        return path


    def atoi(self,text):
        return int(text) if text.isdigit() else text

    def natural_keys(self,text):
        # print(split(r'(\d+)', text))
        return [ self.atoi(c) for c in split(r'(\d+)', text) ] # APGMAP final curves + resume computation
        #return [ self.atoi(c) for c in split(r'(\+|-)\d+(\.\d+)?', text) ] # ADMMLim final curves
    
    def natural_keys_ADMMLim(self,text): # Sort by scientific or float numbers
        #return [ self.atoi(c) for c in split(r'(\d+)', text) ] # APGMAP final curves + resume computation
        match_number = compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
        final_list = [float(x) for x in findall(match_number, text)] # Extract scientific of float numbers in string
        return final_list # ADMMLim final curves
        
    def has_numbers(self,inputString):
        return any(char.isdigit() for char in inputString)

    def ImageAndItToResumeComputation(self,sorted_files, it, folder_sub_path):
        sorted_files.sort(key=self.natural_keys)
        last_file = sorted_files[-1]
        if ("=" in last_file): # post reco mode
            last_file = last_file[-10:]
            last_file = "it_" + last_file.split("=",1)[1]
        last_iter = int(findall(r'(\w+?)(\d+)', last_file.split('.')[0])[0][-1])
        initialimage = ' -img ' + folder_sub_path + '/' + last_file
        it += ' -skip-it ' + str(last_iter)
        
        return initialimage, it, last_iter

    def linear_regression(self, x, y):
        x_mean = x.mean()
        y_mean = y.mean()
        
        B1_num = ((x - x_mean) * (y - y_mean)).sum()
        B1_den = ((x - x_mean)**2).sum()
        B1 = B1_num / B1_den
                        
        return B1

    def defineTotalNbIter_beta_rho(self,method,config,task,stopping_criterion=True):
        if (method == 'ADMMLim'):
            try:
                self.path_stopping_criterion = self.subroot + self.suffix + '/' + format(0) + '_adaptive_stopping_criteria.log'
                with open(self.path_stopping_criterion) as f:
                    first_line = f.readline() # Read first line to get second one
                    self.total_nb_iter = min(int(f.readline().rstrip()) - self.i_init, config["nb_outer_iteration"] - self.i_init + 1)
                    #self.total_nb_iter = int(self.total_nb_iter / self.i_init) # if 1 out of i_init iterations was saved
                    #self.total_nb_iter = config["nb_outer_iteration"] - self.i_init + 1 # Override value
            except:
                self.total_nb_iter = config["nb_outer_iteration"] - self.i_init + 1
                #self.total_nb_iter = int(self.total_nb_iter / self.i_init) # if 1 out of i_init iterations was saved
            self.beta = config["alpha"]
        elif ('nested' in method or 'Gong' in method or 'DIPRecon' in method):
            if ('post_reco' in task):
                self.total_nb_iter = config["sub_iter_DIP"]
            else:
                try:
                    if (stopping_criterion):
                        self.path_stopping_criterion = self.subroot + 'Block2/' + self.suffix + '/' + 'IR_stopping_criteria.log'
                        with open(self.path_stopping_criterion) as f:
                            first_line = f.readline() # Read first line to get second one
                            #self.total_nb_iter = min(int(f.readline().rstrip()) - self.i_init, config["nb_outer_iteration"] - self.i_init + 1)
                            self.total_nb_iter = int(f.readline().rstrip()) - self.i_init - 1
                            #self.total_nb_iter = int(self.total_nb_iter / self.i_init) # if 1 out of i_init iterations was saved
                            #self.total_nb_iter = config["nb_outer_iteration"] - self.i_init + 1 # Override value
                    else:
                        self.total_nb_iter = config["max_iter"]    
                except:
                    self.total_nb_iter = config["max_iter"]
        else:
            self.total_nb_iter = self.max_iter

            if (config["method"] == 'AML'):
                self.beta = config["A_AML"]
            if (config["method"] == 'BSREM' or 'nested' in config["method"] or 'Gong' in config["method"] or 'DIPRecon' in config["method"] or 'APGMAP' in config["method"]):
                self.rho = config["rho"]
                self.beta = self.rho

    def modify_input_line_edge(self,config):
        addon = "line_edge" # mu_DIP = 10
        addon = "high_pixel" # mu_DIP = 20
        # addon = "reduce_cold_value_MR" # mu_DIP = 3
        addon = "remove_cold" # mu_DIP = 4
        addon = "nothing"
        # addon = "remove_ellipse_MR"
        if (addon == "line_edge"):
            phantom_ROI = self.points_in_circle_edge(0/4,0/4,150/4,self.PETImage_shape)
            for couple in phantom_ROI:
                edge_value = config["rho"]
                self.image_net_input[couple] = edge_value
        elif (addon == "high_pixel"):
            edge_value = config["rho"]
            self.image_net_input[10,10] = edge_value
        elif (addon == "reduce_cold_value_MR"):
            self.image_net_input[self.cold_ROI == 1] = config["rho"]
        elif (addon == "remove_cold"):
            self.image_net_input[35:59,35:59] = 30
        elif (addon == "remove_ellipse_MR"):
            self.image_net_input[23:49,60:73] = 30
        # import matplotlib.pyplot as plt
        # import numpy as np
        # plt.imshow(self.image_net_input,vmin=np.min(self.image_net_input),vmax=np.max(self.image_net_input),cmap='gray')
        # plt.show()

    def extract_likelihood_from_log(self,path_log):
        theLog = read_table(path_log)
        fileRows = column_stack([theLog[col].str.contains("Log-likelihood", na=False) for col in theLog])
        likelihoodRows = array(theLog.loc[fileRows == 1])
        for rows in likelihoodRows:
            theLikelihoodRowString = rows[0][22:44]
            if theLikelihoodRowString[0] == '-':
                theLikelihoodRowString = '0'
            likelihood = float(theLikelihoodRowString)
            if (hasattr(self,"likelihoods_alpha")):
                self.likelihoods_alpha.append(likelihood)
            self.likelihoods.append(likelihood)