## Python libraries

# Useful
from pathlib import Path
from os import getcwd, makedirs
from os.path import exists, isfile
from functools import partial
from ray import tune
from numpy import dtype, fromfile, argwhere, isnan, zeros, squeeze, ones_like, mean, std, sum, array, column_stack, transpose, dstack, meshgrid, arange, where, unique, concatenate, setdiff1d, isin, ones, float32
from numpy import max as max_np
from numpy import min as min_np
from pandas import read_table
from matplotlib.pyplot import imshow, figure, colorbar, savefig, title, gcf, axis, show, xlabel
from re import split, findall, compile
from pandas import DataFrame

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
        if ("PSF" in config):
            if (not config["PSF"]):
                self.PSF = False
            else:
                self.PSF = True
        else: # Default is to use PSF
            self.PSF = True
        if ("end_to_end" in config): # Check if run DNA with end to end mode
            if (config["end_to_end"]):
                self.end_to_end = True
            else:
                self.end_to_end = False
        else:
            self.end_to_end = False

        # MIC study
        if ("nested" in self.method or "Gong" in self.method or "DIPRecon" in self.method):
            if ("override_SC_init" in config):
                self.override_SC_init = config['override_SC_init']
            else:
                self.override_SC_init = False

            if ("read_only_MV_csv" not in config):
                config["read_only_MV_csv"] = False

            if ("diffusion_model_like" in config):
                self.diffusion_model_like = config["diffusion_model_like"]
            else:
                self.diffusion_model_like = 0
            
            if ("diffusion_model_like_each_DIP" in config):
                self.diffusion_model_like_each_DIP = config["diffusion_model_like_each_DIP"]
                # if (self.diffusion_model_like_each_DIP == 0):
                #     self.nb_DIP_inputs = 1
                # else:
                #     self.nb_DIP_inputs = int(1/self.diffusion_model_like_each_DIP)
            else:
                self.diffusion_model_like_each_DIP = 0
                # self.nb_DIP_inputs = 1

            if ("several_DIP_inputs" in config): # Put several times the input
                self.several_DIP_inputs = config["several_DIP_inputs"]
            else:
                self.several_DIP_inputs = 1

            # Override number of DIP sub iterations if several inputs
            # if ("results" not in config["task"]):
            # config["sub_iter_DIP"] = int(config["sub_iter_DIP"] / self.several_DIP_inputs)

        self.subroot_data = root + '/data/Algo/' # Directory root
        
        if (config["task"] != "show_metrics_results_already_computed_following_step"):
            # Initialize useful variables
            self.subroot = self.subroot_data + 'debug/'*self.debug + self.phantom + '/'+ 'replicate_' + str(self.replicate) + '/' + self.method + '/' # Directory root
            self.subroot_metrics = self.subroot_data + 'debug/'*self.debug + 'metrics/' + self.phantom + '/'+ 'replicate_' + str(self.replicate) + '/' # Directory root for metrics
            self.suffix = self.suffix_func(config) # self.suffix to make difference between raytune runs (different hyperparameters)
            self.suffix_metrics = self.suffix_func(config,NNEPPS=True) # self.suffix with NNEPPS information
            if ("post_reco" in config["task"] and "post_reco" not in self.suffix):
                if ("post_reco_in_suffix" not in config):
                    self.suffix = "post_reco" + ' ' + self.suffix
                    self.suffix_metrics = config["task"][:10] + ' ' + self.suffix_metrics
                else:
                    if (config["post_reco_in_suffix"]):
                        self.suffix = "post_reco" + ' ' + self.suffix
                        self.suffix_metrics = config["task"][:10] + ' ' + self.suffix_metrics

            # if ("end_to_end" in config["task"] and "end_to_end" not in self.suffix):
                # self.suffix = "end_to_end" + ' ' + self.suffix
                # self.suffix_metrics = config["task"][:10] + ' ' + self.suffix_metrics

            # Define PET input dimensions according to input data dimensions
            self.PETImage_shape_str = self.read_input_dim(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.hdr')
            self.PETImage_shape = self.input_dim_str_to_list(self.PETImage_shape_str)

            # Define sinogram dimensions
            # self.scanner = "mCT_2D" # to be removed
            if (self.simulation and self.scanner == "mMR_2D"):
                self.sinogram_shape = (344,252,1)
                self.sinogram_shape_transpose = (252,344,1)
            elif (self.simulation and self.scanner == "mCT_2D"):
                self.sinogram_shape = (336,336,1)
                self.sinogram_shape_transpose = (336,336,1)
            
            # # Define ROIs for image0 phantom, otherwise it is already done in the database
            # if (self.phantom == "image0" or self.phantom == "image2_0" and config["task"] != "show_metrics_results_already_computed"):
            #     self.define_ROI_image0(self.PETImage_shape,self.subroot_data)
            # elif (self.phantom == "image2_3D" and config["task"] != "show_metrics_results_already_computed"):
            #     self.define_ROI_image2_3D(self.PETImage_shape,self.subroot_data)
            # elif (("4_" in self.phantom or self.phantom == "image400_0" or self.phantom == "image40_0" or self.phantom == "image40_1") and config["task"] != "show_metrics_results_already_computed"):
            #     self.define_ROI_new_phantom(self.PETImage_shape,self.subroot_data)
            # elif ((self.phantom == "image50_1" or "50_2" in self.phantom) and config["task"] != "show_metrics_results_already_computed"):
            #     self.define_ROI_brain_with_tumors(self.PETImage_shape,self.subroot_data)
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
        Path(self.subroot_data + 'Data/initialization/random').mkdir(parents=True, exist_ok=True)
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
            tune.run(partial(self.do_everything,root=root,only_suffix_replicate_file = only_suffix_replicate_file), config=config,local_dir = getcwd() + '/runs',resources_per_trial={"cpu": 1})
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
        elif (self.phantom == "imageUHR_IEC"):
            self.scanner = "UHR"
        
        else:
            self.scanner = "mMR_2D"

        # Define if simulation or not
        if ("50_" in self.phantom or "4_" in self.phantom or "2_" in self.phantom or "40_" in self.phantom or self.phantom == "imageUHR_IEC"):
            self.simulation = True
        else:
            self.simulation = False

        # Define PET input dimensions according to input data dimensions
        self.PETImage_shape_str = self.read_input_dim(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.hdr')
        self.PETImage_shape = self.input_dim_str_to_list(self.PETImage_shape_str)

        # # Loading Ground Truth image to compute metrics
        # self.image_gt = self.fijii_np(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.img',shape=(self.PETImage_shape),type_im='<f')            


        # Define ROIs for image0 phantom, otherwise it is already done in the database
        if (self.phantom == "image0" or self.phantom == "image2_0" and config["task"] != "show_metrics_results_already_computed"):
            self.define_ROI_image0(self.PETImage_shape,self.subroot_data)
        elif (self.phantom == "image2_3D" and config["task"] != "show_metrics_results_already_computed"):
            self.define_ROI_image2_3D(self.PETImage_shape,self.subroot_data)
        elif (("4_" in self.phantom or self.phantom == "image400_0" or self.phantom == "image40_0" or self.phantom == "image40_1") and config["task"] != "show_metrics_results_already_computed"):
            self.define_ROI_new_phantom(self.PETImage_shape,self.subroot_data)
        elif ((self.phantom == "image50_1" or "50_2" in self.phantom) and config["task"] != "show_metrics_results_already_computed"):
            self.define_ROI_brain_with_tumors(self.PETImage_shape,self.subroot_data)
        elif ((self.phantom == "imageUHR_IEC")):
            self.define_ROI_IEC_3D(self.PETImage_shape,self.subroot_data)

        # Defining ROIs
        self.phantom_ROI = self.get_phantom_ROI(self.phantom)
        if (self.simulation):
            bkg_ROI_path = self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "background_mask" + self.phantom[5:] + '.raw'
            cold_ROI_path = self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "cold_mask" + self.phantom[5:] + '.raw'
            if (isfile(bkg_ROI_path) or isfile(cold_ROI_path)):
                self.read_ROIs(bkg_ROI_path,cold_ROI_path)
            else:
                self.initializeGeneralVariables(config,root)

    def read_ROIs(self,bkg_ROI_path,cold_ROI_path):

        self.bkg_ROI = self.fijii_np(bkg_ROI_path, shape=(self.PETImage_shape),type_im='<f')
        
        # Define hot ROI according to the phantom
        if ("4_" in self.phantom or self.phantom == "image400_0" or self.phantom == "image40_0" or self.phantom == "image40_1" or self.phantom == "image50_1" or "50_2" in self.phantom):              
            self.hot_perfect_match_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "tumor_perfect_match_ROI_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
            self.hot_MR_recon = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "tumor_MR_mask_whole" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
            if ("50_2" not in self.phantom):
                self.hot_TEP_match_square_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "tumor_TEP_match_square_ROI_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
            else:
                self.hot_TEP_match_square_ROI = self.hot_perfect_match_ROI
            # This ROIs has already been defined, but is computed for the sake of simplicity
            self.hot_ROI = self.hot_perfect_match_ROI
            # TEP only tumor (if there is one)
            if (self.phantom == "image50_1"):
                self.hot_TEP_ROI_ref = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "tumor_white_matter_ref" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
            if ("50" in self.phantom):
                # Use the hot_TEP_ROI to actually store the MR only region
                self.hot_TEP_ROI = self.hot_MR_recon
            else:
                self.hot_TEP_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "tumor_TEP_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
        elif ("UHR_IEC" in self.phantom):
            self.hot1_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "hot1_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
            self.hot2_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "hot2_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
            self.hot3_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "hot3_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
            self.hot4_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "hot4_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
        else:
            self.hot_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "tumor_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
            # These ROIs do not exist, so put them equal to hot ROI for the sake of simplicity
            self.hot_TEP_ROI = array(self.hot_ROI)
            self.hot_TEP_match_square_ROI = array(self.hot_ROI)
            self.hot_perfect_match_ROI = array(self.hot_ROI)
            self.hot_MR_recon = array(self.hot_ROI)
        
        # Define cold ROI according to the phantom
        if ("UHR_IEC" in self.phantom):
            self.cold1_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "cold1_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
            self.cold2_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "cold2_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
        else:
            self.cold_ROI = self.fijii_np(cold_ROI_path, shape=(self.PETImage_shape),type_im='<f')
            if ("4" in self.phantom):
                self.cold_inside_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "cold_inside_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
                self.cold_edge_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "cold_edge_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
            else:
                self.cold_inside_ROI = self.cold_ROI
                self.cold_edge_ROI = self.cold_ROI

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
            if ('BSREM' in config["method"]['grid_search'][0] or 'nested' in config["method"]['grid_search'][0] or 'Gong' in config["method"]['grid_search'][0] or 'DIPRecon' in config["method"]['grid_search'][0] or 'APGMAP' in config["method"]['grid_search'][0]):
                config.pop("post_smoothing", None)
            if ((('ADMMLim' not in config["method"]['grid_search'][0] and 'nested' not in config["method"]['grid_search'][0]) and config["method"]['grid_search'][0] != 'ADMMLim_Bowsher' and "nested" not in config["method"]['grid_search'][0]) or "APGMAP" in config["recoInNested"]['grid_search'][0]):
                #config.pop("nb_inner_iteration", None)
                config.pop("alpha", None)
                config.pop("adaptive_parameters", None)
                config.pop("mu_adaptive", None)
                config.pop("tau", None)
                config.pop("tau_max", None)
                config.pop("stoppingCriterionValue", None)
                config.pop("saveSinogramsUAndV", None)
                #config.pop("xi", None)
            elif ((config["method"]['grid_search'][0] == 'ADMMLim' or config["method"]['grid_search'][0] == 'ADMMLim_Bowsher' or "nested" in config["method"]['grid_search'][0]) and config["adaptive_parameters"]['grid_search'][0] == "nothing"):
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
            if ("nested" in config["method"]['grid_search'][0] or "Gong" in config["method"]['grid_search'][0] or "DIPRecon" in config["method"]['grid_search'][0]):
                if ("end_to_end" in config):
                    if (config["end_to_end"]):
                        config.pop("sub_iter_DIP", None)
                    else:
                        config.pop("end_to_end", None)
                else:
                    config.pop("end_to_end", None)
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
        if (("ADMMLim" in config["method"] and "nested" not in config["method"]) or config["method"] == "ADMMLim_Bowsher"):
            config_copy.pop('nb_outer_iteration',None)
        elif ("post_reco" in config_copy["task"]):
            if ("post_reco_in_suffix" not in config_copy):
                config_copy.pop("sub_iter_DIP", None)
            else:
                if (config["post_reco_in_suffix"]):
                    config_copy.pop("sub_iter_DIP", None)
        if list(config.keys())[-1] == "sub_iter_DIP": # Remove sub_iter_DIP if it is the last key because it comes from an override in iEndToEnd.py
            config_copy.pop('sub_iter_DIP',None)
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

    def fijii_np_old(self,path,shape,type_im=None):
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
                # type_im = ('<f')*(type_im=='<f') + ('<d')*(type_im=='<d')
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
                # type_im = ('<f')*(type_im=='<d') + ('<d')*(type_im=='<f')
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
    
    def fijii_np(self,path,shape,type_im='<f'):
        """"Transforming raw data to numpy array"""
        if (type_im is None):
            if (self.FLTNB == 'float'):
                type_im = '<f'
            elif (self.FLTNB == 'double'):
                type_im = '<d'

        file_path=(path)
        dtype_np = dtype(type_im)
        with open(file_path, 'rb') as fid:
            data = fromfile(fid,dtype_np)
            image = data.reshape(shape)
                        
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
        if (1 in image_corrupt.shape): # 2D
            nb_slices = 1
        else: # 3D
            image_corrupt = transpose(image_corrupt)
            nb_slices = image_corrupt.shape[2]

        image_corrupt_scaled = ones_like(image_corrupt)
        param1 = zeros(nb_slices)
        param2 = zeros(nb_slices)
        
        for slice in range(nb_slices):
            if (scaling == 'standardization'):
                image_corrupt_scaled[:,:,slice], param1[slice], param2[slice] = self.stand_imag(image_corrupt[:,:,slice])
            elif (scaling == 'normalization'):
                image_corrupt_scaled[:,:,slice], param1[slice], param2[slice] = self.norm_imag(image_corrupt[:,:,slice])
            elif (scaling == 'normalization_init'):
                image_corrupt_scaled[:,:,slice], param1[slice], param2[slice] = self.norm_init_imag(image_corrupt[:,:,slice],param1,param2)
            elif (scaling == 'positive_normalization'):
                image_corrupt_scaled[:,:,slice], param1[slice], param2[slice] = self.norm_positive_imag(image_corrupt[:,:,slice])
            else: # No scaling required
                image_corrupt_scaled[:,:,slice], param1[slice], param2[slice] = squeeze(image_corrupt), 0, 0

        
        if (1 not in image_corrupt.shape): # 3D
            image_corrupt_scaled = transpose(image_corrupt_scaled)

        return image_corrupt_scaled, param1, param2 

    def descale_imag(self,image, param_scale1, param_scale2, scaling='standardization'):
        """ Descaling of input """
        try:
            image_np = image.detach().numpy()
        except:
            image_np = image
        # image_np = image_np.astype(np.float64)
        
        if (1 in image_np.shape): # 2D
            nb_slices = 1
        else: # 3D
            image_np = transpose(image_np)
            nb_slices = image_np.shape[2]
        
        nb_slices = len(param_scale1)

        for slice in range(nb_slices):
            if (scaling == 'standardization'):
                image_np[:,:,slice] = self.destand_numpy_imag(image_np[:,:,slice], param_scale1[slice], param_scale2[slice])
            elif (scaling == 'normalization'):
                image_np[:,:,slice] = self.denorm_numpy_imag(image_np[:,:,slice], param_scale1[slice], param_scale2[slice])
            elif (scaling == 'normalization_init'):
                image_np[:,:,slice] = self.denorm_numpy_imag(image_np[:,:,slice], param_scale1[slice], param_scale2[slice])
            elif (scaling == 'positive_normalization'):
                image_np[:,:,slice] = self.denorm_numpy_positive_imag(image_np[:,:,slice], param_scale1[slice], param_scale2[slice])
            else: # No scaling required
                print("no scaling required")

        if (1 not in image_np.shape): # 3D
            image_np = transpose(image_np)

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

    def assignROI(self, classResults):
        if (self.simulation):
            classResults.bkg_ROI = self.bkg_ROI
            if ("UHR_IEC" not in self.phantom):
                classResults.hot_TEP_ROI = self.hot_TEP_ROI
                if (self.phantom == "image50_1"):
                    classResults.hot_TEP_ROI_ref = self.hot_TEP_ROI_ref
                classResults.hot_TEP_match_square_ROI = self.hot_TEP_match_square_ROI
                classResults.hot_perfect_match_ROI = self.hot_perfect_match_ROI
                classResults.hot_MR_recon = self.hot_MR_recon
                classResults.hot_ROI = self.hot_ROI
                classResults.cold_ROI = self.cold_ROI
                classResults.cold_inside_ROI = self.cold_inside_ROI
                classResults.cold_edge_ROI = self.cold_edge_ROI
            else:
                classResults.hot1P_ROI = self.hot1_ROI
                classResults.hot2_ROI = self.hot2_ROI
                classResults.hot3_ROI = self.hot3_ROI
                classResults.hot4_ROI = self.hot4_ROI
                classResults.cold1_ROI = self.cold1_ROI
                classResults.cold2_ROI = self.cold2_ROI

    def assignVariablesFromResults(self,classResults):
        classResults.nb_replicates = self.nb_replicates
        classResults.debug = self.debug
        if (hasattr(self, 'rho')):
            classResults.rho = self.rho
        classResults.fixed_hyperparameters_list = self.fixed_hyperparameters_list
        classResults.hyperparameters_list = self.hyperparameters_list
        classResults.phantom_ROI = self.phantom_ROI
        classResults.scanner = self.scanner
        classResults.simulation = self.simulation

    def points_in_circle(self,center_x,center_y,center_z,radius,PETImage_shape,inner_circle=True): # x and y are inverted in an array compared to coordinates
        center_y += int(PETImage_shape[0]/2)
        center_x += int(PETImage_shape[1]/2)
        dim_y = PETImage_shape[1]
        dim_x = PETImage_shape[0]
        if (len(PETImage_shape) >= 2):
            center_z += int(PETImage_shape[2]/2)
            dim_z = PETImage_shape[2]
        else:
            dim_z = 1

        if (dim_z == 1):
            x, y = meshgrid(arange(dim_x), arange(dim_y))
            mask = (x+0.5-center_x)**2 + (y+0.5-center_y)**2 <= radius**2
        else:
            x, y, z = meshgrid(arange(dim_x), arange(dim_y), arange(dim_z))
            mask = (x+0.5-center_x)**2 + (y+0.5-center_y)**2 + (z+0.5-center_z)**2 <= radius**2
        liste = argwhere(mask)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(mask)
        # plt.show()
        
        # liste = [] 
        # for x in range(dim_x):
        #     for y in range(dim_y):
        #         for z in range(dim_z):
        #             if (x+0.5-center_x)**2 + (y+0.5-center_y)**2 <= radius**2:
        #                 liste.append((x,y,z))

        return liste, mask.astype(float32)

    def points_in_cylinder(self,center_y,center_x,center_z,radius,length,PETImage_shape,inner_circle=True): # x and y are inverted in an array compared to coordinates
        center_x += int(PETImage_shape[0]/2)
        center_y += int(PETImage_shape[1]/2)
        dim_x = PETImage_shape[0]
        dim_y = PETImage_shape[1]
        if (len(PETImage_shape) >= 2):
            center_z += int(PETImage_shape[2]/2)
            dim_z = PETImage_shape[2]
        else:
            dim_z = 1
            
        x, y, z = meshgrid(arange(dim_x), arange(dim_y), arange(dim_z))
        mask_circle = ((x-center_x)**2 + (y-center_y)**2 <= radius**2) & (z-center_z >= -length / 2) & (z-center_z <= length / 2)
        liste = argwhere(mask_circle)

        return liste

    def define_ROI_image0(self,PETImage_shape,subroot):
        voxel_size = 4

        phantom_ROI, mask = self.points_in_circle(0/voxel_size, 0/voxel_size, 0/voxel_size, 150/voxel_size, PETImage_shape)
        cold_ROI, mask = self.points_in_circle(-40/voxel_size, -40/voxel_size, 0/voxel_size, 40/voxel_size-1, PETImage_shape)
        hot_ROI, mask = self.points_in_circle(50/voxel_size, 10/voxel_size, 0/voxel_size, 20/voxel_size-1, PETImage_shape)
        cold_ROI_bkg, mask = self.points_in_circle(-40/voxel_size, -40/voxel_size, 0/voxel_size, 40/voxel_size+1, PETImage_shape)
        hot_ROI_bkg, mask = self.points_in_circle(50/voxel_size, 10/voxel_size, 0/voxel_size, 20/voxel_size+1, PETImage_shape)
        phantom_ROI_bkg, mask = self.points_in_circle(0/voxel_size, 0/voxel_size, 0/voxel_size, 150/voxel_size-1, PETImage_shape)

        # Create background ROI by removing other ROIs
        all_defined_ROI = [phantom_ROI_bkg, cold_ROI_bkg, cold_ROI_bkg]
        bkg_ROI = self.create_bkg(all_defined_ROI, PETImage_shape)

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

        voxel_size = 4

        phantom_ROI, mask = self.points_in_circle(0/voxel_size, 0/voxel_size, 0/voxel_size, 150/voxel_size, PETImage_shape)
        cold_ROI, mask = self.points_in_circle(-40/voxel_size, -40/voxel_size, 0/voxel_size, 40/voxel_size-1, PETImage_shape)
        hot_ROI, mask = self.points_in_circle(50/voxel_size, 10/voxel_size, 0/voxel_size, 20/voxel_size-1, PETImage_shape)
        cold_ROI_bkg, mask = self.points_in_circle(-40/voxel_size, -40/voxel_size, 0/voxel_size, 40/voxel_size+1, PETImage_shape)
        hot_ROI_bkg, mask = self.points_in_circle(50/voxel_size, 10/voxel_size, 0/voxel_size, 20/voxel_size+1, PETImage_shape)
        phantom_ROI_bkg, mask = self.points_in_circle(0/voxel_size, 0/voxel_size, 0/voxel_size, 150/voxel_size-1, PETImage_shape)

        # Create background ROI by removing other ROIs
        all_defined_ROI = [phantom_ROI_bkg, cold_ROI_bkg, cold_ROI_bkg]
        bkg_ROI = self.create_bkg(all_defined_ROI, PETImage_shape)

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

        voxel_size = 4

        # Define ROIs
        phantom_ROI, phantom_mask = self.points_in_circle(0/voxel_size, 0/voxel_size, 0/voxel_size, 150/voxel_size, PETImage_shape)
        cold_ROI, cold_mask = self.points_in_circle(-40/voxel_size, -40/voxel_size, 0/voxel_size, 40/voxel_size-1, PETImage_shape)
        cold_inside_ROI, cold_inside_mask = self.points_in_circle(-40/voxel_size, -40/voxel_size, 0/voxel_size, 40/voxel_size-3, PETImage_shape)

        # Create edges of cold ROI by removing other inside voxels in cold_inside_ROI from cold_ROI
        cold_edge_ROI = [cold_ROI, cold_inside_ROI]
        cold_edge_ROI = self.create_bkg(cold_edge_ROI,PETImage_shape)
        
        _, tumor_TEP_mask = self.points_in_circle(50/voxel_size, 10/voxel_size, 0/voxel_size, 20/voxel_size-1, PETImage_shape)
        _, tumor_TEP_match_square_ROI_mask = self.points_in_circle(-20/voxel_size, 70/voxel_size, 0/voxel_size, 20/voxel_size-1, PETImage_shape)
        _, tumor_perfect_match_ROI_mask = self.points_in_circle(50/voxel_size, 90/voxel_size, 0/voxel_size, 20/voxel_size-1, PETImage_shape)

        cold_ROI_bkg, mask = self.points_in_circle(-40/voxel_size, -40/voxel_size, 0/voxel_size, 40/voxel_size+1, PETImage_shape)
        hot_TEP_ROI_bkg, mask = self.points_in_circle(50/voxel_size, 10/voxel_size, 0/voxel_size, 20/voxel_size+1, PETImage_shape)
        hot_TEP_match_square_ROI_bkg, mask = self.points_in_circle(-20/voxel_size, 70/voxel_size, 0/voxel_size, 20/voxel_size+1, PETImage_shape)
        hot_perfect_match_ROI_bkg, mask = self.points_in_circle(50/voxel_size, 90/voxel_size, 0/voxel_size, 20/voxel_size+1, PETImage_shape)
        phantom_ROI_bkg, mask = self.points_in_circle(0/voxel_size, 0/voxel_size, 0/voxel_size, 150/voxel_size-1, PETImage_shape)


        # Create background ROI by removing other ROIs
        all_defined_ROI = [phantom_ROI_bkg, cold_ROI_bkg, hot_TEP_ROI_bkg, hot_TEP_match_square_ROI_bkg, hot_perfect_match_ROI_bkg]
        bkg_ROI = self.create_bkg(all_defined_ROI, PETImage_shape)

        # cold_mask = zeros(PETImage_shape, dtype='<f')
        # cold_inside_mask = zeros(PETImage_shape, dtype='<f')
        cold_edge_mask = zeros(PETImage_shape, dtype='<f')
        # tumor_TEP_mask = zeros(PETImage_shape, dtype='<f')
        # tumor_TEP_match_square_ROI_mask = zeros(PETImage_shape, dtype='<f')
        # tumor_perfect_match_ROI_mask = zeros(PETImage_shape, dtype='<f')
        # phantom_mask = zeros(PETImage_shape, dtype='<f')
        bkg_mask = zeros(PETImage_shape, dtype='<f')
        tumor_3a_mask_whole = zeros(PETImage_shape, dtype=float32)
        tumor_3a_mask_whole[26:40,63:69] = 1

        # ROI_list = [cold_ROI, cold_inside_ROI, cold_edge_ROI, hot_TEP_ROI, hot_TEP_match_square_ROI, hot_perfect_match_ROI, phantom_ROI, bkg_ROI]
        # mask_list = [cold_mask, cold_inside_mask, cold_edge_mask, tumor_TEP_mask, tumor_TEP_match_square_ROI_mask, tumor_perfect_match_ROI_mask, phantom_mask, bkg_mask]
        ROI_list = [cold_edge_ROI, bkg_ROI]
        mask_list = [cold_edge_mask, bkg_mask]

        for i in range(len(ROI_list)):
            ROI = ROI_list[i]
            mask = mask_list[i]
            for couple in ROI:
                mask[tuple(couple)] = 1 # Convert into tuple to avoid bad indexing

        # Storing into file instead of defining them at each metrics computation
        self.save_img(cold_mask, subroot+'Data/database_v2/' + self.phantom + '/' + "cold_mask" + self.phantom[5:] + '.raw')
        self.save_img(cold_inside_mask, subroot+'Data/database_v2/' + self.phantom + '/' + "cold_inside_mask" + self.phantom[5:] + '.raw')
        self.save_img(cold_edge_mask, subroot+'Data/database_v2/' + self.phantom + '/' + "cold_edge_mask" + self.phantom[5:] + '.raw')
        self.save_img(tumor_TEP_mask, subroot+'Data/database_v2/' + self.phantom + '/' + "tumor_TEP_mask" + self.phantom[5:] + '.raw')
        self.save_img(tumor_TEP_match_square_ROI_mask, subroot+'Data/database_v2/' + self.phantom + '/' + "tumor_TEP_match_square_ROI_mask" + self.phantom[5:] + '.raw')
        self.save_img(tumor_perfect_match_ROI_mask, subroot+'Data/database_v2/' + self.phantom + '/' + "tumor_perfect_match_ROI_mask" + self.phantom[5:] + '.raw')
        self.save_img(phantom_mask, subroot+'Data/database_v2/' + self.phantom + '/' + "phantom_mask" + self.phantom[5:] + '.raw')
        self.save_img(bkg_mask, subroot+'Data/database_v2/' + self.phantom + '/' + "background_mask" + self.phantom[5:] + '.raw')
        self.save_img(tumor_3a_mask_whole, subroot+'Data/database_v2/' + self.phantom + '/' + "tumor_MR_mask_whole" + self.phantom[5:] + '.raw')


    def define_ROI_brain_with_tumors(self,PETImage_shape,subroot):

        # Define external radius to not take into account in ROI definition
        remove_external_radius = 1 # MIC abstract 2022, 2023
        # remove_external_radius = 3 # ROIs further away from true edges

        voxel_size = 2

        # Define ROIs
        tumor_1a_ROI, mask = self.points_in_circle(15,-25,0/voxel_size,4-remove_external_radius,PETImage_shape)

        # tumor_1b_ROI, mask = self.points_in_circle(0,25,4,PETImage_shape)
        xx,yy = meshgrid(arange(65,72),arange(53,59))
        tumor_1b_ROI = list(map(tuple, dstack([xx.ravel(), yy.ravel()])[0]))
        for tuple_ in [(72,53),(72,52),(73,51),(74,50),(75,49),(72,58),(43,42),(42,42),(41,42),(40,42),(39,42),(45,43),(44,43),(43,43),(42,43),(45,44),(44,44),(46,66),(46,67),(45,67),(46,66),(44,68),(45,68)]:
            tumor_1b_ROI.append(tuple_)

        phantom_ROI, mask = self.points_in_circle(-25, 0, 0/voxel_size, 8 - remove_external_radius, PETImage_shape)
        tumor_2_MR_ROI = phantom_ROI
        tumor_2_PET_ROI, mask = self.points_in_circle(-27, 0, 0/voxel_size, 4, PETImage_shape) # Do not remove external radius to show effect of intermediate setting
        tumor_3a_ROI, mask = self.points_in_circle(13, 25, 0/voxel_size, 4 - remove_external_radius, PETImage_shape)
        tumor_3a_ROI_whole, mask = self.points_in_circle(13, 25, 0/voxel_size, 4, PETImage_shape)
        tumor_3a_ref_ROI, mask = self.points_in_circle(-11, 25, 0/voxel_size, 4 - remove_external_radius, PETImage_shape)
        
        tumor_3a_mask = zeros(PETImage_shape, dtype='<f')
        tumor_3a_mask_whole = zeros(PETImage_shape, dtype='<f')
        tumor_3a_mask_ref = zeros(PETImage_shape, dtype='<f')
        tumor_1b_mask = zeros(PETImage_shape, dtype='<f')
        tumor_1a_mask = zeros(PETImage_shape, dtype='<f')
        tumor_2_PET_mask = zeros(PETImage_shape, dtype='<f')

        ROI_MR_list = [tumor_1a_ROI,tumor_2_MR_ROI,tumor_3a_ROI,tumor_3a_mask_ref]
        ROI_PET_list = [tumor_1a_ROI,tumor_2_PET_ROI]

        ROI_list = [tumor_3a_ROI,tumor_1b_ROI,tumor_1a_ROI,tumor_2_PET_ROI,tumor_3a_ROI_whole,tumor_3a_ref_ROI]
        mask_list = [tumor_3a_mask, tumor_1b_mask, tumor_1a_mask, tumor_2_PET_mask,tumor_3a_mask_whole,tumor_3a_mask_ref]
        for i in range(len(ROI_list)):
            ROI = ROI_list[i]
            mask = mask_list[i]
            for couple in ROI:
                mask[couple] = 1

        # Storing into file instead of defining them at each metrics computation
        self.save_img(tumor_1b_mask, subroot+'Data/database_v2/' + self.phantom + '/' + "cold_mask" + self.phantom[5:] + '.raw')
        self.save_img(tumor_3a_mask_whole, subroot+'Data/database_v2/' + self.phantom + '/' + "tumor_MR_mask_whole" + self.phantom[5:] + '.raw')
        self.save_img(tumor_1a_mask, subroot+'Data/database_v2/' + self.phantom + '/' + "tumor_perfect_match_ROI_mask" + self.phantom[5:] + '.raw')
        # if (self.phantom == "image50_1"):
        #     self.save_img(tumor_1a_mask, subroot+'Data/database_v2/' + self.phantom + '/' + "tumor_mask" + self.phantom[5:] + '.raw')
        if (self.phantom == "image50_1"):
            self.save_img(tumor_2_PET_mask, subroot+'Data/database_v2/' + self.phantom + '/' + "tumor_TEP_match_square_ROI_mask" + self.phantom[5:] + '.raw')
            self.save_img(tumor_3a_mask, subroot+'Data/database_v2/' + self.phantom + '/' + "tumor_MR_mask" + self.phantom[5:] + '.raw') # Useless
            self.save_img(tumor_3a_mask_ref, subroot+'Data/database_v2/' + self.phantom + '/' + "tumor_white_matter_ref" + self.phantom[5:] + '.raw')
            # self.save_img(TAKE_FROM_ERODED_WHITE_MATTER_IN_OTHER_FILE, subroot+'Data/database_v2/' + self.phantom + '/' + "background_mask" + self.phantom[5:] + '.raw') # Useless for computation, just for constitency with other phantoms


    def define_ROI_IEC_3D(self,PETImage_shape,subroot):
        
        voxel_size = 1.2

        cold1_ROI, mask = self.points_in_circle(49.54/voxel_size, 28.6/voxel_size, 0/voxel_size, 5/voxel_size-1, PETImage_shape)
        phantom_ROI = self.points_in_cylinder(0/voxel_size, 0/voxel_size, 0/voxel_size, 105/voxel_size, 180/voxel_size, PETImage_shape)
        cold2_ROI, mask = self.points_in_circle(0/voxel_size, 57.2/voxel_size, 0/voxel_size, 6.5/voxel_size-1, PETImage_shape)
        hot1_ROI, mask = self.points_in_circle(-49.54/voxel_size, 28.6/voxel_size, 0/voxel_size, 8.5/voxel_size-1, PETImage_shape)
        hot2_ROI, mask = self.points_in_circle(-49.54/voxel_size, -28.6/voxel_size, 0/voxel_size, 11/voxel_size-1, PETImage_shape)
        hot3_ROI, mask = self.points_in_circle(0/voxel_size, -57.2/voxel_size, 0/voxel_size, 14/voxel_size-1, PETImage_shape)
        hot4_ROI, mask = self.points_in_circle(49.54/voxel_size, -28.6/voxel_size, 0/voxel_size, 18.5/voxel_size-1, PETImage_shape)

        cold1_ROI_bkg, mask = self.points_in_circle(49.54/voxel_size, 28.6/voxel_size, 0/voxel_size, 5/voxel_size+1, PETImage_shape)
        cold2_ROI_bkg, mask = self.points_in_circle(0/voxel_size, 57.2/voxel_size, 0/voxel_size, 6.5/voxel_size+1, PETImage_shape)
        hot1_ROI_bkg, mask = self.points_in_circle(-49.54/voxel_size, 28.6/voxel_size, 0/voxel_size, 8.5/voxel_size+1, PETImage_shape)
        hot2_ROI_bkg, mask = self.points_in_circle(-49.54/voxel_size, -28.6/voxel_size, 0/voxel_size, 11/voxel_size+1, PETImage_shape)
        hot3_ROI_bkg, mask = self.points_in_circle(0/voxel_size, -57.2/voxel_size, 0/voxel_size, 14/voxel_size+1, PETImage_shape)
        hot4_ROI_bkg, mask = self.points_in_circle(49.54/voxel_size, -28.6/voxel_size, 0/voxel_size, 18.5/voxel_size+1, PETImage_shape)
        
        phantom_ROI_bkg = self.points_in_cylinder(0/voxel_size, 0/voxel_size, 0/voxel_size, 150/voxel_size + 1, 180/voxel_size + 1, PETImage_shape)

        # Create background ROI by removing other ROIs
        all_defined_ROI = [phantom_ROI_bkg, cold1_ROI_bkg, cold2_ROI_bkg, hot1_ROI_bkg, hot2_ROI_bkg, hot3_ROI_bkg, hot4_ROI_bkg]
        bkg_ROI = self.create_bkg(all_defined_ROI, PETImage_shape)

        # Create masks for each ROI
        cold1_mask = zeros(PETImage_shape, dtype='<f')
        cold2_mask = zeros(PETImage_shape, dtype='<f')
        hot1_mask = zeros(PETImage_shape, dtype='<f')
        hot2_mask = zeros(PETImage_shape, dtype='<f')
        hot3_mask = zeros(PETImage_shape, dtype='<f')
        hot4_mask = zeros(PETImage_shape, dtype='<f')
        phantom_mask = zeros(PETImage_shape, dtype='<f')
        bkg_mask = zeros(PETImage_shape, dtype='<f')

        ROI_list = [cold1_ROI, cold2_ROI, hot1_ROI, hot2_ROI, hot3_ROI, hot4_ROI, phantom_ROI, bkg_ROI]
        mask_list = [cold1_mask, cold2_mask, hot1_mask, hot2_mask, hot3_mask, hot4_mask, phantom_mask, bkg_mask]

        # Fill mask with one for each ROI
        for i in range(len(ROI_list)):
            ROI = ROI_list[i]
            mask = mask_list[i]
            # Convert the coordinates in cold1_ROI to NumPy arrays
            x_coords, y_coords, z_coords = array(ROI).T
            mask[x_coords, y_coords, z_coords] = 1

        # Storing into file instead of defining them at each metrics computation
        self.save_img(transpose(phantom_mask,axes=(2,0,1)), subroot+'Data/database_v2/' + self.phantom + '/' + "phantom_mask" + self.phantom[5:] + '.raw')
        self.save_img(transpose(cold1_mask, axes=(2,0,1)), subroot+'Data/database_v2/' + self.phantom + '/' + "cold1_mask" + self.phantom[5:] + '.raw')
        self.save_img(transpose(cold2_mask, axes=(2,0,1)), subroot+'Data/database_v2/' + self.phantom + '/' + "cold2_mask" + self.phantom[5:] + '.raw')
        self.save_img(transpose(hot1_mask, axes=(2,0,1)), subroot+'Data/database_v2/' + self.phantom + '/' + "hot1_mask" + self.phantom[5:] + '.raw')
        self.save_img(transpose(hot2_mask, axes=(2,0,1)), subroot+'Data/database_v2/' + self.phantom + '/' + "hot2_mask" + self.phantom[5:] + '.raw')
        self.save_img(transpose(hot3_mask, axes=(2,0,1)), subroot+'Data/database_v2/' + self.phantom + '/' + "hot3_mask" + self.phantom[5:] + '.raw')
        self.save_img(transpose(hot4_mask, axes=(2,0,1)), subroot+'Data/database_v2/' + self.phantom + '/' + "hot4_mask" + self.phantom[5:] + '.raw')
        self.save_img(transpose(bkg_mask, axes=(2,0,1)), subroot+'Data/database_v2/' + self.phantom + '/' + "background_mask" + self.phantom[5:] + '.raw')
        
        # Define GT phantom
        self.define_GT_IEC_3D(PETImage_shape,subroot, cold1_mask, cold2_mask, hot1_mask, hot2_mask, hot3_mask, hot4_mask, phantom_mask)

    def define_GT_IEC_3D(self,PETImage_shape,subroot, cold1_mask, cold2_mask, hot1_mask, hot2_mask, hot3_mask, hot4_mask, phantom_mask):
        image_GT = ones(PETImage_shape, dtype='<f')
        # Cold regions to 0
        image_GT = image_GT * (1 - cold1_mask)
        image_GT = image_GT * (1 - cold2_mask)
        ### Arbritrary value for background and hot ROIs. Only ratio is known
        ratio_hot_bkg = 4 # known
        # Define background
        image_GT = image_GT * phantom_mask
        # Define hot regions
        image_GT = image_GT * (1 - hot1_mask) + ratio_hot_bkg * hot1_mask
        image_GT = image_GT * (1 - hot2_mask) + ratio_hot_bkg * hot2_mask
        image_GT = image_GT * (1 - hot3_mask) + ratio_hot_bkg * hot3_mask
        image_GT = image_GT * (1 - hot4_mask) + ratio_hot_bkg * hot4_mask

        # Storing into file instead of defining them at each metrics computation
        self.save_img(transpose(image_GT,axes=(2,0,1)), subroot+'Data/database_v2/' + self.phantom + '/' + self.phantom + '.img')
        self.save_img(transpose(image_GT,axes=(2,0,1)), subroot+'Data/database_v2/' + self.phantom + '/' + self.phantom + '.raw')

    def create_bkg(self, all_defined_ROI, PETImage_shape):
        # Create a DataFrame from the concatenated arrays
        if (PETImage_shape[2] == 1):
            df = DataFrame(concatenate(all_defined_ROI, axis=0), columns=['x', 'y'])
        else:
            df = DataFrame(concatenate(all_defined_ROI, axis=0), columns=['x', 'y', 'z'])
        # df = DataFrame(concatenate(all_defined_ROI, axis=0), columns=['x', 'y'])
        # Remove duplicates from both arrays
        df_unique = df.drop_duplicates(keep=False)
        bkg_ROI = df_unique.values
        return bkg_ROI

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
            # imshow(image, cmap=cmap_,vmin=0,vmax=1) # Showing all images with same contrast and white is zero (gray_r)
            imshow(image, cmap=cmap_,vmin=0.06,vmax=0.145) # Showing all images with same contrast and white is zero (gray_r)
        else:
            imshow(image, cmap=cmap_,vmin=min_np(image_gt),vmax=1.25*max_np(image_gt)) # Showing all images with same contrast and white is zero (gray_r)
        axis('off')
        #show()

        # if (isnan(sum(image))):
        #     raise ValueError("NaNs detected in image. Stopping computation (" + "replicate_" + str(i) + "/" + suffix + ")")

        # Saving this figure locally
        Path(self.subroot + 'Images/tmp/' + suffix).mkdir(parents=True, exist_ok=True)
        #system('rm -rf' + self.subroot + 'Images/tmp/' + suffix + '/*')
        from textwrap import wrap
        if (MIC_show):
            cbar = colorbar()
            import matplotlib.pyplot as plt
            plt.tight_layout()
            plt.rcParams['figure.figsize'] = 10, 10
            import matplotlib
            font = {'family' : 'normal',
            'size'   : 14}
            matplotlib.rc('font', **font)
            cbar.ax.tick_params(labelsize=20) 
        else:
            colorbar()
            axis('off')

        # title("<0.06",fontsize=20)
        title(">0.14",fontsize=20)
        savefig(self.subroot + 'Images/tmp/' + suffix + '/' + name + '_' + str(i) + '.png')
        
        # added line for small title of interest
        suffix = self.suffix_func(self.config,hyperparameters_list = ["lr", "opti_DIP"])
        
        wrapped_title = "\n".join(wrap(suffix, 80))

        #title(wrapped_title + "\n" + name,fontsize=8)
        #title(wrapped_title,fontsize=10)
        title(wrapped_title,fontsize=16)
        # Adding this figure to tensorboard
        writer.add_figure(name,gcf(),global_step=i,close=True)# for videos, using slider to change image with global_step

    # def castor_common_command_line(self, subroot, PETImage_shape_str, phantom, replicates, post_smoothing=0):
    #     executable = 'castor-recon'
    #     header_file = ' -df ' + subroot + 'Data/database_v2/' + phantom + '/data' + phantom[5:] + '_' + str(replicates) + '/data' + phantom[5:] + '_' + str(replicates) + '.cdh' # PET data pat
    #     dim = ' -dim ' + PETImage_shape_str
    #     vb = ' -vb 3'
    #     th = ' -th ' + str(self.nb_threads)
    #     if (self.scanner == "UHR"):
    #         vox = ' -vox 1.2,1.2,1.2'
    #         proj = ' -proj distanceDriven'
    #         psf = ''
    #         sensitivity = " -sensitivity " + self.subroot_data + 'Data/database_v2/' + self.phantom + '/data' + self.phantom[5:] + '_' + str(config["replicates"]) + '/sensitivity' + self.phantom[5:] + '_' + str(config["replicates"]) + '.cdh'
    #     else:
    #         sensitivity = "" # No sensitivity for histogram data 
    #         if (self.scanner != "mMR_3D"):
    #             proj = ' -proj incrementalSiddon'
    #             if (self.phantom != "image50_0" and self.phantom != "image50_1" and "50_2" not in self.phantom):
    #                 vox = ' -vox 4,4,4'
    #             else:
    #                 vox = ' -vox 2,2,2'
    #         else:
    #             vox = ' -vox 2.08626,2.08626,2.03125'
    #             if ("1" in PETImage_shape_str.split(',')): # 2D
    #                 psf = ' -conv gaussian,4,1,3.5::psf'
    #             else: # 3D
    #                 if (self.scanner == "mMR_3D"):
    #                     psf = ' -conv gaussian,4.5,4.5,3.5::psf' # isotropic psf in simulated phantoms
    #                 else:
    #                     psf = ' -conv gaussian,4,4,3.5::psf' # isotropic psf in simulated phantoms

    #     # No PSF if it was not asked by user
    #     if (not self.PSF):
    #         psf = ''


    #     if (post_smoothing != 0):
    #         if ("1" in PETImage_shape_str.split(',')): # 2D
    #             conv = ' -conv gaussian,' + str(post_smoothing) + ',1,3.5::post'
    #         else: # 3D
    #             conv = ' -conv gaussian,' + str(post_smoothing) + ',' + str(post_smoothing) + ',3.5::post' # isotropic post smoothing
    #     else:
    #         conv = ''
    #     # Computing likelihood
    #     if (self.castor_foms):
    #         opti_like = ' -opti-fom'
    #     else:
    #         opti_like = ''

    #     return executable + dim + vox + header_file + vb + th + proj + opti_like + psf + conv + sensitivity
    

    def castor_common_command_line(self, subroot, PETImage_shape_str, phantom, replicates, post_smoothing=0,mlem_quick=False):
        executable = 'castor-recon'
        dim = ' -dim ' + PETImage_shape_str
        vb = ' -vb 3'
        th = ' -th ' + str(self.nb_threads)
        if (not mlem_quick):
            header_file = ' -df ' + subroot + 'Data/database_v2/' + phantom + '/data' + phantom[5:] + '_' + str(replicates) + '/data' + phantom[5:] + '_' + str(replicates) + '.cdh' # PET data pat
        else:
            header_file = ' -df ' + self.subroot_data + 'Data/database_v2/' + self.phantom + '/data' + self.phantom[5:] + '_' + str(self.config["replicates"]) + '/data' + self.phantom[5:] + '_' + str(self.config["replicates"]) + '.cdh' # PET data path
        if (self.scanner == "UHR"):
            vox = ' -vox 1.2,1.2,1.2'
            proj = ' -proj distanceDriven'
            psf = ''
            sensitivity = " -sens " + self.subroot_data + 'Data/database_v2/' + self.phantom + '/sensitivity' + self.phantom[5:] + '_' + str(self.config["replicates"]) + '.hdr'
        else:
            sensitivity = "" # No sensitivity for histogram data 
            if (self.scanner != "mMR_3D"):
                proj = ' -proj incrementalSiddon'
                if (self.phantom != "image50_0" and self.phantom != "image50_1" and "50_2" not in self.phantom):
                    vox = ' -vox 4,4,4'
                else:
                    vox = ' -vox 2,2,2'
            else:
                vox = ' -vox 2.08626,2.08626,2.03125'
                if ("1" in PETImage_shape_str.split(',')): # 2D
                    psf = ' -conv gaussian,4,1,3.5::psf'
                else: # 3D
                    if (self.scanner == "mMR_3D"):
                        psf = ' -conv gaussian,4.5,4.5,3.5::psf' # isotropic psf in simulated phantoms
                    else:
                        psf = ' -conv gaussian,4,4,3.5::psf' # isotropic psf in simulated phantoms

        # No PSF if it was not asked by user
        if (not self.PSF):
            psf = ''


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

        return executable + dim + vox + header_file + vb + th + proj + opti_like + psf + conv + sensitivity

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
            # Choose penalty config file according to Bowsher weights or not
            penaltyStrength = ' -pnlt-beta ' + str(self.beta)
            if ("Bowsher" in self.config):
                if (self.config["Bowsher"]):
                    Bowsher = True
                else:
                    Bowsher = False
            else:
                Bowsher = False
            if (Bowsher):
                pnlt = ' -pnlt ' + penalty + ':' + self.subroot_data + method + '_MRF_Bowsher.conf'
                pnlt += ' -multimodal ' + self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '_mr.hdr'
            else:
                pnlt = ' -pnlt ' + penalty + ':' + self.subroot_data + method + '_MRF.conf'


        elif (method == 'BSREM'):
            opti = ' -opti ' + method + ':' + self.subroot_data + method + '.conf'
            # Choose penalty config file according to Bowsher weights or not
            penaltyStrength = ' -pnlt-beta ' + str(self.beta)
            if ("Bowsher" in self.config):
                if (self.config["Bowsher"]):
                    Bowsher = True
                else:
                    Bowsher = False
            else:
                Bowsher = False
            if (Bowsher):
                pnlt = ' -pnlt ' + penalty + ':' + self.subroot_data + method + '_MRF_Bowsher.conf'
                pnlt += ' -multimodal ' + self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '_mr.hdr'
            else:
                pnlt = ' -pnlt ' + penalty + ':' + self.subroot_data + method + '_MRF.conf'
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

                    # Choose penalty config file according to Bowsher weights or not
                    if ("Bowsher" in self.config):
                        if (self.config["Bowsher"]):
                            Bowsher = True
                        else:
                            Bowsher = False
                    else:
                        Bowsher = False

                    if (Bowsher):
                        pnlt = ' -pnlt ' + penalty + ':' + self.subroot_data + method + '_MRF_Bowsher.conf'
                        pnlt += ' -multimodal ' + self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '_mr.hdr'
                    elif penalty == "MRF":
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
                image_gt = self.fijii_np(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.img',shape=(self.PETImage_shape),type_im='<f')
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
        if ("scaled" in last_file):
            last_file = sorted_files[-2]
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
                if ("post_reco_in_suffix" not in config):
                    self.total_nb_iter = config["sub_iter_DIP"]
                else:
                    if (config["post_reco_in_suffix"]):
                        self.total_nb_iter = config["sub_iter_DIP"]
                    else:
                        self.total_nb_iter = config["sub_iter_DIP_initial_and_final"]
            else:
                try:
                    if (stopping_criterion):
                        self.path_stopping_criterion = self.subroot + 'Block2/' + self.suffix + '/' + 'IR_stopping_criteria.log'
                        with open(self.path_stopping_criterion) as f:
                            first_line = f.readline() # Read first line to get second one
                            #self.total_nb_iter = min(int(f.readline().rstrip()) - self.i_init, config["nb_outer_iteration"] - self.i_init + 1)
                            # self.total_nb_iter = int(f.readline().rstrip()) - self.i_init - 1
                            self.total_nb_iter = int(f.readline().rstrip())
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
            phantom_ROI, mask = self.points_in_circle_edge(0/4,0/4,0/4,150/4,self.PETImage_shape)
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
