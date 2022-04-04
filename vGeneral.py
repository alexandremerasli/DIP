## Python libraries

# Useful
from pathlib import Path
import os
from functools import partial
from ray import tune
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import sys

import abc
class vGeneral(abc.ABC):
    @abc.abstractmethod
    def __init__(self,config):
        print("__init__")
        self.experiment = "not updated"

    def split_config(self,config):
        fixed_config = dict(config)
        hyperparameters_config = dict(config)
        print(self.hyperparameters_list)
        for key in config.keys():
            if key in self.hyperparameters_list:
                fixed_config.pop(key, None)
            else:
                hyperparameters_config.pop(key, None)

        return fixed_config, hyperparameters_config

    def initializeGeneralVariables(self,fixed_config,hyperparameters_config,root):
        """General variables"""

        # Initialize some parameters from fixed_config
        self.finetuning = fixed_config["finetuning"]
        print(fixed_config["image"])
        self.phantom = fixed_config["image"]
        self.net = fixed_config["net"]
        self.method = fixed_config["method"]
        self.processing_unit = fixed_config["processing_unit"]
        self.nb_threads = fixed_config["nb_threads"]
        self.max_iter = fixed_config["max_iter"] # Outer iterations
        self.experiment = fixed_config["experiment"] # Label of the experiment
        self.replicate = fixed_config["replicates"] # Label of the replicate
        self.nb_threads = fixed_config["nb_threads"]

        # Initialize useful variables
        self.subroot = root + '/data/Algo/' + 'replicate_' + str(self.replicate) + '/' # Directory root
        self.subroot_data = root + '/data/Algo/' # Directory root
        self.suffix = self.suffix_func(hyperparameters_config) # self.suffix to make difference between raytune runs (different hyperparameters)
        self.suffix_metrics = self.suffix_func(hyperparameters_config,NNEPPS=True) # self.suffix with NNEPPS information

        # Define PET input dimensions according to input data dimensions
        self.PETImage_shape_str = self.read_input_dim(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.hdr')
        self.PETImage_shape = self.input_dim_str_to_list(self.PETImage_shape_str)

        # Define ROIs for image0 phantom, otherwise it is already done in the database
        if (self.phantom == "image0"):
            self.define_ROI_image0(self.PETImage_shape,self.subroot_data)

        return hyperparameters_config

    def createDirectoryAndConfigFile(self,hyperparameters_config):
        Path(self.subroot+'Block1/' + self.suffix + '/before_eq22').mkdir(parents=True, exist_ok=True) # CASToR path
        Path(self.subroot+'Block1/' + self.suffix + '/during_eq22').mkdir(parents=True, exist_ok=True) # CASToR path
        Path(self.subroot+'Block1/' + self.suffix + '/out_eq22').mkdir(parents=True, exist_ok=True) # CASToR path

        Path(self.subroot+'Images/out_final/'+format(self.experiment)+'/').mkdir(parents=True, exist_ok=True) # Output of the framework (Last output of the DIP)

        Path(self.subroot+'Comparison/' + self.method + '_' + str(self.nb_threads)  + '/' + self.suffix).mkdir(parents=True, exist_ok=True) # CASToR path

        Path(self.subroot+'Block2/checkpoint/'+format(self.experiment)+'/').mkdir(parents=True, exist_ok=True)
        Path(self.subroot+'Block2/out_cnn/'+ format(self.experiment)+'/').mkdir(parents=True, exist_ok=True) # Output of the DIP block every outer iteration
        Path(self.subroot+'Block2/out_cnn/vae').mkdir(parents=True, exist_ok=True) # Output of the DIP block every outer iteration
        Path(self.subroot+'Block2/out_cnn/cnn_metrics/'+ format(self.experiment)+'/').mkdir(parents=True, exist_ok=True) # DIP block metrics
        Path(self.subroot+'Block2/out_cnn/'+ format(self.experiment)+'/like/').mkdir(parents=True, exist_ok=True) # folder for Likelihood calculation (using CASTOR)
        Path(self.subroot+'Block2/x_label/'+format(self.experiment) + '/').mkdir(parents=True, exist_ok=True) # x corrupted - folder

        Path(self.subroot+'Block2/checkpoint/'+format(self.experiment)+'/').mkdir(parents=True, exist_ok=True)
        Path(self.subroot+'Block2/out_cnn/'+ format(self.experiment)+'/').mkdir(parents=True, exist_ok=True)
        Path(self.subroot+'Block2/mu/'+ format(self.experiment)+'/').mkdir(parents=True, exist_ok=True)
        Path(self.subroot+'Block2/out_cnn/cnn_metrics/'+ format(self.experiment)+'/').mkdir(parents=True, exist_ok=True)

        Path(self.subroot_data + 'Data/initialization').mkdir(parents=True, exist_ok=True)
                
        Path(self.subroot_data+'metrics/' + self.method + '/' + self.suffix_metrics).mkdir(parents=True, exist_ok=True) # CASToR path

    def runRayTune(self,config,root,task):
        # Check parameters incompatibility
        self.parametersIncompatibility(config,task)

        # Launch raytune
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

    def parametersIncompatibility(self,config,task):
        # Additional variables needing every values in config
        # Number of replicates 
        self.nb_replicates = config["replicates"]['grid_search'][-1]
        if (task == "show_results_replicates" or task == "show_results"):
            config["replicates"] = tune.grid_search([0]) # Only put 1 value to avoid running same run several times (only for results with several replicates)

        # Do not scale images if network input is uniform of if Gong's method
        if config["input"]['grid_search'] == 'uniform': # Do not standardize or normalize if uniform, otherwise NaNs
            config["scaling"] = "nothing"
        if (len(config["method"]['grid_search']) == 1):
            if config["method"]['grid_search'][0] == 'Gong':
                config["scaling"]['grid_search'] = ["nothing"]
            # Do not use subsets so do not use mlem sequence for ADMM Lim, because of stepsize computation in ADMMLim in CASToR
            if config["method"]['grid_search'][0] == "nested":
                config["mlem_sequence"]['grid_search'] = [False]

        # Remove hyperparameters list
        self.hyperparameters_list = config["hyperparameters"]
        config.pop("hyperparameters", None)
        
        # Remove NNEPPS=False if True is selected for computation
        if (len(config["NNEPPS"]['grid_search']) > 1 and False in config["NNEPPS"]['grid_search'] and 'results' not in task):
            print("No need for computation without NNEPPS")
            config["NNEPPS"]['grid_search'] = [True]

        # Delete hyperparameters specific to others optimizer 
        if (len(config["method"]['grid_search']) == 1):
            if (config["method"]['grid_search'][0] != "AML"):
                config.pop("A_AML", None)
            if (config["method"]['grid_search'][0] != "ADMMLim" and config["method"]['grid_search'][0] != "nested"):
                config.pop("sub_iter_MAP", None)
                config.pop("nb_iter_second_admm", None)
                config.pop("alpha", None)
            if (config["method"]['grid_search'][0] != "nested" and config["method"]['grid_search'][0] != "Gong"):
                config.pop("lr", None)
                config.pop("sub_iter_DIP", None)
                config.pop("opti_DIP", None)
                config.pop("skip_connections", None)
                config.pop("scaling", None)
                config.pop("input", None)
                config.pop("d_DD", None)
                config.pop("k_DD", None)
        else:
            if ('results' not in task):
                raise ValueError("Please do not put several methods at the same time for computation.")
        
        if (task == "show_results_replicates"):
            # List of beta values
            if (len(config["method"]['grid_search']) == 1):
                if (config["method"]['grid_search'][0] == 'ADMMLim'):
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

    def do_everything(self,config,root):
        # Retrieve fixed parameters and hyperparameters from config dictionnary
        fixed_config, hyperparameters_config = self.split_config(config)
        # Initialize variables
        self.initializeGeneralVariables(fixed_config,hyperparameters_config,root)
        self.initializeSpecific(fixed_config,hyperparameters_config,root)
        # Run task computation
        self.runComputation(config,fixed_config,hyperparameters_config,root)
        # Store suffix to retrieve all suffixes in main.py for metrics
        text_file = open(self.subroot_data + 'suffixes_for_last_run_' + fixed_config["method"] + '.txt', "a")
        text_file.write(self.suffix_metrics + "\n")
        text_file.close()



    """"""""""""""""""""" Useful functions """""""""""""""""""""
    def write_hdr(self,subroot,L,subpath,phantom,variable_name='',subroot_output_path='',matrix_type='img'):
        """ write a header for the optimization transfer solution (it's use as CASTOR input)"""
        if (len(L) == 1):
            i = L[0]
            if variable_name != '':
                ref_numbers = format(i) + '_' + variable_name
            else:
                ref_numbers = format(i)
        else:
            i = L[0]
            k = L[1]
            if variable_name != '':
                ref_numbers = format(i) + '_' + format(k) + '_' + variable_name
            else:
                ref_numbers = format(i)
        filename = subroot_output_path + '/'+ subpath + '/' + ref_numbers +'.hdr'
        with open(self.subroot_data + 'Data/MLEM_reco_for_init/' + phantom + '/' + phantom + '_it1.hdr') as f:
            with open(filename, "w") as f1:
                for line in f:
                    if line.strip() == ('!name of data file := ' + phantom + '_it1.img'):
                        f1.write('!name of data file := '+ ref_numbers +'.img')
                        f1.write('\n') 
                    elif line.strip() == ('patient name := ' + phantom + '_it1'):
                        f1.write('patient name := ' + ref_numbers)
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

    def suffix_func(self,hyperparameters_config,NNEPPS=False):
        hyperparameters_config_copy = dict(hyperparameters_config)
        if (NNEPPS==False):
            hyperparameters_config_copy.pop('NNEPPS',None)
        hyperparameters_config_copy.pop('nb_iter_second_admm',None)
        suffix = "config"
        for key, value in hyperparameters_config_copy.items():
            suffix +=  "_" + key[:min(len(key),5)] + "=" + str(value)
        return suffix

    def read_input_dim(self,file_path):
        # Read CASToR header file to retrieve image dimension """
        with open(file_path) as f:
            for line in f:
                if 'matrix size [1]' in line.strip():
                    dim1 = [int(s) for s in line.split() if s.isdigit()][-1]
                if 'matrix size [2]' in line.strip():
                    dim2 = [int(s) for s in line.split() if s.isdigit()][-1]
                if 'matrix size [3]' in line.strip():
                    dim3 = [int(s) for s in line.split() if s.isdigit()][-1]

        # Create variables to store dimensions
        PETImage_shape = (dim1,dim2)
        PETImage_shape_str = str(dim1) + ','+ str(dim2) + ',' + str(dim3)
        if (dim3 > 1):
            raise ValueError("3D not implemented yet")
        print('image shape :', PETImage_shape)
        return PETImage_shape_str

    def input_dim_str_to_list(self,PETImage_shape_str):
        return [int(e.strip()) for e in PETImage_shape_str.split(',')][:-1]

    def fijii_np(self,path,shape,type='<f'):
        """"Transforming raw data to numpy array"""
        file_path=(path)
        dtype = np.dtype(type)
        fid = open(file_path, 'rb')
        data = np.fromfile(fid,dtype)
        image = data.reshape(shape)
        return image

    def norm_imag(self,img):
        """ Normalization of input - output [0..1] and the normalization value for each slide"""
        if (np.max(img) - np.min(img)) != 0:
            return (img - np.min(img)) / (np.max(img) - np.min(img)), np.min(img), np.max(img)
        else:
            return img, np.min(img), np.max(img)

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
        if (np.max(img) - np.min(img)) != 0:
            return img / np.max(img), np.min(img), np.max(img)
        else:
            return img, 0, np.max(img)

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
        """ Standardization of input - output with mean 0 and std 1 for each slide"""
        mean=np.mean(image_corrupt)
        std=np.std(image_corrupt)
        image_center = image_corrupt - mean
        image_corrupt_std = image_center / std
        return image_corrupt_std,mean,std

    def destand_numpy_imag(self,image, mean, std):
        """ Destandardization of input - output with mean 0 and std 1 for each slide"""
        return image * std + mean

    def destand_imag(self,image, mean, std):
        image_np = image.detach().numpy()
        return self.destand_numpy_imag(image_np, mean, std)

    def rescale_imag(self,image_corrupt, scaling):
        """ Scaling of input """
        if (scaling == 'standardization'):
            return self.stand_imag(image_corrupt)
        elif (scaling == 'normalization'):
            return self.norm_positive_imag(image_corrupt)
        elif (scaling == 'positive_normalization'):
            return self.norm_imag(image_corrupt)
        else: # No scaling required
            return image_corrupt, 0, 0

    def descale_imag(self,image, param_scale1, param_scale2, scaling='standardization'):
        """ Descaling of input """
        image_np = image.detach().numpy()
        if (scaling == 'standardization'):
            return self.destand_numpy_imag(image_np, param_scale1, param_scale2)
        elif (scaling == 'normalization'):
            return self.denorm_numpy_imag(image_np, param_scale1, param_scale2)
        elif (scaling == 'positive_normalization'):
            return self.denorm_numpy_positive_imag(image_np, param_scale1, param_scale2)
        else: # No scaling required
            return image_np

    def save_img(self,img,name):
        fp=open(name,'wb')
        img.tofile(fp)
        print('Succesfully save in:', name)

    def find_nan(self,image):
        """ find NaN values on the image"""
        idx = np.argwhere(np.isnan(image))
        print('index with NaN value:',len(idx))
        for i in range(len(idx)):
            image[idx[i,0],idx[i,1]] = 0
        print('index with NaN value:',len(np.argwhere(np.isnan(image))))
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

        cold_mask = np.zeros(PETImage_shape, dtype='<f')
        tumor_mask = np.zeros(PETImage_shape, dtype='<f')
        phantom_mask = np.zeros(PETImage_shape, dtype='<f')
        bkg_mask = np.zeros(PETImage_shape, dtype='<f')

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

    def write_image_tensorboard(self,writer,image,name,suffix,image_gt,i=0,full_contrast=False):
        # Creating matplotlib figure with colorbar
        plt.figure()
        if (len(image.shape) != 2):
            print('image is ' + str(len(image.shape)) + 'D, plotting only 2D slice')
            image = image[:,:,0]
        if (full_contrast):
            plt.imshow(image, cmap='gray_r',vmin=np.min(image),vmax=np.max(image)) # Showing each image with maximum contrast  
        else:
            plt.imshow(image, cmap='gray_r',vmin=0,vmax=1.25*np.max(image_gt)) # Showing all images with same contrast
        plt.colorbar()
        #plt.axis('off')

        # Saving this figure locally
        Path(self.subroot + 'Images/tmp/' + suffix).mkdir(parents=True, exist_ok=True)
        #os.system('rm -rf' + self.subroot + 'Images/tmp/' + suffix + '/*')
        print(self.subroot + 'Images/tmp/' + suffix + '/' + name + '_' + str(i) + '.png')
        plt.savefig(self.subroot + 'Images/tmp/' + suffix + '/' + name + '_' + str(i) + '.png')
        from textwrap import wrap
        wrapped_title = "\n".join(wrap(suffix, 50))
        plt.title(wrapped_title,fontsize=12)
        # Adding this figure to tensorboard
        writer.add_figure(name,plt.gcf(),global_step=i,close=True)# for videos, using slider to change image with global_step

    def castor_command_line_func(self,method,phantom,replicate,PETImage_shape_str,rho,alpha,i,k=0):
        if (method == 'nested'):
            if (i==0): # For first iteration, put rho to zero
                if (k!=-1): # For first iteration, do not put both rho and alpha to zero
                    rho = 0
            if (k==-1): # For first iteration, put alpha to zero (small value to be accepted by CASToR)
                alpha = 0

        castor_command_line = self.castor_admm_command_line(self.subroot_data, method, PETImage_shape_str, alpha, rho, phantom, replicate)

        return castor_command_line

    def castor_admm_command_line(self, subroot, method, PETImage_shape_str, alpha, rho, phantom, replicates, only_Lim=False, pnlt=''):
        
        # castor-recon command line
        header_file = ' -df ' + subroot + 'Data/database_v2/' + phantom + '/data' + phantom[-1] + '_' + str(replicates) + '/data' + phantom[-1] + '_' + str(replicates) + '.cdh' # PET data path

        executable = 'castor-recon'
        dim = ' -dim ' + PETImage_shape_str
        vox = ' -vox 4,4,4'
        vb = ' -vb 3'
        th = ' -th ' + str(self.nb_threads) # set it to 1, as multithreading does not work for now with ADMMLim optimizer
        proj = ' -proj incrementalSiddon'

        psf = ' -conv gaussian,4,1,3.5::psf'

        # Command line for calculating the Likelihood
        #opti_like = ' -opti-fom'
        opti_like = ''
        
        if (method == 'nested' or method == 'ADMMLim'):
            if (rho == 0): # Special case where we do not want to penalize reconstruction (not taking into account network output)
                # Seg fault in CASToR...
                pnlt = ''
                pnlt_beta = ''
                # Not clean, but works to put rho == 0 in CASToR
                if (~only_Lim): # DIP + ADMM reconstruction, so choose DIP_ADMM penalty from CASToR
                    pnlt = ' -pnlt DIP_ADMM'
                pnlt_beta = ' -pnlt-beta ' + str(rho)
            else:      
                if (~only_Lim): # DIP + ADMM reconstruction, so choose DIP_ADMM penalty from CASToR
                    pnlt = ' -pnlt DIP_ADMM'
                pnlt_beta = ' -pnlt-beta ' + str(rho)
            if (alpha == 0): # Special case where we only want to fit network output (when v has not been initialized with data)
                alpha = 1E-10 # Do not put 0, otherwise CASToR will not work
            opti = ' -opti ADMMLim' + ',' + str(alpha)
        
        elif (method == 'Gong'):
            opti = ' -opti OPTITR'
            pnlt = ' -pnlt OPTITR'
            pnlt_beta = ' -pnlt-beta ' + str(rho)

        pnlt = ''
        castor_command_line_x = executable + dim + vox + header_file + vb + th + proj + opti + opti_like + pnlt + pnlt_beta + psf
        return castor_command_line_x
