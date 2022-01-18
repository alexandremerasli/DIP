## Python libraries

# Useful
import os
from pathlib import Path

# Math
import numpy as np

# Local files to import
from utils.utils_func import *
from vGeneral import vGeneral

import abc
class vReconstruction(vGeneral):
    @abc.abstractmethod
    def __init__(self,config,root):
        print('__init__')

    def runComputation(self,config,hyperparameters_config,root):
        """ Implement me! """
        pass

    def initializeSpecific(self,hyperparameters_config,root):
        self.createDirectoryAndConfigFile(hyperparameters_config)

        # Specific hyperparameters for reconstruction module (Do it here to have raytune hyperparameters_config hyperparameters selection)
        self.rho = hyperparameters_config["rho"]
        self.alpha = hyperparameters_config["alpha"]
        self.sub_iter_MAP = hyperparameters_config["sub_iter_MAP"]

        # Initialize and save mu variable from ADMM
        self.mu = 0* np.ones((self.PETImage_shape[0], self.PETImage_shape[1]), dtype='<f')
        print("self.suffix")
        print(self.suffix)
        save_img(self.mu,self.subroot+'Block2/mu/'+ format(self.test)+'/mu_' + format(-1) + self.suffix + '.img')

        # Ininitializing DIP output and first image x with f_init and image_init
        if (self.method == "nested"): # Nested needs 1 to not add any prior information at the beginning, and to initialize x computation to uniform with 1
            self.f_init = np.ones((self.PETImage_shape[0],self.PETImage_shape[1]), dtype='<f')
            self.image_init_path_without_extension = '1_im_value_cropped'
            # self.image_init_path_without_extension = 'BSREM_it30_REF_cropped'
        elif (self.method == "Gong"): # Gong initialization with 60th iteration of MLEM (normally, DIP trained with this image as label...)
            self.f_init = fijii_np(self.subroot + 'Data/initialization/' + 'BSREM_it30_REF_cropped.img',shape=(self.PETImage_shape))
            #self.f_init = fijii_np(self.subroot + 'Data/initialization/' + 'MLEM_it60_REF_cropped.img',shape=(self.PETImage_shape))
            self.image_init_path_without_extension = '1_im_value_cropped' # OPTITR initialization, so first image in MLEM (1 iteration) computation. Not written in Gong, but perhaps not giving any prior information

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