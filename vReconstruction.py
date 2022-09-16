## Python libraries

# Useful
from genericpath import isfile
import os
from pathlib import Path
import time
from shutil import copy

# Math
import numpy as np
import pandas as pd

# Local files to import
from vGeneral import vGeneral

import abc
class vReconstruction(vGeneral):
    @abc.abstractmethod
    def __init__(self,config, *args, **kwargs):
        print('__init__')

    def runComputation(self,config,root):
        """ Implement me! """
        pass

    def initializeSpecific(self,config,root, *args, **kwargs):
        self.createDirectoryAndConfigFile(config)

        # Specific hyperparameters for reconstruction module (Do it here to have raytune config hyperparameters selection)
        if (config["method"] != "MLEM" and config["method"] != "OSEM" and config["method"] != "AML"):
            self.rho = config["rho"]
        else:
            self.rho = 0
        if ('ADMMLim' in config["method"] or config["method"] == "nested" or config["method"] == "Gong"):
            if (config["method"] != "ADMMLim"):
                self.unnested_1st_global_iter = config["unnested_1st_global_iter"]
            else:
                self.unnested_1st_global_iter = None
            if (config["method"] == "Gong"):
                self.alpha = None
            else:
                self.alpha = config["alpha"]
                self.adaptive_parameters = config["adaptive_parameters"]
                if (self.adaptive_parameters == "nothing"): # set mu, tau, xi to any values, there will not be used in CASToR
                    self.mu_adaptive = np.NaN
                    self.tau = np.NaN
                    self.xi = np.NaN
                else:
                    self.mu_adaptive = config["mu_adaptive"]
                    self.tau = config["tau"]
                    self.xi = config["xi"]
        self.image_init_path_without_extension = config["image_init_path_without_extension"]
        self.tensorboard = config["tensorboard"]

        # Initialize and save mu variable from ADMM
        if (self.method == "nested" or self.method == "Gong"):
            self.mu = 0* np.ones((self.PETImage_shape[0], self.PETImage_shape[1], self.PETImage_shape[2]))
            self.save_img(self.mu,self.subroot+'Block2/' + self.suffix + '/mu/'+ format(self.experiment)+'/mu_' + format(-1) + self.suffix + '.img')

        '''
        # Launch short MLEM reconstruction
        path_mlem_init = self.subroot_data + 'Data/MLEM_reco_for_init/' + self.phantom
        my_file = Path(path_mlem_init + '/' + self.phantom + '/' + self.phantom + '_it1.img')
        if (not my_file.is_file()):
            print("self.nb_replicates",self.nb_replicates)
            if (self.nb_replicates == 1):
                header_file = ' -df ' + self.subroot_data + 'Data/database_v2/' + self.phantom + '/data' + self.phantom[5:] + '/data' + self.phantom[5:] + '.cdh' # PET data path
            else:
                header_file = ' -df ' + self.subroot_data + 'Data/database_v2/' + self.phantom + '/data' + self.phantom[5:] + '_' + str(config["replicates"]) + '/data' + self.phantom[5:] + '_' + str(config["replicates"]) + '.cdh' # PET data path
            executable = 'castor-recon'
            optimizer = 'MLEM'
            output_path = ' -dout ' + path_mlem_init # Output path for CASTOR framework
            dim = ' -dim ' + self.PETImage_shape_str
            vox = ' -vox 4,4,4'
            vb = ' -vb 3'
            it = ' -it 1:1'
            opti = ' -opti ' + optimizer
            th = ' -th ' + str(self.nb_threads) # must be set to 1 for ADMMLim, as multithreading does not work for now with ADMMLim optimizer
            print(executable + dim + vox + output_path + header_file + vb + it + opti + th)
            os.system(executable + dim + vox + output_path + header_file + vb + it + opti + th) # + ' -fov-out 95')
        '''
        
    def castor_reconstruction(self,writer, i, subroot, nb_outer_iteration, experiment, config, method, phantom, replicate, suffix, image_gt, f, mu, PETImage_shape, PETImage_shape_str, alpha, image_init_path_without_extension):
        start_time_block1 = time.time()
        mlem_sequence = config['mlem_sequence']

        # Save image f-mu in .img and .hdr format - block 1
        subroot_output_path = (subroot + 'Block1/' + suffix)
        path_before_eq_22 = (subroot_output_path + '/before_eq22/')
        self.save_img(f-mu, path_before_eq_22 + format(i) + '_f_mu.img')
        self.write_hdr(self.subroot_data,[i],'before_eq22',phantom,'f_mu',subroot_output_path)
        f_mu_for_penalty = ' -multimodal ' + subroot_output_path + '/before_eq22/' + format(i) + '_f_mu' + '.hdr' # Will be removed if first global iteration and unnested_1st_global_iter (rho == 0)
        subdir = 'during_eq22'

        # Initialization
        if (method == 'nested'):            
            x = self.ADMMLim_general(config, i, subdir, subroot_output_path, f_mu_for_penalty,writer,image_gt)
        elif (method == 'Gong'):

            # Choose number of argmax iteration for (second) x computation
            if (mlem_sequence):
                #it = ' -it 2:56,4:42,6:36,4:28,4:21,2:14,2:7,2:4,2:2,2:1' # large subsets sequence to approximate argmax, too many subsets for 2D, but maybe ok for 3D
                it = ' -it 16:28,4:21,2:14,2:7,2:4,2:2,2:1' # large subsets sequence to approximate argmax, 2D
            else:
                it = ' -it ' + str(nb_outer_iteration) + ':1' # Only 2 iterations (Gong) to compute argmax, if we estimate it is an enough precise approximation. Only 1 according to conjugate gradient in Lim et al.

            # Define command line to run OPTITR with CASToR
            castor_command_line_x = self.castor_common_command_line(self.subroot_data, self.PETImage_shape_str, self.phantom, self.replicate) + self.castor_opti_and_penalty(self.method, self.penalty, self.rho, i, self.unnested_1st_global_iter)
            # Initialize image
            if (i == -1):   # choose initial image for CASToR reconstruction
                initialimage = ' -img ' + self.subroot_data + 'Data/initialization/' + image_init_path_without_extension + '.hdr' if image_init_path_without_extension != "" else '' # initializing CASToR PLL reconstruction with image_init or with CASToR default values
            elif (i >= 0):
                #initialimage = ' -img ' + subroot_output_path + '/' + subdir + '/' + format(i-1) + '_' + format(config["nb_outer_iteration"]) + '_it' + str(config["nb_inner_iteration"]) + '.hdr'
                # Trying to initialize OPTITR
                #initialimage = ' -img ' + self.subroot_data + 'Data/initialization/' + 'BSREM_it30_REF_cropped.hdr'
                initialimage = ' -img ' + self.subroot_data + 'Data/initialization/' + '1_im_value_cropped.hdr'
                #initialimage = ' -img ' + subroot_output_path + '/' + subdir + '/' + format(i-1) + '_it' + str(config["nb_inner_iteration"]) + '.hdr'

            base_name_i = format(i)
            full_output_path_i = subroot_output_path + '/' + subdir + '/' + base_name_i
            x_reconstruction_command_line = castor_command_line_x + ' -fout ' + full_output_path_i + it + f_mu_for_penalty + initialimage            
            print(x_reconstruction_command_line)
            os.system(x_reconstruction_command_line)

            if (mlem_sequence):
                x = self.fijii_np(full_output_path_i + '_it30.img', shape=(PETImage_shape))
            else:
                x = self.fijii_np(full_output_path_i + '_it' + str(config["nb_outer_iteration"]) + '.img', shape=(PETImage_shape))
            print(full_output_path_i + '_it' + str(config["nb_outer_iteration"]) + '.img')

            self.write_image_tensorboard(writer,x,"x after optimization transfer over iterations",suffix,image_gt, i) # Showing all corrupted images with same contrast to compare them together
            self.write_image_tensorboard(writer,x,"x after optimization transfer over iterations (FULL CONTRAST)",suffix,image_gt, i,full_contrast=True) # Showing all corrupted images with same contrast to compare them together

        print("--- %s seconds - second ADMM (CASToR) iteration ---" % (time.time() - start_time_block1))

        # Save image x in .img and .hdr format - block 1
        name = (subroot+'Block1/' + suffix + '/out_eq22/' + format(i) + '.img')
        self.save_img(x, name)
        self.write_hdr(subroot,[i],'out_eq22',phantom,'',subroot_output_path)

        # Save x_label for load into block 2 - CNN as corrupted image (x_label)
        x_label = x + mu
        # Save x_label in .img and .hdr format
        name=(subroot+'Block2/' + self.suffix + '/x_label/'+format(experiment) + '/' + format(i) +'_x_label' + suffix + '.img')
        self.save_img(x_label, name)

        return x_label

    def compute_x_v_u_ADMM(self,x_reconstruction_command_line,subdir,i,phantom,subroot_output_path,subroot):
        # Compute x,u,v
        os.system(x_reconstruction_command_line)
        # Write v and u hdr files
        self.write_hdr(subroot,[i],subdir,phantom,'v',subroot_output_path=subroot_output_path,matrix_type='sino')
        self.write_hdr(subroot,[i],subdir,phantom,'u',subroot_output_path=subroot_output_path,matrix_type='sino')


    def ADMMLim_general(self, config, i, subdir, subroot_output_path,f_mu_for_penalty,writer=None,image_gt=None):
        if (self.method == "nested"):
            self.post_smoothing = 0
        castor_command_line_x = self.castor_common_command_line(self.subroot_data, self.PETImage_shape_str, self.phantom, self.replicate, self.post_smoothing)

        if (i == 0):   # choose initial image for CASToR reconstruction
            initialimage = ' -img ' + self.subroot_data + 'Data/initialization/' + self.image_init_path_without_extension + '.hdr' if self.image_init_path_without_extension != "" else '' # initializing CASToR PLL reconstruction with image_init or with CASToR default values
            # Trying to initialize ADMMLim
            #initialimage = ' -img ' + self.subroot_data + 'Data/initialization/' + 'BSREM_it30_REF_cropped.hdr'
            #initialimage = ' -img ' + self.subroot_data + 'Data/initialization/' + '1_im_value_cropped.hdr'
        elif (i >= 1):
            #initialimage = ' -img ' + subroot_output_path + '/' + subdir + '/' +format(i-1) + '_' + format(config["nb_outer_iteration"]) + '_it' + str(config["nb_inner_iteration"]) + '.hdr'
            #initialimage = ' -img ' + subroot_output_path + '/' + subdir + '/' +format(i-1) + '_it' + str(config["nb_outer_iteration"]) + '.hdr'
            # Last image for next global iteration
            initialimage = ' -img ' + subroot_output_path + '/' + 'out_eq22' + '/' +format(i-1) + '.hdr'
            # Uniform image for next global iteration
            #initialimage = ' -img ' + self.subroot_data + 'Data/initialization/' + self.image_init_path_without_extension + '.hdr' if self.image_init_path_without_extension != "" else '' # initializing CASToR PLL reconstruction with image_init or with CASToR default values

    
        base_name_i = format(i)
        full_output_path_i = subroot_output_path + '/' + subdir + '/' + base_name_i

        if ('ADMMLim' in self.method):
            # Compute one ADMM iteration (x, v, u)
            if (self.post_smoothing): # Apply post smoothing for vizualization
                if ("1" in self.PETImage_shape_str.split(',')): # 2D
                    conv = ' -conv gaussian,' + str(self.post_smoothing) + ',1,3.5::post'
                else: # 3D
                    conv = ' -conv gaussian,' + str(self.post_smoothing) + ',' + str(self.post_smoothing) + ',3.5::post' # isotropic post smoothing
            else:
                conv = ''
        else:
            conv = ''

        # Optimizer and penalty in command line, change rho if first global iteration and unnested_1st_global_iter
        opti_and_penalty = self.castor_opti_and_penalty(self.method, self.penalty, self.rho, i, self.unnested_1st_global_iter)
        # If rho is 0, remove f_mu_for_penalty
        if ((self.rho == 0) or (i==0 and self.unnested_1st_global_iter) or (i==-1 and not self.unnested_1st_global_iter)): # For first iteration, put rho to zero
            f_mu_for_penalty = ''

        # Number of iterations from config dictionnary
        it = ' -it ' + str(config["nb_outer_iteration"]) + ':1'  # 1 subset

        x_reconstruction_command_line = castor_command_line_x \
                                        + opti_and_penalty \
                                        + ' -fout ' + full_output_path_i + it \
                                        + initialimage + f_mu_for_penalty \
                                        + conv # we need f-mu so that ADMM optimizer works, even if we will not use it...

        print(x_reconstruction_command_line)
        self.compute_x_v_u_ADMM(x_reconstruction_command_line, subdir, i, self.phantom, subroot_output_path, self.subroot_data)

        #'''
        # -- AdaptiveRho ---- AdaptiveRho ---- AdaptiveRho ---- AdaptiveRho ---- AdaptiveRho ---- AdaptiveRho --


        self.path_stopping_criterion = subroot_output_path + '/' + subdir + '/' + format(i) + '_adaptive_stopping_criteria.log'
        if(isfile(self.path_stopping_criterion)):
            theLog = pd.read_table(self.path_stopping_criterion)
            finalOuterIterRow = theLog.loc[[0]]

            finalOuterIterRowArray = np.array(finalOuterIterRow)
            finalOuterIterRowString = finalOuterIterRowArray[0, 0]
            finalOuterIter = int(finalOuterIterRowString)
            print("finalOuterIter",finalOuterIter)
        else:
            finalOuterIter = config["nb_outer_iteration"]

        for outer_it in range(1,finalOuterIter+1):
            path_adaptive = subroot_output_path + '/' + subdir + '/' + format(i) + '_adaptive_it' + format(outer_it) + '.log'
            theLog = pd.read_table(path_adaptive)
            relativePrimalResidualRow = theLog.loc[[6]]
            relativePrimalResidualRowArray = np.array(relativePrimalResidualRow)
            relativePrimalResidualRowString = relativePrimalResidualRowArray[0, 0]
            relativePrimalResidual = float(relativePrimalResidualRowString)
            print("relPrimal",relativePrimalResidual)

        for outer_it in range(1,finalOuterIter+1):
            path_adaptive = subroot_output_path + '/' + subdir + '/' + format(i) + '_adaptive_it' + format(outer_it) + '.log'
            theLog = pd.read_table(path_adaptive)
            relativeDualResidualRow = theLog.loc[[8]]
            relativeDualResidualRowArray = np.array(relativeDualResidualRow)
            relativeDualResidualRowString = relativeDualResidualRowArray[0, 0]
            relativeDualResidual = float(relativeDualResidualRowString)
            print("relDual",relativeDualResidual)
        #'''
        
        if (self.method == "nested" and self.tensorboard):
            for k in range(1,finalOuterIter,max(finalOuterIter//10,1)):
                x = self.fijii_np(full_output_path_i + '_it' + str(k) + '.img', shape=(self.PETImage_shape))
                self.write_image_tensorboard(writer,x,"x in ADMM1 over iterations",self.suffix,500, 0+k+i*config["nb_outer_iteration"]) # Showing all corrupted images with same contrast to compare them together
                self.write_image_tensorboard(writer,x,"x in ADMM1 over iterations(FULL CONTRAST)",self.suffix,500, 0+k+i*config["nb_outer_iteration"],full_contrast=True) # Showing all corrupted images with same contrast to compare them together
            return x
