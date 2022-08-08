## Python libraries

# Useful
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
    def __init__(self,config):
        print('__init__')

    def runComputation(self,config,fixed_config,hyperparameters_config,root):
        """ Implement me! """
        pass

    def initializeSpecific(self,fixed_config,hyperparameters_config,root):
        self.createDirectoryAndConfigFile(hyperparameters_config)

        # Specific hyperparameters for reconstruction module (Do it here to have raytune hyperparameters_config hyperparameters selection)
        if (fixed_config["method"] != "MLEM" and fixed_config["method"] != "AML"):
            self.rho = hyperparameters_config["rho"]
        else:
            self.rho = 0
        if ('ADMMLim' in fixed_config["method"] or fixed_config["method"] == "nested" or fixed_config["method"] == "Gong"):
            self.nb_inner_iteration = hyperparameters_config["nb_inner_iteration"]
            print(self.nb_inner_iteration)
            print('aaaaaaaaaaaaaaaaaaaa')
            if (fixed_config["method"] == "Gong"):
                self.alpha = None
            else:
                self.alpha = hyperparameters_config["alpha"]
        self.image_init_path_without_extension = fixed_config["image_init_path_without_extension"]

        # Ininitializing DIP output and first image x with f_init and image_init
        if (self.method == "nested"): # Nested needs 1 to not add any prior information at the beginning, and to initialize x computation to uniform with 1
            self.f_init = np.ones((self.PETImage_shape[0],self.PETImage_shape[1],self.PETImage_shape[2]))
        elif (self.method == "Gong"): # Gong initialization with 60th iteration of MLEM (normally, DIP trained with this image as label...)
            #self.f_init = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'BSREM_it30_REF_cropped.img',shape=(self.PETImage_shape),type='<f')
            self.f_init = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'MLEM_it60_REF_cropped.img',shape=(self.PETImage_shape),type='<f')

        # Initialize and save mu variable from ADMM
        if (self.method == "nested" or self.method == "Gong"):
            self.mu = 0* np.ones((self.PETImage_shape[0], self.PETImage_shape[1], self.PETImage_shape[2]))
            self.save_img(self.mu,self.subroot+'Block2/mu/'+ format(self.experiment)+'/mu_' + format(-1) + self.suffix + '.img')

        # Launch short MLEM reconstruction
        path_mlem_init = self.subroot_data + 'Data/MLEM_reco_for_init/' + self.phantom
        my_file = Path(path_mlem_init + '/' + self.phantom + '/' + self.phantom + '_it1.img')
        if (~my_file.is_file()):
            print("self.nb_replicates",self.nb_replicates)
            if (self.nb_replicates == 1):
                header_file = ' -df ' + self.subroot_data + 'Data/database_v2/' + self.phantom + '/data' + self.phantom[-1] + '/data' + self.phantom[-1] + '.cdh' # PET data path
            else:
                header_file = ' -df ' + self.subroot_data + 'Data/database_v2/' + self.phantom + '/data' + self.phantom[-1] + '_' + str(fixed_config["replicates"]) + '/data' + self.phantom[-1] + '_' + str(fixed_config["replicates"]) + '.cdh' # PET data path
            executable = 'castor-recon'
            optimizer = 'MLEM'
            output_path = ' -dout ' + path_mlem_init # Output path for CASTOR framework
            dim = ' -dim ' + self.PETImage_shape_str
            vox = ' -vox 4,4,4'
            vb = ' -vb 3'
            it = ' -it 1:1'
            opti = ' -opti ' + optimizer
            os.system(executable + dim + vox + output_path + header_file + vb + it + opti) # + ' -fov-out 95')

    def castor_reconstruction(self,writer, i, subroot, nb_inner_iteration, experiment, hyperparameters_config, method, phantom, replicate, suffix, image_gt, f, mu, PETImage_shape, PETImage_shape_str, rho, alpha, image_init_path_without_extension):
        start_time_block1 = time.time()
        mlem_sequence = hyperparameters_config['mlem_sequence']

        # Save image f-mu in .img and .hdr format - block 1
        subroot_output_path = (subroot + 'Block1/' + suffix)
        path_before_eq_22 = (subroot_output_path + '/before_eq22/')
        self.save_img(f-mu, path_before_eq_22 + format(i) + '_f_mu.img')
        self.write_hdr(self.subroot_data,[i],'before_eq22',phantom,'f_mu',subroot_output_path)
        f_mu_for_penalty = ' -multimodal ' + subroot_output_path + '/before_eq22/' + format(i) + '_f_mu' + '.hdr'        
        subdir = 'during_eq22'

        # Initialization
        if (method == 'nested'):            
            x = self.ADMMLim_general(hyperparameters_config, i, subdir, subroot_output_path, f_mu_for_penalty,writer,image_gt)
        elif (method == 'Gong'):

            # Choose number of argmax iteration for (second) x computation
            if (mlem_sequence):
                #it = ' -it 2:56,4:42,6:36,4:28,4:21,2:14,2:7,2:4,2:2,2:1' # large subsets sequence to approximate argmax, too many subsets for 2D, but maybe ok for 3D
                it = ' -it 16:28,4:21,2:14,2:7,2:4,2:2,2:1' # large subsets sequence to approximate argmax, 2D
            else:
                it = ' -it ' + str(nb_inner_iteration) + ':1' # Only 2 iterations (Gong) to compute argmax, if we estimate it is an enough precise approximation. Only 1 according to conjugate gradient in Lim et al.

            # Define command line to run OPTITR with CASToR
            castor_command_line_x = self.castor_common_command_line(self.subroot_data, self.PETImage_shape_str, self.phantom, self.replicate) + self.castor_opti_and_penalty(self.method, self.penalty, self.rho, i)
            # Initialize image
            if (i == -1):   # choose initial image for CASToR reconstruction
                initialimage = ' -img ' + self.subroot_data + 'Data/initialization/' + image_init_path_without_extension + '.hdr' if image_init_path_without_extension != "" else '' # initializing CASToR PLL reconstruction with image_init or with CASToR default values
            elif (i >= 0):
                #initialimage = ' -img ' + subroot_output_path + '/' + subdir + '/' + format(i-1) + '_' + format(hyperparameters_config["nb_outer_iteration"]) + '_it' + str(hyperparameters_config["nb_inner_iteration"]) + '.hdr'
                # Trying to initialize OPTITR
                #initialimage = ' -img ' + self.subroot_data + 'Data/initialization/' + 'BSREM_it30_REF_cropped.hdr'
                initialimage = ' -img ' + self.subroot_data + 'Data/initialization/' + '1_im_value_cropped.hdr'
                #initialimage = ' -img ' + subroot_output_path + '/' + subdir + '/' + format(i-1) + '_it' + str(hyperparameters_config["nb_inner_iteration"]) + '.hdr'

            base_name_k_next = format(i)
            full_output_path_k_next = subroot_output_path + '/' + subdir + '/' + base_name_k_next
            x_reconstruction_command_line = castor_command_line_x + ' -fout ' + full_output_path_k_next + it + f_mu_for_penalty + initialimage            
            print(x_reconstruction_command_line)
            os.system(x_reconstruction_command_line)

            if (mlem_sequence):
                x = self.fijii_np(full_output_path_k_next + '_it30.img', shape=(PETImage_shape))
            else:
                x = self.fijii_np(full_output_path_k_next + '_it' + str(hyperparameters_config["nb_inner_iteration"]) + '.img', shape=(PETImage_shape))

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
        name=(subroot+'Block2/x_label/'+format(experiment) + '/' + format(i) +'_x_label' + suffix + '.img')
        self.save_img(x_label, name)

        return x_label

    def compute_x_v_u_ADMM(self,x_reconstruction_command_line,subdir,i,phantom,subroot_output_path,subroot):
        # Compute x,u,v
        os.system(x_reconstruction_command_line)
        # Write v and u hdr files
        self.write_hdr(subroot,[i],subdir,phantom,'v',subroot_output_path=subroot_output_path,matrix_type='sino')
        self.write_hdr(subroot,[i],subdir,phantom,'u',subroot_output_path=subroot_output_path,matrix_type='sino')


    def ADMMLim_general(self, hyperparameters_config, i, subdir, subroot_output_path,f_mu_for_penalty,writer=None,image_gt=None):
        
        k_init = -1
        full_output_path_k_next = subroot_output_path + '/' + subdir + '/' + format(i) + '_' + format(k_init+1)

        # Define command line to run ADMM with CASToR, to compute v^0
        if (i == 0):   # choose initial image for CASToR reconstruction
            x_for_init_v = ' -img ' + self.subroot_data + 'Data/initialization/' + self.image_init_path_without_extension + '.hdr' if self.image_init_path_without_extension != "" else '' # initializing CASToR PLL reconstruction with image_init or with CASToR default values
            #x_for_init_v = ' -img ' + self.subroot_data + 'Data/initialization/' + '1_im_value' + '.hdr' if image_init_path_without_extension != "" else '' # initializing CASToR PLL reconstruction with image_init or with CASToR default values
        elif (i >= 1):
            x_for_init_v = ' -img ' + subroot_output_path + '/' + subdir + '/' +format(i-1) + '_' + format(hyperparameters_config["nb_outer_iteration"]) + '_it1.hdr'
                        

        if (self.method == "nested"):
            self.post_smoothing = 0
        castor_command_line_x = self.castor_common_command_line(self.subroot_data, self.PETImage_shape_str, self.phantom, self.replicate, self.post_smoothing)
        
        '''
        print('vvvvvvvvvvv0000000000')
        x_reconstruction_command_line = castor_command_line_x + self.castor_opti_and_penalty(self.method, self.penalty, self.rho, i) + ' -fout ' + full_output_path_k_next + ' -it 1:1' + x_for_init_v + f_mu_for_penalty # we need f-mu so that ADMM optimizer works, even if we will not use it...
        print(x_reconstruction_command_line)
        self.compute_x_v_u_ADMM(x_reconstruction_command_line,subdir,i-1,self.phantom,subroot_output_path,self.subroot_data)
        # Copy u^-1 coming from CASToR to v^0
        copy(full_output_path_k_next + '_u.img', full_output_path_k_next + '_v.img')
        self.write_hdr(self.subroot_data,[i,k_init+1],subdir,self.phantom,'v',subroot_output_path,matrix_type='sino')
        
        # Then initialize u^0 (u^-1 in CASToR)
        if (i == 0):   # choose initial image for CASToR reconstruction
            copy(self.subroot_data + 'Data/initialization/0_sino_value.img', full_output_path_k_next + '_u.img')
        elif (i >= 1):
            copy(subroot_output_path + '/' + subdir + '/' +format(i-1) + '_' + format(hyperparameters_config["nb_outer_iteration"]) + '_u.img', full_output_path_k_next + '_u.img')
        self.write_hdr(self.subroot_data,[i,k_init+1],subdir,self.phantom,'u',subroot_output_path,matrix_type='sino')
        '''

        print("xxxxxxxxxxxxxxxxxxxxxxx")
        if (i == 0):   # choose initial image for CASToR reconstruction
            initialimage = ' -img ' + self.subroot_data + 'Data/initialization/' + self.image_init_path_without_extension + '.hdr' if self.image_init_path_without_extension != "" else '' # initializing CASToR PLL reconstruction with image_init or with CASToR default values
        elif (i >= 1):
            initialimage = ' -img ' + subroot_output_path + '/' + subdir + '/' +format(i-1) + '_' + format(hyperparameters_config["nb_outer_iteration"]) + '_it' + str(hyperparameters_config["nb_inner_iteration"]) + '.hdr'
            # Trying to initialize ADMMLim
            #initialimage = ' -img ' + self.subroot_data + 'Data/initialization/' + 'BSREM_it30_REF_cropped.hdr'
            initialimage = ' -img ' + self.subroot_data + 'Data/initialization/' + '1_im_value_cropped.hdr'
    
        base_name_i = format(i)
        base_name_i_next = format(i+1)
        full_output_path_i = subroot_output_path + '/' + subdir + '/' + base_name_i
        full_output_path_i_next = subroot_output_path + '/' + subdir + '/' + base_name_i_next
        v_for_additional_data = ' -additional-data ' + full_output_path_i + '_v.hdr'
        u_for_additional_data = ' -additional-data ' + full_output_path_i + '_u.hdr'

        if ('ADMMLim' in self.method):
            # Compute one ADMM iteration (x, v, u)
            if (self.post_smoothing): # Apply post smoothing for vizualization
                #conv = ''
                conv = ' -conv gaussian,' + str(hyperparameters_config["post_smoothing"]) + ',1,3.5::post'
            else:
                conv = ''
        else:
            conv = ''

        # Number of iterations from config dictionnary
        it = ' -it ' + str(hyperparameters_config["nb_outer_iteration"]) + ':1'  # 1 subset

        x_reconstruction_command_line = castor_command_line_x \
                                        + self.castor_opti_and_penalty(self.method, self.penalty, self.rho) \
                                        + ' -fout ' + full_output_path_i_next + it \
                                        + initialimage + f_mu_for_penalty \
                                        + conv # we need f-mu so that ADMM optimizer works, even if we will not use it...

        print(x_reconstruction_command_line)
        self.compute_x_v_u_ADMM(x_reconstruction_command_line, subdir, i, self.phantom, subroot_output_path, self.subroot_data)

        # -- AdaptiveRho ---- AdaptiveRho ---- AdaptiveRho ---- AdaptiveRho ---- AdaptiveRho ---- AdaptiveRho --
        '''
        path_adaptive = subroot_output_path + '/' + subdir + '/' + format(i) + '_' + format(
            k+1) + '_adaptive.log'
        theLog = pd.read_table(path_adaptive)
        theAdaptiveAlphaRow = theLog.loc[[0]]
        theAdaptiveAlphaRowArray = np.array(theAdaptiveAlphaRow)
        theAdaptiveAlphaRowString = theAdaptiveAlphaRowArray[0, 0]
        adaptiveAlpha = float(theAdaptiveAlphaRowString)
        print('*************************************************************************************************************************************', 'k+1 =', k+1, 'adaptive alpha =', adaptiveAlpha)
        self.alpha = adaptiveAlpha
        castor_command_line_x = self.castor_common_command_line(self.subroot_data, self.PETImage_shape_str,
                                                                self.phantom, self.replicate,
                                                                self.post_smoothing)
        '''







        if (self.method == "nested"):
            x = self.fijii_np(full_output_path_k_next + '_it1.img', shape=(self.PETImage_shape))
            
            self.write_image_tensorboard(writer,x,"x in ADMM1 over iterations",self.suffix,500, 0+1+i*hyperparameters_config["nb_outer_iteration"]) # Showing all corrupted images with same contrast to compare them together
            self.write_image_tensorboard(writer,x,"x in ADMM1 over iterations(FULL CONTRAST)",self.suffix,500, 0+1+i*hyperparameters_config["nb_outer_iteration"],full_contrast=True) # Showing all corrupted images with same contrast to compare them together

        if (self.method == "nested"):
            return x
