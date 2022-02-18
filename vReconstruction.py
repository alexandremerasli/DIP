## Python libraries

# Useful
import os
from pathlib import Path
import time
from shutil import copy

# Math
import numpy as np

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
        self.rho = hyperparameters_config["rho"]
        self.alpha = hyperparameters_config["alpha"]
        self.sub_iter_MAP = hyperparameters_config["sub_iter_MAP"]
        self.image_init_path_without_extension = fixed_config["image_init_path_without_extension"]

        # Ininitializing DIP output and first image x with f_init and image_init
        if (self.method == "nested"): # Nested needs 1 to not add any prior information at the beginning, and to initialize x computation to uniform with 1
            self.f_init = np.ones((self.PETImage_shape[0],self.PETImage_shape[1]), dtype='<f')
        elif (self.method == "Gong"): # Gong initialization with 60th iteration of MLEM (normally, DIP trained with this image as label...)
            #self.f_init = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'BSREM_it30_REF_cropped.img',shape=(self.PETImage_shape))
            self.f_init = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'MLEM_it60_REF_cropped.img',shape=(self.PETImage_shape))

        # Initialize and save mu variable from ADMM
        self.mu = 0* np.ones((self.PETImage_shape[0], self.PETImage_shape[1]), dtype='<f')
        print("self.suffix")
        print(self.suffix)
        self.save_img(self.mu,self.subroot+'Block2/mu/'+ format(self.experiment)+'/mu_' + format(-1) + self.suffix + '.img')

        # Launch short MLEM reconstruction
        path_mlem_init = self.subroot_data + 'Data/MLEM_reco_for_init/' + self.phantom
        my_file = Path(path_mlem_init + '/' + self.phantom + '/' + self.phantom + '_it1.img')
        if (~my_file.is_file()):
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

    def castor_reconstruction(self,writer, i, subroot, sub_iter_MAP, experiment, hyperparameters_config, method, phantom, replicate, suffix, image_gt, f, mu, PETImage_shape, PETImage_shape_str, rho, alpha, image_init_path_without_extension):
        start_time_block1 = time.time()
        mlem_sequence = hyperparameters_config['mlem_sequence']
        nb_iter_second_admm = hyperparameters_config["nb_iter_second_admm"]

        # Save image f-mu in .img and .hdr format - block 1
        subroot_output_path = (subroot + 'Block1/' + suffix)
        path_before_eq_22 = (subroot_output_path + '/before_eq22/')
        path_during_eq_22 = (subroot_output_path + '/during_eq22/')
        self.save_img(f-mu, path_before_eq_22 + format(i) + '_f_mu.img')
        self.write_hdr(subroot,[i],'before_eq22',phantom,'f_mu',subroot_output_path)
        f_mu_for_penalty = ' -multimodal ' + subroot_output_path + '/before_eq22/' + format(i) + '_f_mu' + '.hdr'
        
        # Initialization
        if (method == 'nested'):
            only_x = False # Freezing u and v computation, just updating x if True

            # x^0
            copy(subroot + 'Data/initialization/' + image_init_path_without_extension + '.img', path_during_eq_22 + format(i) + '_-1_it' + str(hyperparameters_config["sub_iter_MAP"]) + '.img')
            self.write_hdr(subroot,[i,-1],'during_eq22',phantom,'x',subroot_output_path)

            # Compute u^0 (u^-1 in CASToR) and store it with zeros, and save in .hdr format - block 1            
            u_0 = 0*np.ones((344,252)) # initialize u_0 to zeros
            self.save_img(u_0,path_during_eq_22 + format(i) + '_-1_u.img')
            self.write_hdr(subroot,[i,-1],'during_eq22',phantom,'u',subroot_output_path,matrix_type='sino')
            
            # Compute v^0 (v^-1 in CASToR) with ADMM_spec_init_v optimizer and save in .hdr format - block 1
            if (i == 0):   # choose initial image for CASToR reconstruction
                x_for_init_v = ' -img ' + subroot + 'Data/initialization/' + image_init_path_without_extension + '.hdr' if image_init_path_without_extension != "" else '' # initializing CASToR MAP reconstruction with image_init or with CASToR default values
                #v^0 is BSREM if we only look at x optimization
                if (only_x):
                    x_for_init_v = ' -img ' + subroot + 'Data/initialization/' + 'BSREM_it30_REF_cropped' + '.hdr' if image_init_path_without_extension != "" else '' # initializing CASToR MAP reconstruction with image_init or with CASToR default values
                #x_for_init_v = ' -img ' + subroot + 'Data/initialization/' + '1_im_value' + '.hdr' if image_init_path_without_extension != "" else '' # initializing CASToR MAP reconstruction with image_init or with CASToR default values
            elif (i >= 1):
                x_for_init_v = ' -img ' + subroot + 'Block1/' + suffix + '/during_eq22/' +format(i-1) + '_' + format(nb_iter_second_admm) + '_it' + str(hyperparameters_config["sub_iter_MAP"]) + '.hdr'
            
            # Useful variables for command line
            k=-3
            base_name_k = format(i) + '_' + format(k)
            base_name_k_next = format(i) + '_' + format(k+1)
            full_output_path_k = subroot_output_path + '/during_eq22/' + base_name_k
            full_output_path_k_next = subroot_output_path + '/during_eq22/' + base_name_k_next
            v_for_additional_data = ' -additional-data ' + full_output_path_k + '_v.hdr'
            u_for_additional_data = ' -additional-data ' + full_output_path_k + '_u.hdr'

            # Define command line to run ADMM with CASToR
            castor_command_line_x = self.castor_command_line_func(method,phantom,replicate,PETImage_shape_str,rho,alpha,i,k)
            x_reconstruction_command_line = castor_command_line_x + ' -fout ' + full_output_path_k_next + ' -it 1:1' + x_for_init_v + f_mu_for_penalty #+ u_for_additional_data + v_for_additional_data # we need f-mu so that ADMM optimizer works, even if we will not use it...
            # Compute one ADMM iteration (x, v, u). When only initializing x, u computation is only the forward model Ax, thus exactly what we want to initialize v
            print('vvvvvvvvvvv0000000000')
            self.compute_x_v_u_ADMM(x_reconstruction_command_line,full_output_path_k_next,'during_eq22',i,k,phantom,only_x,subroot_output_path,subroot)
            copy(path_during_eq_22 + base_name_k_next + '_u.img', path_during_eq_22 + format(i) + '_-1_v.img')
            self.write_hdr(subroot,[i,-1],'during_eq22',phantom,'v',subroot_output_path,matrix_type='sino')
                
        # Choose number of argmax iteration for (second) x computation
        if (mlem_sequence):
            #it = ' -it 2:56,4:42,6:36,4:28,4:21,2:14,2:7,2:4,2:2,2:1' # large subsets sequence to approximate argmax, too many subsets for 2D, but maybe ok for 3D
            it = ' -it 16:28,4:21,2:14,2:7,2:4,2:2,2:1' # large subsets sequence to approximate argmax, 2D
        else:
            it = ' -it ' + str(sub_iter_MAP) + ':1' # Only 2 iterations (Gong) to compute argmax, if we estimate it is an enough precise approximation. Only 1 according to conjugate gradient in Lim et al.
            #it = ' -it ' + '5:14' # Only 2 iterations to compute argmax, if we estimate it is an enough precise approximation 
        
        # Whole computation
        if (method == 'nested'):
            # Second ADMM computation
            for k in range(-1,nb_iter_second_admm): # iteration -1 is to initialize v fitting data
                # Initialize variables for command line
                if (k == -1):
                    if (i == 0):   # choose initial image for CASToR reconstruction
                        initialimage = ' -img ' + subroot + 'Data/initialization/' + image_init_path_without_extension + '.hdr' if image_init_path_without_extension != "" else '' # initializing CASToR MAP reconstruction with image_init or with CASToR default values
                    elif (i >= 1):
                        initialimage = ' -img ' + subroot + 'Block1/' + suffix + '/during_eq22/' +format(i-1) + '_' + format(nb_iter_second_admm) + '_it' + str(hyperparameters_config["sub_iter_MAP"]) + '.hdr'
                        # Trying to initialize ADMMLim
                        #initialimage = ' -img ' + subroot + 'Data/initialization/' + 'BSREM_it30_REF_cropped.hdr'
                        initialimage = ' -img ' + subroot + 'Data/initialization/' + '1_im_value_cropped.hdr'
                        if (only_x):
                            initialimage = ' -img ' + subroot + 'Block1/' + suffix + '/during_eq22/' + format(i-1) + '_' + format(nb_iter_second_admm) + '_it' + str(hyperparameters_config["sub_iter_MAP"]) + '.hdr'

                else:
                    initialimage = ' -img ' + subroot + 'Block1/' + suffix + '/during_eq22/' + format(i) + '_' + format(k) + '_it' + str(hyperparameters_config["sub_iter_MAP"]) + '.hdr'

                base_name_k = format(i) + '_' + format(k)
                base_name_k_next = format(i) + '_' + format(k+1)
                full_output_path_k = subroot_output_path + '/during_eq22/' + base_name_k
                full_output_path_k_next = subroot_output_path + '/during_eq22/' + base_name_k_next
                f_mu_for_penalty = ' -multimodal ' + subroot_output_path + '/before_eq22/' + format(i) + '_f_mu' + '.hdr'
                v_for_additional_data = ' -additional-data ' + full_output_path_k + '_v.hdr'
                u_for_additional_data = ' -additional-data ' + full_output_path_k + '_u.hdr'

                # Define command line to run ADMM with CASToR
                castor_command_line_x = self.castor_command_line_func(method,phantom,replicate,PETImage_shape_str,rho,alpha,i,k)
                # Compute one ADMM iteration (x, v, u)
                x_reconstruction_command_line = castor_command_line_x + ' -fout ' + full_output_path_k_next + it + f_mu_for_penalty + u_for_additional_data + v_for_additional_data + initialimage    
                print('xxxxxxxxxuuuuuuuuuuuvvvvvvvvv')
                self.compute_x_v_u_ADMM(x_reconstruction_command_line,full_output_path_k_next,'during_eq22',i,k,phantom,only_x,subroot_output_path,subroot)

                x = self.fijii_np(full_output_path_k_next + '_it' + str(hyperparameters_config["sub_iter_MAP"]) + '.img', shape=(PETImage_shape[0],PETImage_shape[1]))
                if (k>=-1):
                    self.write_image_tensorboard(writer,x,"x in second ADMM over iterations",suffix,image_gt, k+1+i*nb_iter_second_admm) # Showing all corrupted images with same contrast to compare them together
                    self.write_image_tensorboard(writer,x,"x in second ADMM over iterations(FULL CONTRAST)",suffix,image_gt, k+1+i*nb_iter_second_admm,full_contrast=True) # Showing all corrupted images with same contrast to compare them together

        elif (method == 'Gong'):
            # Define command line to run ADMM with CASToR
            castor_command_line_x = self.castor_command_line_func(method,phantom,replicate,PETImage_shape_str,rho,alpha,i)
            # Initialize image
            if (i == 0):   # choose initial image for CASToR reconstruction
                initialimage = ' -img ' + subroot + 'Data/initialization/' + image_init_path_without_extension + '.hdr' if image_init_path_without_extension != "" else '' # initializing CASToR MAP reconstruction with image_init or with CASToR default values
            elif (i >= 1):
                initialimage = ' -img ' + subroot + 'Block1/' + suffix + '/during_eq22/' + format(i-1) + '_' + format(nb_iter_second_admm) + '_it' + str(hyperparameters_config["sub_iter_MAP"]) + '.hdr'
                # Trying to initialize OPTITR
                #initialimage = ' -img ' + subroot + 'Data/initialization/' + 'BSREM_it30_REF_cropped.hdr'
                initialimage = ' -img ' + subroot + 'Data/initialization/' + '1_im_value_cropped.hdr'
                #initialimage = ' -img ' + subroot + 'Block1/' + suffix + '/during_eq22/' + format(i-1) + '_it' + str(hyperparameters_config["sub_iter_MAP"]) + '.hdr'

            base_name_k_next = format(i)
            full_output_path_k_next = subroot_output_path + '/during_eq22/' + base_name_k_next
            x_reconstruction_command_line = castor_command_line_x + ' -fout ' + full_output_path_k_next + it + f_mu_for_penalty + initialimage            
            os.system(x_reconstruction_command_line)

            if (mlem_sequence):
                x = self.fijii_np(full_output_path_k_next + '_it30.img', shape=(PETImage_shape))
            else:
                x = self.fijii_np(full_output_path_k_next + '_it1.img', shape=(PETImage_shape))
            self.write_image_tensorboard(writer,x,"x after optimization transfer over iterations",suffix,image_gt, i) # Showing all corrupted images with same contrast to compare them together
            self.write_image_tensorboard(writer,x,"x after optimization transfer over iterations (FULL CONTRAST)",suffix,image_gt, i,full_contrast=True) # Showing all corrupted images with same contrast to compare them together


        print("--- %s seconds - second ADMM (CASToR) iteration ---" % (time.time() - start_time_block1))

        # Load previously computed image with CASToR ADMM optimizers

        if (method == 'nested'):
            x = self.fijii_np(full_output_path_k_next + '_it' + str(hyperparameters_config["sub_iter_MAP"]) + '.img', shape=(PETImage_shape))
        elif (method == 'Gong'):
            if (mlem_sequence):
                x = self.fijii_np(full_output_path_k_next + '_it30.img', shape=(PETImage_shape))
            else:
                x = self.fijii_np(full_output_path_k_next + '_it1.img', shape=(PETImage_shape))

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

    def compute_x_v_u_ADMM(self,x_reconstruction_command_line,full_output_path,subdir,i,k,phantom,only_x,subroot_output_path,subroot):
        # Compute x,u,v
        os.system(x_reconstruction_command_line)
        # Write x hdr file
        #self.write_hdr(subroot,[i,k+1],subdir,phantom,'it_'+str(self.sub_iter_MAP),subroot_output_path=subroot_output_path)
        # Write v hdr file and change v file if only x computation is needed
        if (only_x):
            copy(subroot_output_path + subdir + format(i) + '_' + format(-1) + '_v.img',full_output_path + '_v.img')
        self.write_hdr(subroot,[i,k+1],subdir,phantom,'v',subroot_output_path=subroot_output_path,matrix_type='sino')
        # Write u hdr file and change u file if only x computation is needed
        if (only_x):
            copy(subroot_output_path + '/' + subdir + '/' + format(i) + '_' + format(-1) + '_u.img',full_output_path + '_u.img')
        self.write_hdr(subroot,[i,k+1],subdir,phantom,'u',subroot_output_path=subroot_output_path,matrix_type='sino')