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
        
        os.system("rm -rf " + self.subroot_phantom+'Block2/' + self.suffix + '/checkpoint/'+format(self.experiment) + "*")


        # Specific hyperparameters for reconstruction module (Do it here to have raytune config hyperparameters selection)
        if (self.method != "MLEM" and self.method != "OSEM" and self.method != "AML"):
            self.rho = config["rho"]
        else:
            self.rho = 0
        if ('ADMMReg' in self.method or  "DNA" in self.method or "DIPRecon" in self.method):
            if (self.method != "ADMMReg"):
                self.unnested_1st_outer_iter = config["unnested_1st_outer_iter"]
            else:
                self.unnested_1st_outer_iter = None
            if ( "DIPRecon" in self.method):
                self.alpha = None
            else:
                if (config["recoInDNA"] == "ADMMReg"):
                    if ("stoppingCriterionValue" in config):
                        self.stoppingCriterionValue = config["stoppingCriterionValue"]
                    else:
                        self.stoppingCriterionValue = 0
                    if ("stoppingCriterionValue" in config):
                        self.saveSinogramsUAndV = config["saveSinogramsUAndV"]
                    else:
                        self.saveSinogramsUAndV = 0
                    self.alpha = config["alpha"]
                    self.adaptive_parameters = config["adaptive_parameters"]
                else:
                    self.alpha = None
                    self.adaptive_parameters = "nothing"
                    self.A_AML = config["A_AML"]
                if (self.adaptive_parameters == "nothing"): # set mu, tau, xi to any values, there will not be used in CASToR
                    self.mu_adaptive = np.NaN
                    self.tau = np.NaN
                    self.xi = np.NaN
                    self.tau_max = np.NaN
                else:
                    self.mu_adaptive = config["mu_adaptive"]
                    self.tau = config["tau"]
                    self.xi = config["xi"]
                    if (self.adaptive_parameters == "both"):
                        self.tau_max = config["tau_max"]
                    else:
                        self.tau_max = np.NaN
        if ("image_init_path_without_extension" in config):
            if (not config["image_init_path_without_extension"]):
                self.image_init_path_without_extension = '1_im_value_cropped'
            else:
                self.image_init_path_without_extension = config["image_init_path_without_extension"]
        else: # Default in CASToR is to initalize reconstruction with a uniform image with ones
            self.image_init_path_without_extension = '1_im_value_cropped'
        self.tensorboard = config["tensorboard"]

        # Initialize and save mu variable from ADMM
        if ("DNA" in self.method or "DIPRecon" in self.method):
            self.mu = 0* np.ones((self.PETImage_shape))
            if config["FLTNB"] == "float":
                self.mu = self.mu.astype(np.float32)
            if (self.PETImage_shape[2] > 1):
                self.mu = self.mu.reshape(self.PETImage_shape[::-1])
            self.save_img(self.mu,self.subroot_phantom+'Block2/' + self.suffix + '/mu/'+ format(self.experiment)+'/mu_' + format(-1) + self.suffix + '.img')

        # Launch short MLEM reconstruction
        self.launch_quick_mlem(config)

    
    def launch_quick_mlem(self,config):
        path_mlem_init = self.subroot + 'Data/MLEM_reco_for_init_hdr/' + self.phantom
        my_file = Path(path_mlem_init + '/' + self.phantom + '_it1.img')
        if (not my_file.is_file()):
            it_option = ' -it 1:1'
            output_path = ' -dout ' + self.subroot + 'Data/MLEM_reco_for_init_hdr/' + self.phantom
            initial_image = ''
            castor_command_line = self.castor_common_command_line(self.subroot, self.PETImage_shape_str, self.phantom, self.replicate,mlem_quick=True) + self.castor_opti_and_penalty("MLEM", self.penalty, self.rho) + it_option + output_path + initial_image
            print(castor_command_line)
            os.system(castor_command_line)

    def castor_reconstruction(self,writer, i, i_init, subroot, nb_inner_iteration, experiment, config, method, phantom, replicate, suffix, image_gt, f, mu, PETImage_shape, PETImage_shape_str, alpha, image_init_path_without_extension):
        start_time_block1 = time.time()
        mlem_sequence = config['mlem_sequence']

        # Save image f-mu in .img and .hdr format - block 1
        if (i == i_init and i_init > 0 and config["unnested_1st_outer_iter"]):   # choose initial image for CASToR reconstruction
            f = self.fijii_np(self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.net + '' + format(i-1) + '_FINAL.img',shape=(self.PETImage_shape),type_im='<f') # loading DIP output
            mu = self.fijii_np(self.subroot_phantom+'Block2/' + self.suffix + '/mu/'+ format(self.experiment)+'/mu_' + format(i-1) + self.suffix + '.img',shape=(self.PETImage_shape)) # loading mu
        elif (i == 0 and config["unnested_1st_outer_iter"]):
            f = self.fijii_np(self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.net + '' + format(i-1) + '_FINAL.img',shape=(self.PETImage_shape),type_im='<f') # loading DIP output
        
        subroot_output_path = (subroot + 'Block1/' + suffix)
        path_before_eq_22 = (subroot_output_path + '/before_eq22/')
        self.save_img(f-mu, path_before_eq_22 + format(i) + '_f_mu.img')
        self.write_hdr(self.subroot,[i],'before_eq22',phantom,'f_mu',subroot_output_path)
        f_mu_for_penalty_path = subroot_output_path + '/before_eq22/' + format(i) + '_f_mu' + '.hdr' # Will be removed if initialization and unnested_1st_outer_iter (rho == 0)
        subdir = 'during_eq22'

        # If rho is 0, remove f_mu_for_penalty
        if ((self.rho == 0) or (i==-1 and not self.unnested_1st_outer_iter)): # For first iteration, put rho to zero
            f_mu_for_penalty_path = ''
        # Write f_mu path in config
        text_file = open(self.subroot_phantom + 'Block1/' + self.suffix  + '/' + 'QUAD.conf', "w")
        text_file.write("# Path to target image (default is uniform image with zeros)" + "\n")
        text_file.write("target image path : " + f_mu_for_penalty_path + "\n")
        text_file.close()
        # Initialization
        self.recoInDNA = config["recoInDNA"]
        if (method == 'DNA'):
            if config["recoInDNA"] == "ADMMReg":
                x = self.ADMMReg_general(config, i, subroot_output_path,writer,image_gt, i_init, subdir=subdir)
            elif config["recoInDNA"] == "APPGML":
                print("APPGML in DNA")
                # Choose number of argmax iteration for (second) x computation
                if (mlem_sequence):
                    self.it_option = ' -it 16:28,4:21,2:14,2:7,2:4,2:2,2:1' # large subsets sequence to approximate argmax, 2D
                else: 
                    self.it_option = ' -it ' + str(nb_inner_iteration) + ':' + str(config["nb_subsets"]) # Put 28 subsets to be quick
                    # self.it_option = ' -it ' + str(nb_inner_iteration) + ':' + str(config["nb_subsets"]) # Only 2 iterations (DIPRecon) to compute argmax, if we estimate it is an enough precise approximation. Only 1 according to conjugate gradient in Lim et al.

                # Write shift A in config
                # Read lines in config file
                try:
                    with open(self.subroot_phantom + 'Block1/' + self.suffix  + '/' + 'APPGML.conf', 'r') as read_config_file:
                        data = read_config_file.readlines()
                except:
                    with open(self.subroot_phantom + 'Block1/' + self.suffix  + '/' + 'APPGML.conf', "w") as write_config_file:
                        with open(self.subroot + 'APPGML_no_replicate.conf', "r") as read_config_file:
                            write_config_file.write(read_config_file.read())
                    with open(self.subroot_phantom + 'Block1/' + self.suffix  + '/' + 'APPGML.conf', 'r') as read_config_file:
                        data = read_config_file.readlines()
                    # Change the line with shift
                for line_idx in range (len(data)):
                    line = data[line_idx]
                    if line.startswith("bound"):
                        data[line_idx] = "bound: " + str(self.A_AML) + "\n"
                # Write everything back
                with open(self.subroot_phantom + 'Block1/' + self.suffix  + '/' + 'APPGML.conf', "w") as write_config_file:
                    write_config_file.writelines(data)
                # Define command line to run OPTITR with CASToR
                castor_command_line_x = self.castor_common_command_line(self.subroot, self.PETImage_shape_str, self.phantom, self.replicate) + self.castor_opti_and_penalty(self.method, self.penalty, self.rho, i, self.unnested_1st_outer_iter)
                # Initialize image
                
                if (i == 0 and not config["unnested_1st_outer_iter"]):   # choose initial image for CASToR reconstruction
                    self.initial_image = ' -img ' + self.subroot_phantom + '/Block2/' + self.suffix + '/out_cnn/' + str(self.experiment) + '/out_' + self.net + str(i-1) + '_FINAL.hdr' # DIPRecon initializes to DIP output at pre iteratio
                elif (i == 0 and config["unnested_1st_outer_iter"]):
                    self.initial_image = ''
                else:
                    if (i == 1 and config["unnested_1st_outer_iter"]):
                        # self.initial_image = ' -img ' + subroot_output_path + '/' + subdir + '/' + format(i-1) + '_it' + str(config["nb_inner_iteration"]) + '.hdr'    
                        self.initial_image = ' -img ' + self.subroot_phantom + '/Block2/' + self.suffix + '/out_cnn/' + str(self.experiment) + '/out_' + self.net + str(i-1) + '_FINAL.hdr'
                    else:
                        self.initial_image = ' -img ' + subroot_output_path + '/' + subdir + '/' + format(i-1) + '_it' + str(config["nb_inner_iteration"]) + '.hdr'    
                    
                base_name_i = format(i)
                full_output_path_i = subroot_output_path + '/' + subdir + '/' + base_name_i
                x_reconstruction_command_line = castor_command_line_x + ' -fout ' + full_output_path_i + self.it_option + self.initial_image
                print(x_reconstruction_command_line)
                os.system(x_reconstruction_command_line)

                if (mlem_sequence):
                    x = self.fijii_np(full_output_path_i + '_it30.img', shape=(PETImage_shape))
                else:
                    x = self.fijii_np(full_output_path_i + '_it' + str(config["nb_inner_iteration"]) + '.img', shape=(PETImage_shape))
                
                self.write_image_tensorboard(writer,x,"x after optimization transfer over iterations",suffix,image_gt, i) # Showing all corrupted images with same contrast to compare them together
                self.write_image_tensorboard(writer,x,"x after optimization transfer over iterations (FULL CONTRAST)",suffix,image_gt, i,full_contrast=True) # Showing all corrupted images with same contrast to compare them together

        elif (method == "DIPRecon"):

            # Choose number of argmax iteration for (second) x computation
            if (mlem_sequence):
                self.it_option = ' -it 16:28,4:21,2:14,2:7,2:4,2:2,2:1' # large subsets sequence to approximate argmax, 2D
            else:
                self.it_option = ' -it ' + str(nb_inner_iteration) + ':1' # Only 2 iterations (DIPRecon) to compute argmax, if we estimate it is an enough precise approximation

            # Define command line to run OPTITR with CASToR
            castor_command_line_x = self.castor_common_command_line(self.subroot, self.PETImage_shape_str, self.phantom, self.replicate) + self.castor_opti_and_penalty(self.method, self.penalty, self.rho, i, self.unnested_1st_outer_iter)
            # Initialize image
            
            if (i == 0 and not config["unnested_1st_outer_iter"]):   # choose initial image for CASToR reconstruction
                self.initial_image = ' -img ' + self.subroot_phantom + '/Block2/' + self.suffix + '/out_cnn/' + str(self.experiment) + '/out_' + self.net + str(i-1) + '_FINAL.hdr' # DIPRecon initializes to DIP output at pre iteratio
            elif (i == 0 and config["unnested_1st_outer_iter"]):
                self.initial_image = ' -img ' + self.subroot + 'Data/initialization/' + '1_im_value_cropped.hdr'
            else:
                if (i == 1 and config["unnested_1st_outer_iter"]):
                    self.initial_image = ' -img ' + subroot_output_path + '/' + subdir + '/' + format(i-1) + '_it' + str(config["nb_inner_iteration"]) + '.hdr'    
                    self.initial_image = ' -img ' + self.subroot_phantom + '/Block2/' + self.suffix + '/out_cnn/' + str(self.experiment) + '/out_' + self.net + str(i-1) + '_FINAL.hdr'
                else:
                    self.initial_image = ' -img ' + subroot_output_path + '/' + subdir + '/' + format(i-1) + '_it' + str(config["nb_inner_iteration"]) + '.hdr'    
                
            base_name_i = format(i)
            full_output_path_i = subroot_output_path + '/' + subdir + '/' + base_name_i
            x_reconstruction_command_line = castor_command_line_x + ' -fout ' + full_output_path_i + self.it_option + self.initial_image            
            print(x_reconstruction_command_line + ' -oit -1')
            os.system(x_reconstruction_command_line + ' -oit -1')

            if (mlem_sequence):
                x = self.fijii_np(full_output_path_i + '_it30.img', shape=(PETImage_shape))
            else:
                x = self.fijii_np(full_output_path_i + '_it' + str(config["nb_inner_iteration"]) + '.img', shape=(PETImage_shape))
            
            print(full_output_path_i + '_it' + str(config["nb_inner_iteration"]) + '.img')

            self.write_image_tensorboard(writer,x,"x after optimization transfer over iterations",suffix,image_gt, i) # Showing all corrupted images with same contrast to compare them together
            self.write_image_tensorboard(writer,x,"x after optimization transfer over iterations (FULL CONTRAST)",suffix,image_gt, i,full_contrast=True) # Showing all corrupted images with same contrast to compare them together

        print("--- %s seconds - second ADMM (CASToR) iteration ---" % (time.time() - start_time_block1))

        # Save image x in .img and .hdr format - block 1
        name = (subroot+'Block1/' + suffix + '/out_eq22/' + format(i) + '.img')
        self.save_img(x, name)
        self.write_hdr(subroot,[i],'out_eq22',phantom,'',subroot_output_path)

        # Save x_label for load into block 2 - NN as corrupted image (x_label)
        x_label = x + mu
        # Save x_label in .img and .hdr format
        name=(subroot+'Block2/' + self.suffix + '/x_label/'+format(experiment) + '/' + format(i) +'_x_label' + suffix + '.img')
        self.save_img(x_label, name)

        return x_label, x

    def compute_x_v_u_ADMM(self,x_reconstruction_command_line,subdir,i,phantom,subroot_output_path,subroot,method, it_name=''):
        # Compute x,u,v
        #os.system(x_reconstruction_command_line + ' -oit 90:' + str(int(self.config["nb_inner_iteration"]*3)))
        if ("DNA" in self.method): # we only need output at last iteration
            if (self.nb_dimensions == 2): # 2D
                os.system(x_reconstruction_command_line + ' -oit -1')
            else:
                os.system(x_reconstruction_command_line)
                # os.system(x_reconstruction_command_line + ' -oit -1')
        else: # ADMMReg, save output at all iterations
            os.system(x_reconstruction_command_line)
        # Change iteration name for header if stopping criterion reached
        try:
            path_stopping_criterion = self.subroot_phantom + self.suffix + '/' + format(0) + '_adaptive_stopping_criteria.log'
            with open(path_stopping_criterion) as f:
                first_line = f.readline() # Read first line to get second one
                it_name = int(f.readline().rstrip())
        except:
            pass
        # Write u and v hdr files
        self.write_hdr(subroot,[i],subdir,phantom,'u_it' + str(it_name),subroot_output_path=subroot_output_path,matrix_type='sino')
        self.write_hdr(subroot,[i],subdir,phantom,'v_it' + str(it_name),subroot_output_path=subroot_output_path,matrix_type='sino')

    def ADMMReg_general(self, config, i, subroot_output_path,writer=None,image_gt=None, i_init=0, subdir=""):
        # if ("DNA" in self.method):
        #     self.post_smoothing = 0
        castor_command_line_x = self.castor_common_command_line(self.subroot, self.PETImage_shape_str, self.phantom, self.replicate)

        base_name_i = format(i)
        full_output_path_i = subroot_output_path + '/' + subdir + '/' + base_name_i

        if ("DNA" in self.method):
            folder_sub_path = os.path.join(self.subroot_phantom,"Block1",self.suffix)
        else:
            folder_sub_path = os.path.join(self.subroot_phantom,self.suffix)
        #''' Continue previous computation if ADMMReg have already been launched with these settings
        if ("ADMMReg" in self.method):
            if (self.ImageAndItToResumeComputation(folder_sub_path, config)):
                u_path = full_output_path_i + '_u_it' + str(self.last_iter) + '.hdr'
                u_for_additional_data = ' -additional-data ' + u_path
                v_path = full_output_path_i + '_v_it' + str(self.last_iter) + '.hdr'
                v_for_additional_data = ',' + v_path

                # Write u and v hdr files for last computed iteration if they do not exist
                if (not os.path.isfile(u_path)):
                    self.write_hdr(self.subroot_phantom,[0],subdir,self.phantom,'u_it' + str(self.last_iter),subroot_output_path=subroot_output_path,matrix_type='sino')
                if (not os.path.isfile(v_path)):
                    self.write_hdr(self.subroot_phantom,[0],subdir,self.phantom,'v_it' + str(self.last_iter),subroot_output_path=subroot_output_path,matrix_type='sino')

                if (self.adaptive_parameters != "nothing"):
                    last_log_file = os.path.join(folder_sub_path,"0_adaptive_it" + str(self.last_iter) + ".log")
                    with open(last_log_file) as f:
                        f.readline() # Read first line to get second one (adaptive alpha value)
                        second_line = f.readline()
                        if (self.FLTNB == 'float'):       
                            self.alpha = np.float32(second_line)
                        elif (self.FLTNB == 'double'):
                            self.alpha = np.float64(second_line)
            else:
                u_for_additional_data = ""
                v_for_additional_data = ""

        # Initialization image for ADMM-Reg inside DNA using previously computed images from outer iteration
        if ("DNA" in self.method):
            self.it_option = ' -it ' + str(config["nb_inner_iteration"]) + ':1'  # 1 subset
            if (not config["use_u_and_v_DNA"] or (i == i_init+1)):
                u_for_additional_data = ''
                v_for_additional_data = ''
            else:
                u_path = subroot_output_path + '/' + subdir + '/' + format(i-1) + '_u_it' + str(config["nb_inner_iteration"]) + '.hdr'
                u_for_additional_data = ' -additional-data ' + u_path
                v_path = subroot_output_path + '/' + subdir + '/' + format(i-1) + '_v_it' + str(config["nb_inner_iteration"]) + '.hdr'
                #v_for_additional_data = ' -additional-data ' + v_path
                v_for_additional_data = ',' + v_path

            if (i == 0 and not config["unnested_1st_outer_iter"]):   # choose initial image for CASToR reconstruction
                self.initial_image = ' -img ' + self.subroot_phantom + '/Block2/' + self.suffix + '/out_cnn/' + str(self.experiment) + '/out_' + self.net + str(i-1) + '_FINAL.hdr' # DIPRecon initializes to DIP output at pre iteratio
            elif (i == 0 and config["unnested_1st_outer_iter"]):
                self.initial_image = ' -img ' + self.subroot_phantom + '/Block2/' + self.suffix + '/out_cnn/' + str(self.experiment) + '/out_' + self.net + str(i-1) + '_FINAL.hdr' # DIPRecon initializes to DIP output at pre iteratio
            else: # Last image for next outer iteration
                if (i == 1 and ((i_init == -1 and not config["unnested_1st_outer_iter"]) or (i_init == 0 and config["unnested_1st_outer_iter"])) and config["unnested_1st_outer_iter"]):
                    self.initial_image = ' -img ' + subroot_output_path + '/' + 'out_eq22' + '/' +format(i-1) + '.hdr'
                    self.initial_image = ' -img ' + self.subroot_phantom + '/Block2/' + self.suffix + '/out_cnn/' + str(self.experiment) + '/out_' + self.net + str(i-1) + '_FINAL.hdr'
                else:
                    self.initial_image = ' -img ' + subroot_output_path + '/' + 'out_eq22' + '/' +format(i-1) + '.hdr'

        if ('ADMMReg' in self.method):
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

        # Optimizer and penalty in command line, change rho if first outer iteration and unnested_1st_outer_iter
        opti_and_penalty = self.castor_opti_and_penalty(self.method, self.penalty, self.rho, i, self.unnested_1st_outer_iter)

        x_reconstruction_command_line = castor_command_line_x \
                                        + opti_and_penalty \
                                        + ' -fout ' + full_output_path_i + self.it_option \
                                        + u_for_additional_data + v_for_additional_data \
                                        + self.initial_image \
                                        + conv # we need f-mu so that ADMM optimizer works, even if we will not use it...

        print(x_reconstruction_command_line)
        self.compute_x_v_u_ADMM(x_reconstruction_command_line, subdir, i, self.phantom, subroot_output_path, self.subroot, self.method, it_name = config["nb_inner_iteration"])

        if (self.adaptive_parameters != "nothing" and config["castor_foms"]):
            #'''
            # -- AdaptiveAlpha ---- AdaptiveAlpha ---- AdaptiveAlpha ---- AdaptiveAlpha ---- AdaptiveAlpha ---- AdaptiveAlpha --
            self.path_stopping_criterion = subroot_output_path + '/' + subdir + '/' + format(i) + '_adaptive_stopping_criteria.log'
            if(isfile(self.path_stopping_criterion)):
                theLog = pd.read_table(self.path_stopping_criterion)
                finalOuterIterRow = theLog.loc[[0]]

                finalOuterIterRowArray = np.array(finalOuterIterRow)
                finalOuterIterRowString = finalOuterIterRowArray[0, 0]
                finalOuterIter = int(finalOuterIterRowString)
                print("finalOuterIter",finalOuterIter)
            else:
                finalOuterIter = config["nb_inner_iteration"]

            for inner_it in range(1,finalOuterIter+1):
                path_adaptive = subroot_output_path + '/' + subdir + '/' + format(i) + '_adaptive_it' + format(inner_it) + '.log'
                theLog = pd.read_table(path_adaptive)
                relativePrimalResidualRow = theLog.loc[[4]]
                relativePrimalResidualRowArray = np.array(relativePrimalResidualRow)
                relativePrimalResidualRowString = relativePrimalResidualRowArray[0, 0]
                relativePrimalResidual = float(relativePrimalResidualRowString)
                print("relPrimal",relativePrimalResidual)

            for inner_it in range(1,finalOuterIter+1):
                path_adaptive = subroot_output_path + '/' + subdir + '/' + format(i) + '_adaptive_it' + format(inner_it) + '.log'
                theLog = pd.read_table(path_adaptive)
                relativeDualResidualRow = theLog.loc[[6]]
                relativeDualResidualRowArray = np.array(relativeDualResidualRow)
                relativeDualResidualRowString = relativeDualResidualRowArray[0, 0]
                relativeDualResidual = float(relativeDualResidualRowString)
                print("relDual",relativeDualResidual)
            #'''
        else:
            finalOuterIter = config["nb_inner_iteration"]
        x = self.fijii_np(full_output_path_i + '_it' + str(finalOuterIter) + '.img', shape=(self.PETImage_shape))
        return x
