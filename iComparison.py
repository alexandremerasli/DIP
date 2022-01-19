# Useful
from pathlib import Path
import os
import numpy as np
from shutil import copy
import argparse

# Local files to import
from vReconstruction import vReconstruction

class iComparison(vReconstruction):
    def __init__(self,config):
        print("__init__")

    def runComputation(self,config,fixed_config,hyperparameters_config,root):
        print("hyperparameters_config",hyperparameters_config)

        # Initializing results class
        from iResults import iResults
        classResults = iResults(fixed_config,hyperparameters_config,root)
        classResults.initializeSpecific(fixed_config,hyperparameters_config,root)
        
        
        '''
        hyperparameters_config = {
        "penalty" : 'MRF',
        #"penalty" : 'DIP_ADMM'
        }
        '''
        beta = [0.0001,0.001,0.01,0.1,1,10]
        beta = [0.01,0.03,0.05,0.07,0.09]
        beta = [0.03,0.035,0.04,0.045,0.05]
        beta = [0.04]
        print("hyperparameters_config",hyperparameters_config)

        if (fixed_config["method"] == 'ADMMLim'):
            self.ADMMLim(fixed_config,hyperparameters_config,beta)
        else:
            if (config["method"] == 'MLEM'):
                beta = [0]
            for i in range(len(beta)):
                print(i)

                # castor-recon command line
                header_file = ' -df ' + self.subroot + 'Data/database_v2/' + self.phantom + '/data' + self.phantom[-1] + '/data' + self.phantom[-1]  + '.cdh' # PET data path

                executable = 'castor-recon'
                dim = ' -dim ' + self.PETImage_shape_str
                vox = ' -vox 4,4,4'
                vb = ' -vb 1'
                it = ' -it ' + str(self.max_iter) + ':28'
                th = ' -th 0'
                proj = ' -proj incrementalSiddon'
                psf = ' -conv gaussian,4,4,3.5::psf'

                if (config["method"] == 'MLEM'):
                    opti = ' -opti ' + config["method"]
                    conv = ' -conv gaussian,8,8,3.5::post'
                    #conv = ''
                    penalty = ''
                    penaltyStrength = ''
                else:
                    opti = ' -opti ' + config["method"] + ':' + self.subroot + 'Comparison/' + 'BSREM.conf'
                    conv = ''
                    penalty = ' -pnlt MRF:' + self.subroot + 'Comparison/' + 'MRF.conf'
                    penaltyStrength = ' -pnlt-beta ' + str(beta[i])

                output_path = ' -dout ' + self.subroot + 'Comparison/' + config["method"] + '_beta_' + str(beta[i]) # Output path for CASTOR framework
                initialimage = ' -img ' + self.subroot+'Data/database_v2/' + self.phantom + '/' + self.phantom + '.hdr'
                initialimage = ''

                # Command line for calculating the Likelihood
                opti_like = ' -opti-fom'
                opti_like = ''

                print("CASToR command line :")
                print("")
                print(executable + dim + vox + output_path + header_file + vb + it + th + proj + opti + opti_like + initialimage + penalty + penaltyStrength + conv + psf)
                print("")
                os.system(executable + dim + vox + output_path + header_file + vb + it + th + proj + opti + opti_like + initialimage + penalty + penaltyStrength + conv + psf) # + ' -fov-out 95')



    def ADMMLim(self,fixed_config,hyperparameters_config,beta):

            # Variables from hyperparameters_config dictionnary
            print("hyperparameters_config",hyperparameters_config)
            it = ' -it ' + str(hyperparameters_config["sub_iter_MAP"]) + ':1' # 1 subset
            penalty = ' -pnlt ' + fixed_config["penalty"]
            if fixed_config["penalty"] == "MRF":
                penalty += ':' + self.subroot + 'Comparison/' + 'MRF.conf'

            only_x = False # Freezing u and v computation, just updating x if True

            # Path variables
            subroot_output_path = (self.subroot + 'Comparison/ADMMLim/' + self.suffix)
            subdir = 'ADMM'
            Path(self.subroot+'Comparison/ADMMLim/').mkdir(parents=True, exist_ok=True) # CASTor path
            Path(self.subroot+'Comparison/ADMMLim/' + self.suffix + '/ADMM').mkdir(parents=True, exist_ok=True) # CASToR path

            i = 0
            k = -2
            full_output_path_k_next = subroot_output_path + '/ADMM/' + format(i) + '_' + format(k+1)

            # Initialize u^0 (u^-1 in CASToR)
            copy(self.subroot + 'Data/initialization/0_sino_value.hdr', full_output_path_k_next + '_u.hdr')
            self.write_hdr(self.subroot,[i,-1],subdir,self.phantom,'u',subroot_output_path,matrix_type='sino')

            # Define command line to run ADMM with CASToR, to compute v^0
            castor_command_line_x = self.castor_admm_command_line(self.subroot, 'Lim', self.PETImage_shape_str, self.alpha, self.rho, self.phantom ,True, penalty)
            initialimage = ' -img ' + self.subroot + 'Data/initialization/' + self.image_init_path_without_extension + '.hdr' if self.image_init_path_without_extension != "" else '' # initializing CASToR MAP reconstruction with image_init or with CASToR default values
            f_mu_for_penalty = ' -multimodal ' + self.subroot + 'Data/initialization/BSREM_it30_REF_cropped.hdr'
            x_for_init_v = ' -img ' + self.subroot + 'Data/initialization/' + self.image_init_path_without_extension + '.hdr' if self.image_init_path_without_extension != "" else '' # initializing CASToR MAP reconstruction with image_init or with CASToR default values
            if (only_x):
                x_for_init_v = ' -img ' + self.subroot + 'Data/initialization/' + 'BSREM_it30_REF_cropped' + '.hdr' if self.image_init_path_without_extension != "" else '' # initializing CASToR MAP reconstruction with image_init or with CASToR default values
                
            # Compute one ADMM iteration (x, v, u) when only initializing x to compute v^0
            if (only_x):
                copy(self.subroot + 'Data/initialization/0_sino_value.hdr', subroot_output_path + '/' + subdir + '/' + format(i) + '_' + format(-1) + '_u.img')
            x_reconstruction_command_line = castor_command_line_x + ' -fout ' + full_output_path_k_next + ' -it 1:1' + x_for_init_v + f_mu_for_penalty # we need f-mu so that ADMM optimizer works, even if we will not use it...
            print('vvvvvvvvvvv0000000000')
            self.compute_x_v_u_ADMM(x_reconstruction_command_line,full_output_path_k_next,subdir,i,k-1,self.phantom,only_x,subroot_output_path,self.subroot)
            self.write_hdr(self.subroot,[i,k+1],'ADMM',self.phantom,'v',subroot_output_path,matrix_type='sino')

            # Compute one ADMM iteration (x, v, u)
            print('xxxxxxxxxxxxxxxxxxxxx')
            for k in range(-1,hyperparameters_config["nb_iter_second_admm"]):
                # Initialize variables for command line
                if (k == -1):
                    if (i == 0):   # choose initial self.phantom for CASToR reconstruction
                        initialimage = ' -img ' + self.subroot + 'Data/initialization/' + self.image_init_path_without_extension + '.hdr' if self.image_init_path_without_extension != "" else '' # initializing CASToR MAP reconstruction with image_init or with CASToR default values
                else:
                    initialimage = ' -img ' + subroot_output_path + '/' + subdir + '/' + format(i) + '_' + format(k) + '_x.hdr'

                base_name_k = format(i) + '_' + format(k)
                base_name_k_next = format(i) + '_' + format(k+1)
                full_output_path_k = subroot_output_path + '/' + subdir + '/' + base_name_k
                full_output_path_k_next = subroot_output_path + '/' + subdir + '/' + base_name_k_next
                v_for_additional_data = ' -additional-data ' + full_output_path_k + '_v.hdr'
                u_for_additional_data = ' -additional-data ' + full_output_path_k + '_u.hdr'

                # Compute one ADMM iteration (x, v, u)
                x_reconstruction_command_line = castor_command_line_x + ' -fout ' + full_output_path_k_next + it + u_for_additional_data + v_for_additional_data + initialimage + f_mu_for_penalty + penalty # we need f-mu so that ADMM optimizer works, even if we will not use it...
                self.compute_x_v_u_ADMM(x_reconstruction_command_line,full_output_path_k_next,subdir,i,k,self.phantom,only_x,subroot_output_path,self.subroot)



'''
import subprocess
root = os.getcwd()
experiment = 24
successful_process = subprocess.call(["python3", root+"/show_castor_results.py", optimizer, str(nb_iter_second_admm), str(experiment),self.suffix,self.phantom,str(beta)]) # Showing results in tensorboard
'''