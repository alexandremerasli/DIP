# Useful
from pathlib import Path
import os
import re


# Local files to import
from vReconstruction import vReconstruction

class iComparison(vReconstruction):
    def __init__(self,config, *args, **kwargs):
        print("__init__")

    def runComputation(self,config,root):

        if (self.method == 'AML' or self.method == 'APGMAP'):
            self.A_AML = config["A_AML"]
        if (self.method == 'AML'):
            self.beta = config["A_AML"]
        elif ('ADMMLim' in self.method):
            self.beta = config["alpha"]
            self.recoInNested = "ADMMLim"
        elif (self.method == 'BSREM' or self.method == 'APGMAP'):
            self.beta = self.rho

        if (self.method != 'BSREM' and self.method != 'nested' and self.method != 'Gong' and self.method != 'APGMAP'):
            self.post_smoothing = config["post_smoothing"]
        else:
            self.post_smoothing = 0

        # castor-recon command line
        if ('ADMMLim' in self.method):
            # Path variables
            subroot_output_path = (self.subroot + self.suffix)
            subdir = 'ADMM' + '_' + str(config["nb_threads"])
            subdir = ''
            f_mu_for_penalty = ' -multimodal ' + self.subroot_data + 'Data/initialization/1_im_value_cropped.hdr' # Will be removed if first global iteration and unnested_1st_global_iter (rho == 0)
            #f_mu_for_penalty = ' -multimodal ' + self.subroot_data + 'Data/initialization/BSREM_it30_REF_cropped.hdr' # Test for DIP_ADMM (will be removed if first global iteration and unnested_1st_global_iter (rho == 0))
            Path(self.subroot + self.suffix + '/' + subdir).mkdir(parents=True, exist_ok=True) # CASToR path
            self.ADMMLim_general(config, 0, subdir, subroot_output_path, f_mu_for_penalty)
        else:
            folder_sub_path = self.subroot + self.suffix
            Path(folder_sub_path).mkdir(parents=True, exist_ok=True) # CASToR path
            output_path = ' -fout ' + folder_sub_path + '/' + self.method # Output path for CASTOR framework
            
            sorted_files = [filename*(self.has_numbers(filename)) for filename in os.listdir(folder_sub_path) if os.path.splitext(filename)[1] == '.hdr']
            #sorted_files = [] # Do not resume computation
            #'''
            if (len(sorted_files) > 0):
                it = ' -it ' + str(self.max_iter) + ':' + str(config["nb_subsets"])
                initialimage, it, last_iter = self.ImageAndItToResumeComputation(sorted_files,it,folder_sub_path)
            else:
                initialimage = ''
                it = ' -it ' + str(self.max_iter) + ':' + str(config["nb_subsets"])

            if (self.method == "APGMAP"):
                # Write shift A in config
                # Read lines in config file
                try:
                    with open(folder_sub_path  + '/' + 'APPGML.conf', 'r') as read_config_file:
                        data = read_config_file.readlines()
                except:
                    with open(folder_sub_path  + '/' + 'APPGML.conf', "w") as write_config_file:
                        with open(self.subroot_data + 'APPGML_no_replicate.conf', "r") as read_config_file:
                            write_config_file.write(read_config_file.read())
                    with open(folder_sub_path  + '/' + 'APPGML.conf', 'r') as read_config_file:
                        data = read_config_file.readlines()
                    # Change the line with shift
                for line_idx in range (len(data)):
                    line = data[line_idx]
                    if line.startswith("bound"):
                        data[line_idx] = "bound: " + str(self.A_AML) + "\n"
                # Write everything back
                with open(folder_sub_path  + '/' + 'APPGML.conf', "w") as write_config_file:
                    write_config_file.writelines(data)


            print("CASToR command line : ")
            print(self.castor_common_command_line(self.subroot_data, self.PETImage_shape_str, self.phantom, self.replicate, self.post_smoothing) + self.castor_opti_and_penalty(self.method, self.penalty, self.rho) + it + output_path + initialimage)
            os.system(self.castor_common_command_line(self.subroot_data, self.PETImage_shape_str, self.phantom, self.replicate, self.post_smoothing) + self.castor_opti_and_penalty(self.method, self.penalty, self.rho) + it + output_path + initialimage)

        # NNEPPS
        if ('ADMMLim' in self.method):
            max_it = config["nb_outer_iteration"]
        else:
            max_it = config["max_iter"]
        
        if (config["NNEPPS"]):
            print("NNEPPS")        
            for it in range(1,max_it + 1):
                self.NNEPPS_function(config,it)
        
        # Initializing results class
        if ((config["average_replicates"] and self.replicate == 1) or (config["average_replicates"] == False)):
            from iResults import iResults
            classResults = iResults(config)
            self.assignVariablesFromResults(classResults)
            self.assignROI(classResults)
            classResults.initializeSpecific(config,root)
            classResults.runComputation(config,root)

    def NNEPPS_function(self,config,it):
        executable='removeNegativeValues.exe'

        if ('ADMMLim' in self.method):
            i = 0
            subdir = 'ADMM' + '_' + str(config["nb_threads"])
            subdir = ''
            input_without_extension = self.subroot + self.suffix + '/' +  subdir  + '/' + format(i) + '_' + str(it) + '_it' + format(config["nb_inner_iteration"])
        else:
            input_without_extension = self.subroot + self.suffix + '/' + self.method + '_beta_' + str(self.beta) + '_it' + format(it)
        
        input = ' -i ' + input_without_extension + '.img'
        output = ' -o ' + input_without_extension + '_NNEPPS' # Without extension !
        
        # The following 9 commands can be used to specify to which part of the image the NNEPPS has to be applied. You can set dim, min, and max as you wish, provided they are consistent. The default value of min is 0. Note that if you specify dim and max, min is automatically set to the correct value.

        dimX=' -dimX ' + str(self.PETImage_shape[0])
        dimY=' -dimY ' + str(self.PETImage_shape[1])
        dimZ=' -dimZ ' + str(self.PETImage_shape[2])

        minX=''
        minY=''
        minZ=''

        maxX=''
        maxY=''
        maxZ='' #' -maxZ 3'

        # The two following variables are the full size of the input image. They are important for a correct reading of the data. If unset, they are assumed to be equal to the previous max value.
        inputSizeX=' -inputSizeX ' + str(self.PETImage_shape[0])
        inputSizeY=' -inputSizeY ' + str(self.PETImage_shape[1])
        inputSizeZ=' -inputSizeZ ' + str(self.PETImage_shape[2])

        nbThreads='' #'-th 8' Don't use this option if you want to use all threads

        # The 3 following lines give the coefficients assigned to the neighbors in each of the three dimensions (only the 1st-order neighbors are considered). They must sum up to 0.5. If voxels are square, the natural choice is 1/6 for each (default value). In the example, other values are provided to favor close neighbors because voxels are cuboids. See the supplementary material for further explanation of these numbers. Note that the value 0 is forbidden. If you are using 1D or 2D images, provide any value to the unused dimensions, and the code will adapt to the fact that the dimensions do not exist. For example, for square pixels using x and y dimensions, 1/6;1/6;1/6 is equivalent to 0.2;0.2;0.1 and to 0.1;0.1;0.3.
        coeffX=' -coeffX 0.108882'
        coeffY=' -coeffY 0.108882'
        coeffZ=' -coeffZ 0.282236'

        skip_initialization=' -skip_initialization' #'-skip_initialization' #Use this option if you want to skip the initialization step.
        critere_stop_init='' #'-critere_stop_init 1.0e-4' by default. Criterion used to stop the initialization step, the lower, the longer the initialization step will be. Unused if -skip_initialization is set.
        skip_algebraic='' #'-skip_algebraic'#Use this option only if you want to skip the main algebraic part and directly write the image after the initialization step.
        precision=' -precision -1' #'-precision 1.0e-3' by default. Use -1 for maximum precision. This is the relative precision used by the main algebraic part to proceed. Unused if -skip_algebraic is set.

        #input and output type. This doesn't affect the precision of the computation, which is always done using doubles. Two possibilities : float or double. Default value: float
        input_type='' #-input_type double'
        output_type='' #-output_type double'

        #Command line (do not modify):
        NNEPPS_command_line = executable + input + output + dimX + dimY + dimZ + nbThreads + coeffX + coeffY + coeffZ + precision + skip_initialization + critere_stop_init + minX + minY + minZ + maxX + maxY + maxZ + inputSizeX + inputSizeY + input_type + output_type + skip_algebraic
        print(NNEPPS_command_line)
        os.system(NNEPPS_command_line)