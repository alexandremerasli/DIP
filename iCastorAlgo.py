# Useful
from pathlib import Path
import os
import re


# Local files to import
from vReconstruction import vReconstruction

class iCastorAlgo(vReconstruction):
    def __init__(self,config, *args, **kwargs):
        print("__init__")

    def initializeSpecific(self,config,root):
        # Initialize specific variables from parent class
        vReconstruction.initializeSpecific(self,config,root)
        
        # Initialize specific variables according to the reconstruction method
        if (self.method == 'AML'):
            self.A_AML = config["A_AML"]
            self.beta = config["A_AML"]
        elif (self.method == 'APPGML'):
            self.A_AML = config["A_AML"]
            self.beta = self.rho
        elif ('ADMMReg' in self.method):
            self.beta = config["alpha"]
            self.recoInDNA = "ADMMReg"
        elif (self.method == 'BSREM'):
            self.beta = self.rho
        elif (self.method == 'MLEM' or self.method == 'OPTITR' or self.method == 'OSEM'):
            pass
        else:
            raise ValueError("Please define first class attributes for the method " + self.method + " in iCastorAlgo.__init__  (not tested)")

        # Post smoothing by CASToR after reconstruction
        if ("post_smoothing" in config):
            self.post_smoothing = config["post_smoothing"]
        else:
            self.post_smoothing = 0

    def runComputation(self,config,root):

        # Initialize specific variables according to the reconstruction method
        self.initializeSpecific(config,root)

        # Create folder for CASToR output
        Path(self.subroot_phantom + self.suffix + '/').mkdir(parents=True, exist_ok=True)
        
        # Define castor-recon command line according to ADMMReg or other methods
        if ('ADMMReg' in self.method):
            # Call function to run ADMMReg from CASToR
            self.ADMMReg_general(config, 0, self.subroot_phantom + self.suffix)
        else:
            # Define general path until suffix folder
            folder_sub_path = self.subroot_phantom + self.suffix
            # Output path for CASTOR framework
            output_path = ' -fout ' + folder_sub_path + '/' + self.method
            
            # Resume computation if images are already computed
            self.ImageAndItToResumeComputation(folder_sub_path, config)

            # Write bound A in APPGML config file
            if (self.method == "APPGML"):
                self.write_bound_A_in_config_file(folder_sub_path)

            # Print CASToR command line and run it
            print("CASToR command line : ")
            print(self.castor_common_command_line(self.subroot, self.PETImage_shape_str, self.phantom, self.replicate, self.post_smoothing) + self.castor_opti_and_penalty(self.method, self.penalty, self.rho) + self.it_option + output_path + self.initial_image)
            os.system(self.castor_common_command_line(self.subroot, self.PETImage_shape_str, self.phantom, self.replicate, self.post_smoothing) + self.castor_opti_and_penalty(self.method, self.penalty, self.rho) + self.it_option + output_path + self.initial_image)

        # Use NNEPPS post-processing if asked by user for each iteration
        if (config["NNEPPS"]):
            for it in range(1,self.max_iter + 1):
                self.NNEPPS_function(config,it)
        
        # Compute metrics after reconstruction
        if ((config["average_replicates"] and self.replicate == 1) or (config["average_replicates"] == False)):
            # Initialize classResults
            from iResults import iResults
            classResults = iResults(config)
            self.assignVariablesFromResults(classResults)
            self.assignROI(classResults)
            classResults.initializeSpecific(config,root)
            # Compute metrics
            classResults.runComputation(config,root)

    def write_bound_A_in_config_file(self,folder_sub_path):
        try:
            # Read APPGML config file
            with open(folder_sub_path  + '/' + 'APPGML.conf', 'r') as read_config_file:
                data = read_config_file.readlines()
        except:
            # If APPGML.conf does not exist, read configuration from APPGML_no_replicate.conf
            with open(self.subroot + 'APPGML_no_replicate.conf', "r") as read_config_file:
                data = read_config_file.readlines()
        # Loop on lines
        for line_idx in range(len(data)):
            # Find line on bound A
            line = data[line_idx]
            if line.startswith("bound"):
                # Replace bound value
                data[line_idx] = "bound: " + str(self.A_AML) + "\n"
        # Write everything in APPGML_no_replicate.conf
        with open(folder_sub_path  + '/' + 'APPGML.conf', "w") as write_config_file:
            write_config_file.writelines(data)

    def NNEPPS_function(self,config,it):
        executable='removeNegativeValues.exe'

        if ('ADMMReg' in self.method):
            i = 0
            subdir = 'ADMM' + '_' + str(config["nb_threads"])
            subdir = ''
            input_without_extension = self.subroot_phantom + self.suffix + '/' +  subdir  + '/' + format(i) + '_' + str(it) + '_it' + format(config["nb_inner_sub_iteration"])
        else:
            input_without_extension = self.subroot_phantom + self.suffix + '/' + self.method + '_beta_' + str(self.beta) + '_it' + format(it)
        
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