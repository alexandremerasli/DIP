## Python libraries

# Useful
import time
import numpy as np

# Local files to import
from vReconstruction import vReconstruction
from vDenoising import vDenoising

class iADMM_DIP(vReconstruction):
    def __init__(self,config, *args, **kwargs):
        print('__init__')

    def runComputation(self,config,root):
        print("DNA reconstruction")

        # Initialize specific variables
        self.initializeSpecific(config,root)
        
        # Set first iteration according to user choice on initialization run
        if (config["unnested_1st_outer_iter"]):
            i_init = 0
        else:
            i_init = -1

        # Loop on outer iterations
        for self.outer_it in range(i_init, self.max_iter):
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! outer iteration !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', self.outer_it)
            start_time_inner_iter = time.time()

            ####################    Block 1 - Reconstruction with CASToR (tomographic reconstruction part of ADMM)    ####################
            if (self.outer_it != i_init or config["unnested_1st_outer_iter"]):
                print("Reconstruction with CASToR")
                
                # Launch CASToR reconstruction (ADMM-Reg if method is DNA, OPTITR if method is DIPRecon)
                self.x_label, self.x = self.castor_reconstruction(self.classResults.writer, self.outer_it, i_init, self.subroot_phantom, config["nb_inner_iteration"], self.experiment, config, self.method, self.phantom, self.replicate, self.suffix, self.classResults.image_gt, self.f, self.mu, self.PETImage_shape, self.PETImage_shape_str, self.alpha, self.image_init_path_without_extension) # without ADMMReg file
                
                # Write corrupted image (x_label = x_CASToR + mu) for each outer iteration in tensorboard
                self.classResults.writeCorruptedImage(self.outer_it,config["nb_inner_iteration"],self.x_label,self.suffix,pet_algo=self.method)

            ####################    Block 2 - NN    ####################
            start_time_block2= time.time()
            print("Denoising in reconstruction")
            
            # Initialize variables and vDenoising object
            self.initializeSettingsForCurrentIteration(config,i_init,root)
            
            # Launch DIP denoising task
            self.classDenoising.initializeSpecific(config,root)
            self.classDenoising.runComputation(config,root)
            
            # At end of DIP denoising, prepare next outer iteration
            self.end_of_DIP_denoising(i_init, config)
            print("--- %s seconds - DIP block ---" % (time.time() - start_time_block2))

            # Saving Final DIP output with name without number of epochs
            self.save_final_DIP_output(i_init, config)

            # Write header with float precision because output of network is a float image
            self.write_hdr_with_overrided_precision()
            
            ####################    Block 3 - mu update    ####################
            # Cast network output to double if other images are in double
            if config["FLTNB"] == "double":
                self.f = self.f.astype(np.float64)
            
            # Save mu variable and compute metrics
            self.save_mu_and_compute_metrics(config,i_init)

            # Check if DNA stopping criterion is reached
            if (self.checkStoppingCriterion(config, i_init)):
                break

        ### Averaging for VAE
        if (self.classDenoising.net == 'DIP_VAE'):
            raise ValueError('Need to code back this part with abstract classes')

    def initialize_f(self,config):
        # Initialize f to NaN to be sure it was overwritten
        self.f = np.NaN * np.ones((self.PETImage_shape))
        if (config["FLTNB"] == "float"):
            self.f = self.f.astype(np.float32)
        self.f = self.f.reshape(self.PETImage_shape[::-1])
        # Initialize f at step before
        self.f_before = self.f

    def initializeClassResults(self,config,root):
        if ((self.average_replicates and self.replicate == 1) or (self.average_replicates == False)):
            from iResults import iResults
            self.classResults = iResults(config)
            self.assignVariablesFromResults(self.classResults)
            self.assignROI(self.classResults)
            self.classResults.initializeSpecific(config,root)

    def initializeSpecific(self,config,root):
        # Initialize variables from parent class
        vReconstruction.initializeSpecific(self,config,root)
        # Initialize f but is not used in first outer iteration because rho=0, only to define f_mu_for_penalty
        self.initialize_f(config)

        # Initializing results class
        self.initializeClassResults(config,root)

        # Initialize self.classDenoising and other variables
        self.classDenoising = None
        self.tau_DIP = config["tau_DIP"]

    def initializeSettingsForCurrentIteration(self,config,i_init,root):
        # If DNA/DIPRecon initialization
        if ((self.outer_it == i_init and ((i_init == -1 and not config["unnested_1st_outer_iter"])) or (config["unnested_1st_outer_iter"]))): # or (self.outer_it == self.max_iter - 1)): # TESTCT_random
            # Set corrupted image to warm start image (pre reconstructed) and save it in block 2 folder
            x_label = self.fijii_np(self.subroot + 'Data/initialization/' + self.phantom + '/' + config["image_init_path_without_extension"] + '/replicate_' + str(self.replicate) + '/' + config["image_init_path_without_extension"] + '.img',shape=(self.PETImage_shape),type_im='<f')
            self.save_img(x_label,self.subroot_phantom+'Block2/' + self.suffix + '/x_label/' + format(self.experiment)+'/'+ format(i_init) +'_x_label' + self.suffix + '.img')
            
            # Set DIP early stopping or not and corresponding finetuning mode for DIP
            self.set_DIP_ES_and_finetuning(algo_state="init")

            # Set binary images and to save locally and in tensorboard
            self.set_when_to_save_DIP_outputs(config, algo_state="init")
            
            ### Initialize vDenoising object
            self.classDenoising = vDenoising(config,self.outer_it)
            # Put anatomical as input if asked by user (old: mu_DIP = 200 is for random only)
            if (not (i_init == 0 and config["unnested_1st_outer_iter"])):
                if ("override_input_to_anat_init" in config):
                    if (self.net == "DIP" and config["override_input_to_anat_init"]):
                        self.classDenoising.override_input = True
                    else:
                        self.classDenoising.override_input = False
                else:
                    self.classDenoising.override_input = False
            else:
                self.classDenoising.override_input = False

            # Set variable to override or not SC at DNA/DIPRecon initialization
            if ("override_SC_init" in config):
                self.classDenoising.override_SC_init = config['override_SC_init']
            else:
                self.classDenoising.override_SC_init = False

            # Set number of DIP iterations at initialization
            self.classDenoising.sub_iter_DIP_init = config["sub_iter_DIP_init"]

            # Initialize other variables
            self.classDenoising.sub_iter_DIP_already_done = 0
            self.sub_iter_DIP_already_done = 0
            self.classDenoising.fixed_hyperparameters_list = self.fixed_hyperparameters_list
            self.classDenoising.hyperparameters_list = self.hyperparameters_list
            self.classDenoising.config = self.config
            self.classDenoising.root = self.root
            self.classDenoising.method = self.method
            self.classDenoising.scanner = self.scanner
            self.classDenoising.simulation = self.simulation
            self.classDenoising.all_images_DIP = self.all_images_DIP
            self.classDenoising.subroot = self.subroot
            self.classDenoising.checkpoint_simple_path = self.subroot_phantom+'Block2/' + self.suffix + '/checkpoint/'
            self.classDenoising.name_run = ""
            self.classDenoising.initializeGeneralVariables(config,root)
        
        # If DNA/DIPRecon outer iterations
        if (self.outer_it == i_init + 1 and ((i_init == -1 and not config["unnested_1st_outer_iter"]) or (i_init == 0 and config["unnested_1st_outer_iter"]))): # TESTCT_random , put back random input
            # Set DIP early stopping or not and corresponding finetuning mode for DIP
            self.set_DIP_ES_and_finetuning(algo_state="outer")

            # Set binary images and to save locally and in tensorboard
            self.set_when_to_save_DIP_outputs(config, algo_state="outer")

            # Do not override input and SC at DNA/DIPRecon outer iterations
            self.classDenoising.override_input = False
            self.classDenoising.override_SC_init = False

        # Set current outer iteration
        self.classDenoising.outer_it = self.outer_it
        # Set path for DIP outputs at current outer iteration
        self.classDenoising.net_outputs_path = self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + '' + format(self.outer_it) + self.suffix + '.img'
        # Redefine number of DIP iterations to reach adding the ones already done
        self.classDenoising.sub_iter_DIP = config["sub_iter_DIP"] + self.sub_iter_DIP_already_done
        # self.classDenoising.sub_iter_DIP_init = config["DIP_it_if_no_ES_found"] + config["patienceNumber"] # Maximum number of initial DIP iterations is set to DIP_it_if_no_ES_found + patienceNumber
        
        # Loading DIP x_label (corrupted image)
        self.classDenoising.image_corrupt = self.fijii_np(self.subroot_phantom+'Block2/' + self.suffix + '/x_label/' + format(self.experiment)+'/'+ format(self.outer_it) +'_x_label' + self.suffix + '.img',shape=(self.PETImage_shape))
        # If scaling all init, also load corrupted image at initialization for scaling
        if ("scaling_all_init" in config):
            if (config["scaling_all_init"]):
                self.classDenoising.image_corrupt_init = self.fijii_np(self.subroot_phantom+'Block2/' + self.suffix + '/x_label/' + format(self.experiment)+'/'+ format(-1) +'_x_label' + self.suffix + '.img',shape=(self.PETImage_shape))

    def set_DIP_ES_and_finetuning(self,algo_state):
        if (algo_state == "init"):
            if self.DIP_early_stopping_when == "all" or self.DIP_early_stopping_when == "init":
                self.DIP_early_stopping = True
                self.finetuning = "ES" # save NN state at ES point for next outer iteration
            else:
                self.DIP_early_stopping = False
                self.finetuning = "last" # save NN state at last epoch for next outer iteration
        elif (algo_state == "outer"):
            if self.DIP_early_stopping_when == "all":
                self.DIP_early_stopping = True
                self.finetuning = "ES" # save NN state at last epoch for next outer iteration
            else:
                self.DIP_early_stopping = False
                self.finetuning = "last" # save NN state at last epoch for next outer iteration
            # Set DIP_early_stopping and finetuning attributes to classDenoising
            self.classDenoising.DIP_early_stopping = self.DIP_early_stopping
            self.classDenoising.finetuning = self.finetuning
        else:
            raise ValueError("algo_state should be init or outer")

    def set_when_to_save_DIP_outputs(self,config,algo_state):
        self.all_images_DIP_when = config["all_images_DIP_when"]
        if (algo_state == "init"):
            if self.all_images_DIP_when == "True" or self.all_images_DIP_when == "True_init":
                self.all_images_DIP = "True"
            elif self.all_images_DIP_when == "Unique":
                self.all_images_DIP = "Unique"
            elif self.all_images_DIP_when == "False":
                self.all_images_DIP = "False"
            else:
                raise ValueError("Please set all_images_DIP_when to True, True_init, Last or False")
        elif (algo_state == "outer"):
            if self.all_images_DIP_when == "True":
                self.all_images_DIP = "True"
            elif (self.all_images_DIP_when == "Unique" or self.all_images_DIP_when == "True_init"):
                self.all_images_DIP = "Unique"
            elif self.all_images_DIP_when == "False":
                self.all_images_DIP = "False"
            else:
                raise ValueError("Please set all_images_DIP_when to True, True_init, Last or False")
            # Set DIP_early_stopping and finetuning attributes to classDenoising
            self.classDenoising.all_images_DIP = self.all_images_DIP
        else:
            raise ValueError("algo_state should be init or outer")
        
    def end_of_DIP_denoising(self, i_init, config):
        # Update number of iterations useful for next outer iteration
        self.sub_iter_DIP_already_done = self.classDenoising.sub_iter_DIP_already_done
        self.sub_iter_DIP_this_outer_it = self.classDenoising.sub_iter_DIP_this_outer_it
        # If DIP early stopping, update number of iterations already done taking into account ES point iteration or user defined DIP_it_if_no_ES_found
        if (self.DIP_early_stopping):
            # DIP ES point found
            if (self.classDenoising.SUCCESS):
                self.classDenoising.sub_iter_DIP_already_done = self.sub_iter_DIP_already_done - self.classDenoising.patienceNumber
                self.sub_iter_DIP_already_done = self.classDenoising.sub_iter_DIP_already_done
            # DIP ES point not found, set number of iterations to DIP_it_if_no_ES_found
            else:
                self.classDenoising.sub_iter_DIP_already_done = config["DIP_it_if_no_ES_found"]
                self.sub_iter_DIP_already_done = self.classDenoising.sub_iter_DIP_already_done

        # Write GT and DIP input in tensorboard
        self.classResults.writeBeginningImages(self.suffix,self.clssDenoising.image_net_input_scale,self.outer_it)
        # Write corrupted image at DNA/DIPRecon initialization in tensorboard
        if (self.outer_it == i_init):
            self.classResults.writeCorruptedImage(0,self.max_iter,self.classDenoising.image_corrupt,self.suffix,pet_algo="to fit",iteration_name="(post reconstruction)")

    def save_final_DIP_output(self, i_init, config):
        # Set f_before to f for next outer iteration
        self.f_before = self.f
        # At initialization, load DIP output according to DIP early stopping point or not
        if (self.outer_it == i_init):
            if (self.classDenoising.SUCCESS):
                self.f = self.fijii_np(self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.classDenoising.net + '' + format(self.outer_it) + "_epoch=" + format(self.classDenoising.sub_iter_DIP - self.classDenoising.patienceNumber - 1) + '.img',shape=(self.PETImage_shape),type_im='<f')
            else:
                self.f = self.fijii_np(self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.classDenoising.net + '' + format(self.outer_it) + "_epoch=" + format(config["DIP_it_if_no_ES_found"] - 1) + '.img',shape=(self.PETImage_shape),type_im='<f')
        else:
            # When using several DIP inputs, load DIP output with MR input for each outer iteration. Otherwise, load DIP output
            if (self.several_DIP_inputs == 1):
                self.f = self.fijii_np(self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.classDenoising.net + '' + format(self.outer_it) + "_epoch=" + format(self.classDenoising.sub_iter_DIP - 1) + '.img',shape=(self.PETImage_shape),type_im='<f')
            else:
                self.f = self.fijii_np(self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.classDenoising.net + '' + format(self.outer_it) + "_epoch=" + format(self.classDenoising.sub_iter_DIP - 1) + '_batchidx=MR_forward.img',shape=(self.PETImage_shape),type_im='<f')
        # Save loaded image with name without number of epochs
        self.save_img(self.f,self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.classDenoising.net + '' + format(self.outer_it) + "_FINAL" + '.img')
        
    def write_hdr_with_overrided_precision(self):
        # Set precision to float for output of network and keep original precision in original_FLTNB
        original_FLTNB = self.FLTNB
        self.FLTNB = 'float'
        # Path to save header
        subroot_output_path = (self.subroot_phantom + 'Block2/' + self.suffix)
        # Write header with float precision
        self.write_hdr(self.subroot_phantom,[self.outer_it],'out_cnn/' + str(self.experiment),self.phantom,'FINAL',subroot_output_path,additional_name='out_' + self.net)
        # Put back original FLTNB precision
        self.FLTNB = original_FLTNB

    def save_mu_and_compute_metrics(self,config,i_init):
        # if DNA/DIPRecon not at initialization, update mu and save it
        if (self.outer_it > i_init or ((i_init > -1 and not config["unnested_1st_outer_iter"]) or (i_init > 0 and config["unnested_1st_outer_iter"]))):
            # Update mu
            self.mu = self.x_label - self.f
            # Save binary image
            self.save_img(self.mu,self.subroot_phantom+'Block2/' + self.suffix + '/mu/'+ format(self.experiment)+'/mu_' + format(self.outer_it) + self.suffix + '.img')
            # Write corrupted image for each outer iteration in tensorboard
            self.classResults.writeCorruptedImage(self.outer_it,config["nb_inner_iteration"],self.mu,self.suffix,pet_algo="mmmmmuuuuuuu")

        if (self.simulation):
            # Compute IR metric (different from others with several replicates)
            self.classResults.compute_IR_bkg(self.PETImage_shape,self.f,self.outer_it,self.classResults.IR_bkg_recon,self.phantom)
            self.classResults.writer.add_scalar('Image roughness in the background (best : 0)', self.classResults.IR_bkg_recon[self.outer_it], self.outer_it+1)
            # Compute IR in whole phantom (different from others with several replicates)
            self.classResults.compute_IR_whole(self.PETImage_shape,self.f,self.outer_it,self.classResults.IR_whole_recon,self.phantom)
            self.classResults.writer.add_scalar('Image roughness in the phantom', self.classResults.IR_whole_recon[self.outer_it], self.outer_it+1)
        # Write output image and metrics to tensorboard
        self.classResults.writeEndImagesAndMetrics(self.outer_it,config["nb_inner_iteration"],self.PETImage_shape,self.f,self.suffix,self.phantom,self.classDenoising.net,pet_algo=self.method)

    def checkStoppingCriterion(self,config, i_init):
        # This stopping criterion is only for DNA on the brain phantom. It stops the algorithm when the IR in the background is too high (values hardcoded)
        if (self.outer_it != i_init or config["unnested_1st_outer_iter"]): # DNA/DIPRecon not at initialization
            if ("50_" in self.phantom):
                # Check if EMA of IR in the background exists, otherwise set it to IR
                if hasattr(self,"IR_bkg_smoothed"):
                    alpha_IR = 0.6
                    self.IR_bkg_smoothed = (1-alpha_IR) * self.IR_bkg_smoothed + alpha_IR * self.classResults.IR_bkg_recon[self.outer_it]
                else:
                    self.IR_bkg_smoothed = self.classResults.IR_bkg_recon[self.outer_it]
                # Wait for 10 outer iterations before stopping
                if (self.outer_it > 10):
                    # Stop DNA computation if IR in the background is too high
                    if (self.IR_bkg_smoothed > 0.5):
                        print("DNA stopping criterion reached")
                        # Save stopping criterion iteration
                        self.path_stopping_criterion = self.subroot_phantom + 'Block2/' + self.suffix + '/' + 'IR_stopping_criteria.log'
                        stopping_criterion_file = open(self.path_stopping_criterion, "w")
                        stopping_criterion_file.write("stopping iteration :" + "\n")
                        stopping_criterion_file.write(str(self.outer_it) + "\n")
                        stopping_criterion_file.close()
                        return 1
        
        # Stopping criterion not reached
        return 0