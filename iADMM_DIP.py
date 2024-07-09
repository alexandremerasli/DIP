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
        if (config["unnested_1st_global_iter"]):
            i_init = 0
        else:
            i_init = -1

        # Loop on global iterations
        for self.global_it in range(i_init, self.max_iter):
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Global iteration !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', self.global_it)
            start_time_outer_iter = time.time()
            
            #if (self.global_it == i_init and not config["unnested_1st_global_iter"]): # enable to avoid pre iteration
            #    continue # enable to avoid pre iteration

            ####################    Block 1 - Reconstruction with CASToR (tomographic reconstruction part of ADMM)    ####################
            if (self.global_it != i_init or config["unnested_1st_global_iter"]): # DIPRecon or DNA after pre iteration
                #if (self.global_it == i_init + 1 and config["unnested_1st_global_iter"] == False): # enable to avoid pre iteration
                #    self.f = self.fijii_np(self.subroot + 'Data/initialization/' + config["f_init"] + '.img',shape=(self.PETImage_shape),type_im='<f') # enable to avoid pre iteration
                self.x_label, self.x = self.castor_reconstruction(self.classResults.writer, self.global_it, i_init, self.subroot_phantom, config["nb_outer_iteration"], self.experiment, config, self.method, self.phantom, self.replicate, self.suffix, self.classResults.image_gt, self.f, self.mu, self.PETImage_shape, self.PETImage_shape_str, self.alpha, self.image_init_path_without_extension) # without ADMMReg file
                # Write corrupted image over ADMM iterations
                self.classResults.writeCorruptedImage(self.global_it,config["nb_outer_iteration"],self.x_label,self.suffix,pet_algo=self.method)

            ####################    Block 2 - NN    ####################yy
            start_time_block2= time.time()
            # Create label corresponding to initial reconstructed image to start with
            self.saveLabel(config,i_init)
            # Initialize vDenoising object if pre iteration
            self.classDenoising = self.initializeSettingsForCurrentIteration(config,i_init,root,self.classDenoising)
            # Loading DIP x_label (corrupted image) from block1
            self.classDenoising.image_corrupt = self.fijii_np(self.subroot_phantom+'Block2/' + self.suffix + '/x_label/' + format(self.experiment)+'/'+ format(self.global_it) +'_x_label' + self.suffix + '.img',shape=(self.PETImage_shape))
            if ("scaling_all_init" in config):
                if (config["scaling_all_init"]):
                    self.classDenoising.image_corrupt_init = self.fijii_np(self.subroot_phantom+'Block2/' + self.suffix + '/x_label/' + format(self.experiment)+'/'+ format(-1) +'_x_label' + self.suffix + '.img',shape=(self.PETImage_shape))
            self.classDenoising.net_outputs_path = self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + '' + format(self.global_it) + self.suffix + '.img'
            self.classDenoising.checkpoint_simple_path = self.subroot_phantom+'Block2/' + self.suffix + '/checkpoint/'
            self.classDenoising.name_run = ""
            # Train network at current global iteration
            self.classDenoising.sub_iter_DIP = config["sub_iter_DIP"] + self.sub_iter_DIP_already_done
            self.classDenoising.sub_iter_DIP_initial_and_final = config["sub_iter_DIP_initial_and_final"] # User defined maximum number of initial DIP iterations
            # self.classDenoising.sub_iter_DIP_initial_and_final = config["DIP_it_if_no_ES_found"] + config["patienceNumber"] # Maximum number of initial DIP iterations is set to DIP_it_if_no_ES_found + patienceNumber
            self.classDenoising.global_it = self.global_it
            # Launch denoising task
            print("Denoising in reconstruction")
            self.classDenoising.initializeSpecific(config,root)
            self.classDenoising.runComputation(config,root)
            
            self.sub_iter_DIP_already_done = self.classDenoising.sub_iter_DIP_already_done
            self.sub_iter_DIP_this_global_it = self.classDenoising.sub_iter_DIP_this_global_it
            if (self.DIP_early_stopping):
                if (self.classDenoising.SUCCESS):
                    self.classDenoising.sub_iter_DIP_already_done = self.sub_iter_DIP_already_done - self.classDenoising.patienceNumber
                    self.sub_iter_DIP_already_done = self.classDenoising.sub_iter_DIP_already_done
                else:
                    self.classDenoising.sub_iter_DIP_already_done = config["DIP_it_if_no_ES_found"]
                    self.sub_iter_DIP_already_done = self.classDenoising.sub_iter_DIP_already_done

            self.classResults.writeBeginningImages(self.suffix,self.classDenoising.image_net_input_scale,self.global_it) # Write GT and DIP input
            if (self.global_it == i_init):
                self.classResults.writeCorruptedImage(0,self.max_iter,self.classDenoising.image_corrupt,self.suffix,pet_algo="to fit",iteration_name="(post reconstruction)")


            print("--- %s seconds - DIP block ---" % (time.time() - start_time_block2))
            # Saving Final DIP output with name without epochs, and f from previous iteration for adaptive rho computation
            self.f_before = self.f
            if (self.several_DIP_inputs == 1):
                if (self.global_it == i_init):
                    if (self.classDenoising.SUCCESS):
                        self.f = self.fijii_np(self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.classDenoising.net + '' + format(self.global_it) + "_epoch=" + format(self.classDenoising.sub_iter_DIP - self.classDenoising.patienceNumber - 1) + '.img',shape=(self.PETImage_shape),type_im='<f') # loading DIP output
                    else:
                        self.f = self.fijii_np(self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.classDenoising.net + '' + format(self.global_it) + "_epoch=" + format(config["DIP_it_if_no_ES_found"] - 1) + '.img',shape=(self.PETImage_shape),type_im='<f') # loading DIP output
                else:
                    self.f = self.fijii_np(self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.classDenoising.net + '' + format(self.global_it) + "_epoch=" + format(self.classDenoising.sub_iter_DIP - 1) + '.img',shape=(self.PETImage_shape),type_im='<f') # loading DIP output
            else: # MIC study : save DIP output with MR input (when using several DIP inputs)
                if (self.global_it == i_init):
                    if (self.classDenoising.SUCCESS):
                        self.f = self.fijii_np(self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.classDenoising.net + '' + format(self.global_it) + "_epoch=" + format(self.classDenoising.sub_iter_DIP - self.classDenoising.patienceNumber - 1) + '.img',shape=(self.PETImage_shape),type_im='<f') # loading DIP output
                    else:
                        self.f = self.fijii_np(self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.classDenoising.net + '' + format(self.global_it) + "_epoch=" + format(self.classDenoising.sub_iter_DIP - 1) + '.img',shape=(self.PETImage_shape),type_im='<f') # loading DIP output
                else:
                    self.f = self.fijii_np(self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.classDenoising.net + '' + format(self.global_it) + "_epoch=" + format(self.classDenoising.sub_iter_DIP - 1) + '_batchidx=MR_forward.img',shape=(self.PETImage_shape),type_im='<f') # loading DIP output
            self.save_img(self.f,self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.classDenoising.net + '' + format(self.global_it) + "_FINAL" + '.img')
            subroot_output_path = (self.subroot_phantom + 'Block2/' + self.suffix)
            # Write header with float precision because output of network is a float image
            original_FLTNB = self.FLTNB
            self.FLTNB = 'float'
            self.write_hdr(self.subroot_phantom,[self.global_it],'out_cnn/' + str(self.experiment),self.phantom,'FINAL',subroot_output_path,additional_name='out_' + self.net)
            self.FLTNB = original_FLTNB
            
            ####################    Block 3 - mu update    ####################
            # Cast network output to double if other images are in double
            if config["FLTNB"] == "double":
                self.f = self.f.astype(np.float64)
            # Save mu variable and compute metrics
            if (self.global_it != i_init or config["unnested_1st_global_iter"]): # DIPRecon after pre iteration
                if (self.global_it > i_init or ((i_init > -1 and not config["unnested_1st_global_iter"]) or (i_init > 0 and config["unnested_1st_global_iter"]))): # at first iteration if rho == 0, let mu to 0 to be equivalent to DIPRecon settings
                    self.mu = self.x_label - self.f
                    self.save_img(self.mu,self.subroot_phantom+'Block2/' + self.suffix + '/mu/'+ format(self.experiment)+'/mu_' + format(self.global_it) + self.suffix + '.img') # saving mu
                    # Write corrupted image over ADMM iterations
                    self.classResults.writeCorruptedImage(self.global_it,config["nb_outer_iteration"],self.mu,self.suffix,pet_algo="mmmmmuuuuuuu")
                    print("--- %s seconds - outer_iteration ---" % (time.time() - start_time_outer_iter))
                if (self.simulation):
                    # Compute IR metric (different from others with several replicates)
                    self.classResults.compute_IR_bkg(self.PETImage_shape,self.f,self.global_it,self.classResults.IR_bkg_recon,self.phantom)
                    self.classResults.writer.add_scalar('Image roughness in the background (best : 0)', self.classResults.IR_bkg_recon[self.global_it], self.global_it+1)
                    # Compute IR in whole phantom (different from others with several replicates)
                    self.classResults.compute_IR_whole(self.PETImage_shape,self.f,self.global_it,self.classResults.IR_whole_recon,self.phantom)
                    self.classResults.writer.add_scalar('Image roughness in the phantom', self.classResults.IR_whole_recon[self.global_it], self.global_it+1)
                # Write output image and metrics to tensorboard
                self.classResults.writeEndImagesAndMetrics(self.global_it,config["nb_outer_iteration"],self.PETImage_shape,self.f,self.suffix,self.phantom,self.classDenoising.net,pet_algo=self.method)

            # DNA stopping criterion
            if (self.global_it != i_init or config["unnested_1st_global_iter"]): # DIPRecon after pre iteration
                if (self.phantom == "image50_1"):
                    # if (self.classResults.IR_bkg_recon[self.global_it] > IR_ref[0]):
                    if hasattr(self,"IR_bkg_smoothed"):
                        alpha_IR = 0.6
                        self.IR_bkg_smoothed = (1-alpha_IR) * self.IR_bkg_smoothed + alpha_IR * self.classResults.IR_bkg_recon[self.global_it]
                    else:
                        self.IR_bkg_smoothed = self.classResults.IR_bkg_recon[self.global_it]
                    if (self.global_it > 10):
                        if (self.IR_bkg_smoothed > 0.5):
                            print("DNA stopping criterion reached")
                            self.path_stopping_criterion = self.subroot_phantom + 'Block2/' + self.suffix + '/' + 'IR_stopping_criteria.log'
                            stopping_criterion_file = open(self.path_stopping_criterion, "w")
                            stopping_criterion_file.write("stopping iteration :" + "\n")
                            stopping_criterion_file.write(str(self.global_it) + "\n")
                            stopping_criterion_file.close()
                            break

        # Saving final image output
        self.save_img(self.f, self.subroot_phantom+'Images/out_final/final_out' + self.suffix + '.img')

        ## Averaging for VAE
        if (self.classDenoising.net == 'DIP_VAE'):
            print('Need to code back this part with abstract classes')

    def initialize_f(self,config):
        # Initialize f to NaN to be sure it was overwritten
        self.f = np.NaN * np.ones((self.PETImage_shape))
        if (config["FLTNB"] == "float"):
            self.f = self.f.astype(np.float32)
        self.f = self.f.reshape(self.PETImage_shape[::-1])
        # Initialize f at step before
        self.f_before = self.f

    def initializeClassResults(self,config,root):
        if ((config["average_replicates"] and self.replicate == 1) or (config["average_replicates"] == False)):
            from iResults import iResults
            self.classResults = iResults(config)
            self.assignVariablesFromResults(self.classResults)
            self.assignROI(self.classResults)
            self.classResults.initializeSpecific(config,root)

    def initializeSpecific(self,config,root):
        # Initialize variables from parent class
        vReconstruction.initializeSpecific(self,config,root)
        # Initialize f but is not used in first global iteration because rho=0, only to define f_mu_for_penalty
        self.initialize_f(config)

        # Initializing results class
        self.initializeClassResults(config,root)

        # Initialize self.classDenoising and other variables
        self.classDenoising = None
        self.tau_DIP = config["tau_DIP"]

    def saveLabel(self,config,i_init):
        if (self.global_it == i_init and not config["unnested_1st_global_iter"]): # DIPRecon or DNA at pre iteration -> only pre train the network
            x_label = self.fijii_np(self.subroot + 'Data/initialization/' + self.phantom + '/' + config["image_init_path_without_extension"] + '/replicate_' + str(self.replicate) + '/' + config["image_init_path_without_extension"] + '.img',shape=(self.PETImage_shape),type_im='<f')
            self.save_img(x_label,self.subroot_phantom+'Block2/' + self.suffix + '/x_label/' + format(self.experiment)+'/'+ format(i_init) +'_x_label' + self.suffix + '.img')

    def initializeSettingsForCurrentIteration(self,config,i_init,root,classDenoising):
        # Initialization
        if ((self.global_it == i_init and ((i_init == -1 and not config["unnested_1st_global_iter"])) or (config["unnested_1st_global_iter"]))): # or (self.global_it == self.max_iter - 1)): # TESTCT_random
            if (self.scanner != "mMR_3D"):
                if self.DIP_early_stopping_when == "all" or self.DIP_early_stopping_when == "init":
                    self.DIP_early_stopping = True
                    self.finetuning = "ES" # save NN state at ES point for next global iteration
                else:
                    self.DIP_early_stopping = False
                    self.finetuning = "last" # save NN state at last epoch for next global iteration
            
            self.all_images_DIP_when = config["all_images_DIP_when"]

            if self.all_images_DIP_when == "True" or self.all_images_DIP_when == "True_init":
                self.all_images_DIP = "True"
            elif self.all_images_DIP_when == "Unique":
                self.all_images_DIP = "Unique"
            elif self.all_images_DIP_when == "False":
                self.all_images_DIP = "False"
            else:
                raise ValueError("Please set all_images_DIP_when to True, True_init, Last or False")
            
            # Initialize vDenoising object
            self.classDenoising = vDenoising(config,self.global_it)
            # Put anatomical as input if asked by user (old: mu_DIP = 200 is for random only)
            if (not (i_init == 0 and config["unnested_1st_global_iter"])):
                if ("override_input_to_anat_init" in config):
                    if (self.net == "DIP" and config["override_input_to_anat_init"]):
                        self.classDenoising.override_input = True
                    else:
                        self.classDenoising.override_input = False
                else:
                    self.classDenoising.override_input = False
            else:
                self.classDenoising.override_input = False

            # MIC study
            if ("override_SC_init" in config):
                self.classDenoising.override_SC_init = config['override_SC_init']
            else:
                self.classDenoising.override_SC_init = False

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
            self.classDenoising.initializeGeneralVariables(config,root)
        
        # During iterations
        if (self.global_it == i_init + 1 and ((i_init == -1 and not config["unnested_1st_global_iter"]) or (i_init == 0 and config["unnested_1st_global_iter"]))): # TESTCT_random , put back random input
            if self.DIP_early_stopping_when == "all":
                self.DIP_early_stopping = True
                self.finetuning = "ES" # save NN state at last epoch for next global iteration
            else:
                self.DIP_early_stopping = False
                self.finetuning = "last" # save NN state at last epoch for next global iteration
            self.classDenoising.DIP_early_stopping = self.DIP_early_stopping
            self.classDenoising.finetuning = self.finetuning
            
            if self.all_images_DIP_when == "True":
                self.all_images_DIP = "True"
            elif (self.all_images_DIP_when == "Unique" or self.all_images_DIP_when == "True_init"):
                self.all_images_DIP = "Unique"
            elif self.all_images_DIP_when == "False":
                self.all_images_DIP = "False"
            else:
                raise ValueError("Please set all_images_DIP_when to True, True_init, Last or False")
            
            self.classDenoising.all_images_DIP = self.all_images_DIP

        
            # Put back original input
            if (self.net == "DIP"):
                self.classDenoising.override_input = False
                self.classDenoising.override_SC_init = False

        return self.classDenoising