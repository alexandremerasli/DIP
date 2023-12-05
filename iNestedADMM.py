## Python libraries

# Useful
import time

import numpy as np

# Local files to import
from vReconstruction import vReconstruction
from vDenoising import vDenoising

class iNestedADMM(vReconstruction):
    def __init__(self,config, *args, **kwargs):
        print('__init__')

    def runComputation(self,config,root):
        print("Nested ADMM reconstruction")

        # Initialize f but is not used in first global iteration because rho=0, only to define f_mu_for_penalty
        self.f = np.NaN * np.ones((self.PETImage_shape))
        self.f = self.f.reshape(self.PETImage_shape[::-1])
        # Initialize f at step before
        self.f_before = self.f

        # Initialize moving averages of norms of relative residuals
        self.ema_primal = np.zeros(self.max_iter)
        self.ema_dual = np.zeros(self.max_iter)

        # Initializing results class
        if ((config["average_replicates"] and self.replicate == 1) or (config["average_replicates"] == False)):
            from iResults import iResults
            classResults = iResults(config)
            classResults.nb_replicates = self.nb_replicates
            classResults.rho = self.rho
            classResults.debug = self.debug
            classResults.hyperparameters_list = self.hyperparameters_list
            classResults.fixed_hyperparameters_list = self.fixed_hyperparameters_list
            classResults.scanner = self.scanner
            classResults.phantom_ROI = self.phantom_ROI
            if ("3D" not in self.phantom):
                classResults.bkg_ROI = self.bkg_ROI
                classResults.hot_TEP_ROI = self.hot_TEP_ROI
                if (self.phantom == "image50_1"):
                    classResults.hot_TEP_ROI_ref = self.hot_TEP_ROI_ref
                classResults.hot_TEP_match_square_ROI = self.hot_TEP_match_square_ROI
                classResults.hot_perfect_match_ROI = self.hot_perfect_match_ROI
                classResults.hot_MR_recon = self.hot_MR_recon
                classResults.hot_ROI = self.hot_ROI
                classResults.cold_ROI = self.cold_ROI
                classResults.cold_inside_ROI = self.cold_inside_ROI
                classResults.cold_edge_ROI = self.cold_edge_ROI
            classResults.initializeSpecific(config,root)
        
        if (config["unnested_1st_global_iter"]):
            i_init = 0
        else:
            i_init = -1

        subroot_output_path = (self.subroot + 'Block1/' + self.suffix)
        path_before_eq_22 = (subroot_output_path + '/before_eq22/')
        import os
        import re
        # Resume previous computation if there is one
        '''
        sorted_files = [filename*(self.has_numbers(filename)) for filename in os.listdir(path_before_eq_22) if os.path.splitext(filename)[1] == '.hdr']
        if (len(sorted_files) > 0):
            sorted_files.sort(key=self.natural_keys)
            last_file = sorted_files[-1]
            i_init = int(re.findall(r'\d+', last_file.split('_')[0])[0])
            config["unnested_1st_global_iter"] = True
        '''

        # Nested ADMM stopping criterion
        if ('nested' in config["method"]):
            # Compute IR for BSREM initialization image
            #IR_ref = 0.25
            im_BSREM = self.fijii_np(self.subroot_data + 'Data/initialization/' + self.phantom + '/BSREM_30it' + '/replicate_' + str(self.replicate) + '/BSREM_it30.img',shape=(self.PETImage_shape),type_im='<d') # loading BSREM initialization image
            IR_ref = [np.NaN]
            classResults.compute_IR_whole(self.PETImage_shape,im_BSREM,0,IR_ref,self.phantom)
            print("BSREM IR : ",IR_ref[0])

        classDenoising = None
        self.tau_DIP = config["tau_DIP"]
        for self.global_it in range(i_init, self.max_iter):
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Global iteration !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', self.global_it)
            start_time_outer_iter = time.time()
            
            #if (self.global_it == i_init and not config["unnested_1st_global_iter"]): # enable to avoid pre iteration
            #    continue # enable to avoid pre iteration

            ####################    Block 1 - Reconstruction with CASToR (tomographic reconstruction part of ADMM)    ####################
            if (self.global_it != i_init or config["unnested_1st_global_iter"]): # Gong or nested after pre iteration
                #if (self.global_it == i_init + 1 and config["unnested_1st_global_iter"] == False): # enable to avoid pre iteration
                #    self.f = self.fijii_np(self.subroot_data + 'Data/initialization/' + config["f_init"] + '.img',shape=(self.PETImage_shape),type_im='<f') # enable to avoid pre iteration
                self.x_label, self.x = self.castor_reconstruction(classResults.writer, self.global_it, i_init, self.subroot, config["nb_outer_iteration"], self.experiment, config, self.method, self.phantom, self.replicate, self.suffix, classResults.image_gt, self.f, self.mu, self.PETImage_shape, self.PETImage_shape_str, self.alpha, self.image_init_path_without_extension) # without ADMMLim file
                # Write corrupted image over ADMM iterations
                classResults.writeCorruptedImage(self.global_it,config["nb_outer_iteration"],self.x_label,self.suffix,pet_algo=config["method"])

            ####################    Block 2 - CNN    ####################yy
            start_time_block2= time.time()
            # Create label corresponding to initial reconstructed image to start with
            self.saveLabel(config,i_init)
            # Initialize vDenoising object if pre iteration
            classDenoising = self.initializeSettingsForCurrentIteration(config,i_init,root,classDenoising)
            # Loading DIP x_label (corrupted image) from block1
            classDenoising.image_corrupt = self.fijii_np(self.subroot+'Block2/' + self.suffix + '/x_label/' + format(self.experiment)+'/'+ format(self.global_it) +'_x_label' + self.suffix + '.img',shape=(self.PETImage_shape))
            if ("scaling_all_init" in config):
                if (config["scaling_all_init"]):
                    classDenoising.image_corrupt_init = self.fijii_np(self.subroot+'Block2/' + self.suffix + '/x_label/' + format(self.experiment)+'/'+ format(-1) +'_x_label' + self.suffix + '.img',shape=(self.PETImage_shape))
            classDenoising.net_outputs_path = self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + '' + format(self.global_it) + self.suffix + '.img'
            classDenoising.checkpoint_simple_path = self.subroot+'Block2/' + self.suffix + '/checkpoint/'
            classDenoising.name_run = ""
            # Train network at current global iteration
            classDenoising.sub_iter_DIP = config["sub_iter_DIP"] + self.sub_iter_DIP_already_done
            classDenoising.sub_iter_DIP_initial_and_final = config["sub_iter_DIP_initial_and_final"]
            classDenoising.global_it = self.global_it
            # Launch denoising task
            print("Denoising in reconstruction")
            classDenoising.initializeSpecific(config,root)
            classDenoising.runComputation(config,root)
            
            self.sub_iter_DIP_already_done = classDenoising.sub_iter_DIP_already_done
            if (config["DIP_early_stopping"]):
                if (classDenoising.SUCCESS):
                    classDenoising.sub_iter_DIP_already_done = self.sub_iter_DIP_already_done - classDenoising.patienceNumber
                    self.sub_iter_DIP_already_done = classDenoising.sub_iter_DIP_already_done

            # # Copy last checkpoint to file "last.ckpt" or to ES checkpoint 
            # import shutil
            # for file in os.listdir(classDenoising.checkpoint_simple_path_exp):
            #     if (self.finetuning != "ES" or self.global_it >= 0):
            #         if ("epoch" in file):
            #             shutil.copy(os.path.join(classDenoising.checkpoint_simple_path_exp,file),os.path.join(classDenoising.checkpoint_simple_path_exp,"last.ckpt"))
            #             os.remove(os.path.join(classDenoising.checkpoint_simple_path_exp,file))
            #     else:
            #         if (file == "epoch=" + str(classDenoising.epochStar) + "-step=" + str(classDenoising.epochStar) + ".ckpt"):
            #             shutil.copy(os.path.join(classDenoising.checkpoint_simple_path_exp,"epoch=" + str(classDenoising.epochStar) + "-step=" + str(classDenoising.epochStar) + ".ckpt"),os.path.join(classDenoising.checkpoint_simple_path_exp,"last.ckpt"))
            #         # os.remove(os.path.join(classDenoising.checkpoint_simple_path_exp,"epoch=" + str(classDenoising.epochStar) + "-step=" + str(classDenoising.epochStar) + ".ckpt"))
            #         else:
            #             os.remove(os.path.join(classDenoising.checkpoint_simple_path_exp,file))
            classResults.writeBeginningImages(self.suffix,classDenoising.image_net_input_scale,self.global_it) # Write GT and DIP input
            if (self.global_it == i_init):
                classResults.writeCorruptedImage(0,self.max_iter,classDenoising.image_corrupt,self.suffix,pet_algo="to fit",iteration_name="(post reconstruction)")


            print("--- %s seconds - DIP block ---" % (time.time() - start_time_block2))
            # Saving Final DIP output with name without epochs, and f from previous iteration for adaptive rho computation
            self.f_before = self.f
            if (self.several_DIP_inputs == 1):
                self.f = self.fijii_np(self.subroot+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + classDenoising.net + '' + format(self.global_it) + "_epoch=" + format(classDenoising.sub_iter_DIP - 1) + '.img',shape=(self.PETImage_shape),type_im='<f') # loading DIP output
            else: # MIC study : save DIP output with MR input (when using several DIP inputs)
                self.f = self.fijii_np(self.subroot+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + classDenoising.net + '' + format(self.global_it) + "_epoch=" + format(classDenoising.sub_iter_DIP - 1) + '_batchidx=MR_forward.img',shape=(self.PETImage_shape),type_im='<f') # loading DIP output
            self.save_img(self.f,self.subroot+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + classDenoising.net + '' + format(self.global_it) + "_FINAL" + '.img')
            subroot_output_path = (self.subroot + 'Block2/' + self.suffix)
            # Write header with float precision because output of network is a float image
            original_FLTNB = self.FLTNB
            self.FLTNB = 'float'
            self.write_hdr(self.subroot,[self.global_it],'out_cnn/' + str(self.experiment),self.phantom,'FINAL',subroot_output_path,additional_name='out_' + self.net)
            self.FLTNB = original_FLTNB
            
            ####################    Block 3 - mu update    ####################
            # Cast network output to double if other images are in double
            if config["FLTNB"] == "double":
                self.f = self.f.astype(np.float64)
            # Save mu variable and compute metrics
            if (self.global_it != i_init or config["unnested_1st_global_iter"]): # Gong after pre iteration
                if (self.global_it > i_init or ((i_init > -1 and not config["unnested_1st_global_iter"]) or (i_init > 0 and config["unnested_1st_global_iter"]))): # at first iteration if rho == 0, let mu to 0 to be equivalent to Gong settings
                    self.mu = self.x_label - self.f
                    self.save_img(self.mu,self.subroot+'Block2/' + self.suffix + '/mu/'+ format(self.experiment)+'/mu_' + format(self.global_it) + self.suffix + '.img') # saving mu
                    # Write corrupted image over ADMM iterations
                    classResults.writeCorruptedImage(self.global_it,config["nb_outer_iteration"],self.mu,self.suffix,pet_algo="mmmmmuuuuuuu")
                    print("--- %s seconds - outer_iteration ---" % (time.time() - start_time_outer_iter))
                if ("3D" not in self.phantom):
                    # Compute IR metric (different from others with several replicates)
                    classResults.compute_IR_bkg(self.PETImage_shape,self.f,self.global_it,classResults.IR_bkg_recon,self.phantom)
                    classResults.writer.add_scalar('Image roughness in the background (best : 0)', classResults.IR_bkg_recon[self.global_it], self.global_it+1)
                    # Compute IR in whole phantom (different from others with several replicates)
                    classResults.compute_IR_whole(self.PETImage_shape,self.f,self.global_it,classResults.IR_whole_recon,self.phantom)
                    classResults.writer.add_scalar('Image roughness in the phantom', classResults.IR_whole_recon[self.global_it], self.global_it+1)
                # Write output image and metrics to tensorboard
                classResults.writeEndImagesAndMetrics(self.global_it,config["nb_outer_iteration"],self.PETImage_shape,self.f,self.suffix,self.phantom,classDenoising.net,pet_algo=config["method"])

            if (self.global_it != i_init): # Do not update rho at Gong pre iteration or at first iteration when rho was 0, and set back to inital value for next iteration
                # Update rho if adaptive rho was asked
                self.computeAdaptiveRho(config,i_init)
                # Write file with residuals and adaptive values
                self.writeAdaptiveRhoFile()

            # Nested ADMM stopping criterion
            if (self.global_it != i_init or config["unnested_1st_global_iter"]): # Gong after pre iteration
                if (self.phantom == "image50_1"):
                    # if (classResults.IR_bkg_recon[self.global_it] > IR_ref[0]):
                    if hasattr(self,"IR_bkg_smoothed"):
                        alpha_IR = 0.6
                        self.IR_bkg_smoothed = (1-alpha_IR) * self.IR_bkg_smoothed + alpha_IR * classResults.IR_bkg_recon[self.global_it]
                    else:
                        self.IR_bkg_smoothed = classResults.IR_bkg_recon[self.global_it]
                    if (self.global_it > 10):
                        if (self.IR_bkg_smoothed > 0.5):
                            print("Nested ADMM stopping criterion reached")
                            self.path_stopping_criterion = self.subroot + 'Block2/' + self.suffix + '/' + 'IR_stopping_criteria.log'
                            stopping_criterion_file = open(self.path_stopping_criterion, "w")
                            stopping_criterion_file.write("stopping iteration :" + "\n")
                            stopping_criterion_file.write(str(self.global_it) + "\n")
                            stopping_criterion_file.close()
                            break

        # Saving final image output
        self.save_img(self.f, self.subroot+'Images/out_final/final_out' + self.suffix + '.img')

        ## Averaging for VAE
        if (classDenoising.net == 'DIP_VAE'):
            print('Need to code back this part with abstract classes')

    def saveLabel(self,config,i_init):
        if (self.global_it == i_init and not config["unnested_1st_global_iter"]): # Gong or nested at pre iteration -> only pre train the network
            if config["image_init_path_without_extension"] == "ADMMLim_it100":
                x_label = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'ADMMLim_100it/replicate_' + str(self.replicate) + '/ADMMLim_it100.img',shape=(self.PETImage_shape),type_im='<d')
            elif config["image_init_path_without_extension"] == "ADMMLim_it1000":
                x_label = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'ADMMLim_1000it/replicate_' + str(self.replicate) + '/ADMMLim_it1000.img',shape=(self.PETImage_shape),type_im='<d')
            elif config["image_init_path_without_extension"] == "MLEM_it60":
                # x_label = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'MLEM_60it/replicate_' + str(self.replicate) + '/MLEM_it60.img',shape=(self.PETImage_shape),type_im='<d')
                x_label = self.fijii_np(self.subroot_data + 'Data/initialization/' + self.phantom + '/MLEM_60it' + '/replicate_' + str(self.replicate) + '/MLEM_it60.img',shape=(self.PETImage_shape),type_im='<f')
            elif config["image_init_path_without_extension"] == "BSREM_it30":
                try:
                    x_label = self.fijii_np(self.subroot_data + 'Data/initialization/' + self.phantom + '/BSREM_30it' + '/replicate_' + str(self.replicate) + '/BSREM_it30.img',shape=(self.PETImage_shape),type_im='<f')
                except:
                    x_label = self.fijii_np(self.subroot_data + 'Data/initialization/' + self.phantom + '/BSREM_30it' + '/replicate_' + str(self.replicate) + '/BSREM_it30.img',shape=(self.PETImage_shape),type_im='<d')
            elif config["image_init_path_without_extension"] == "BSREM_3D_it30":
                x_label = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'BSREM_3D_it30.img',shape=(self.PETImage_shape),type_im='<d')
            self.save_img(x_label,self.subroot+'Block2/' + self.suffix + '/x_label/' + format(self.experiment)+'/'+ format(i_init) +'_x_label' + self.suffix + '.img')

    def initializeSettingsForCurrentIteration(self,config,i_init,root,classDenoising):
        # If pre or last iteration, do WMV and initialize vDenoising object if pre iteration
        if ((self.global_it == i_init and ((i_init == -1 and not config["unnested_1st_global_iter"])) or (config["unnested_1st_global_iter"]))): # or (self.global_it == self.max_iter - 1)): # TESTCT_random
            if (self.scanner != "mMR_3D"):
                # if (self.phantom != "image50_0"):
                config["DIP_early_stopping"] = True # WMV for pre and last iteration, instead of 300 iterations of Gong
                config["finetuning"] = "ES" # save NN state at ES point for next global iteration
            config["all_images_DIP"] = "True"
            # Pre iteration : put CT as input and initialize vDenoising object
            # if (self.global_it != self.max_iter - 1 or config["unnested_1st_global_iter"]):
            # if (self.global_it != self.max_iter - 1):
            # Initialize vDenoising object
            classDenoising = vDenoising(config,self.global_it)
            # Put CT as input (mu_DIP = 200 is for random only)
            if (not (i_init == 0 and config["unnested_1st_global_iter"])):
                if (self.net == "DIP" and config["mu_DIP"] != 200):
                    classDenoising.override_input = True
                else:
                    classDenoising.override_input = False
            else:
                classDenoising.override_input = False

            # MIC study
            if ("override_SC_init" in config):
                classDenoising.override_SC_init = config['override_SC_init']
            else:
                classDenoising.override_SC_init = False

            # Initialize other variables
            classDenoising.sub_iter_DIP_already_done = 0
            self.sub_iter_DIP_already_done = 0
            classDenoising.fixed_hyperparameters_list = self.fixed_hyperparameters_list
            classDenoising.hyperparameters_list = self.hyperparameters_list
            classDenoising.debug = self.debug
            classDenoising.config = self.config
            classDenoising.root = self.root
            classDenoising.method = self.method
            classDenoising.scanner = self.scanner
            classDenoising.initializeGeneralVariables(config,root)
        
        # During iterations, do not do WMV
        if (self.global_it == i_init + 1 and ((i_init == -1 and not config["unnested_1st_global_iter"]) or (i_init == 0 and config["unnested_1st_global_iter"]))): # TESTCT_random , put back random input
            config["DIP_early_stopping"] = False
            config["finetuning"] = "last" # save NN state at last epoch for next global iteration
            if ("3D" not in self.phantom):
                config["all_images_DIP"] = "Last" # Only save last image to save space
                # config["all_images_DIP"] = "True" # Save all images for 3D to understand DIP behavior
            else:
                config["all_images_DIP"] = "True" # Save all images for 3D to understand DIP behavior
                # config["all_images_DIP"] = "True" # Save all images for 3D to understand DIP behavior
                config["all_images_DIP"] = "Last" # Only save last image to save space
            # Put back original input
            if (self.net == "DIP"):
                classDenoising.override_input = False
                classDenoising.override_SC_init = False

        return classDenoising

    def computeAdaptiveRho(self,config,i_init):
        # Adaptive rho update
        self.primal_residual_norm = np.linalg.norm((self.x - self.f) / max(np.linalg.norm(self.x),np.linalg.norm(self.f)))
        self.dual_residual_norm = np.linalg.norm((self.f - self.f_before)) / np.linalg.norm(self.mu)
        if (config["adaptive_parameters_DIP"] == "both"): # tau_DIP is tau_max if adaptive tau
            new_tau = 1 / config["xi_DIP"] * np.sqrt(self.primal_residual_norm / self.dual_residual_norm)
            if (new_tau >= 1 and new_tau < config["tau_DIP"]):
                self.tau_DIP = new_tau
            elif (new_tau < 1 and new_tau > 1 / config["tau_DIP"]):
                self.tau_DIP = 1 / new_tau
            else:
                self.tau_DIP = config["tau_DIP"]
        else:
            self.tau_DIP = config["tau_DIP"]
        if (config["adaptive_parameters_DIP"] == "rho" or config["adaptive_parameters_DIP"] == "both"):
            previous_rho = self.rho
            #'''
            if (self.primal_residual_norm > config["xi_DIP"] * config["mu_DIP"] * self.dual_residual_norm):
                self.rho *= self.tau_DIP
            elif (self.dual_residual_norm > 1/config["xi_DIP"] * config["mu_DIP"] * self.primal_residual_norm):
                self.rho /= self.tau_DIP
            #'''
            else:
                print("Keeping rho for next global iteration.")

            # Do the same scaling for mu
            coeff_rho = self.rho / previous_rho
            self.mu /= coeff_rho

        if (config["adaptive_parameters_DIP"] == "rho_decay"):
        
            # Compute moving averages of norms of relative residuals
            alpha_EMA = 0.1
            self.ema_primal[self.global_it] = (1-alpha_EMA) * self.ema_primal[self.global_it - 1] + alpha_EMA * self.primal_residual_norm
            self.ema_dual[self.global_it] = (1-alpha_EMA) * self.ema_dual[self.global_it - 1] + alpha_EMA * self.dual_residual_norm

            if ((self.global_it == i_init + 1) or (self.ema_primal[self.global_it] < self.ema_primal[self.global_it - 1]) and (self.ema_dual[self.global_it] < self.ema_dual[self.global_it - 1])):
                self.rho *= self.tau_DIP
                # Do the same scaling for mu
                coeff_rho = self.tau_DIP
                self.mu /= coeff_rho
            else:
                raise ValueError("EMA relative residuals increasing")
    
    def writeAdaptiveRhoFile(self):
        text_file = open(self.subroot + 'Block2/' + self.suffix + '/adaptive_it' + str(self.global_it) + '.log', "w")
        text_file.write("adaptive rho :" + "\n")
        text_file.write(str(self.rho) + "\n")
        text_file.write("adaptive tau :" + "\n")
        text_file.write(str(self.tau_DIP) + "\n")
        text_file.write("relPrimal :" + "\n")
        text_file.write(str(self.primal_residual_norm) + "\n")
        text_file.write("relDual :" + "\n")
        text_file.write(str(self.dual_residual_norm) + "\n")

        text_file.write("norm of x(n+1) - f(n+1) :" + "\n")
        text_file.write(str(np.linalg.norm(self.x - self.f)) + "\n")
        text_file.write("norm of x(n+1) :" + "\n")
        text_file.write(str(np.linalg.norm(self.x)) + "\n")
        text_file.write("norm of f(n+1) :" + "\n")
        text_file.write(str(np.linalg.norm(self.f)) + "\n")
        text_file.write("norm of f(n+1) - f(n) :" + "\n")
        text_file.write(str(np.linalg.norm(self.f - self.f_before)) + "\n")
        text_file.write("norm of mu(n+1) :" + "\n")
        text_file.write(str(np.linalg.norm(self.mu)) + "\n")
        text_file.close()