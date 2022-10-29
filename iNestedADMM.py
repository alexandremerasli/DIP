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
        self.f = np.ones((self.PETImage_shape))
        self.f = self.f.reshape(self.PETImage_shape[::-1])
        # Initialize f at step before
        self.f_before = self.f

        # Initializing results class
        if ((config["average_replicates"] and self.replicate == 1) or (config["average_replicates"] == False)):
            from iResults import iResults
            classResults = iResults(config)
            classResults.nb_replicates = self.nb_replicates
            classResults.rho = self.rho
            classResults.debug = self.debug
            classResults.hyperparameters_list = self.hyperparameters_list
            classResults.initializeSpecific(config,root)
        
        if (config["unnested_1st_global_iter"]):
            i_init = 0
        else:
            i_init = -1

        subroot_output_path = (self.subroot + 'Block1/' + self.suffix)
        path_before_eq_22 = (subroot_output_path + '/before_eq22/')
        import os
        import re
        sorted_files = [filename*(self.has_numbers(filename)) for filename in os.listdir(path_before_eq_22) if os.path.splitext(filename)[1] == '.hdr']
        if (len(sorted_files) > 0):
            sorted_files.sort(key=self.natural_keys)
            last_file = sorted_files[-1]
            i_init = int(re.findall(r'\d+', last_file.split('_')[0])[0])
            config["unnested_1st_global_iter"] = True
        


        for self.global_it in range(i_init, self.max_iter):
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Global iteration !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', self.global_it)
            start_time_outer_iter = time.time()
            
            #if (self.global_it == i_init and not config["unnested_1st_global_iter"]): # enable to avoid pre iteration
            #    continue # enable to avoid pre iteration

            if (self.global_it != i_init or config["unnested_1st_global_iter"]): # Gong after pre iteration
                # Block 1 - Reconstruction with CASToR (tomographic reconstruction part of ADMM)
                #if (self.global_it == i_init + 1 and config["unnested_1st_global_iter"] == False): # enable to avoid pre iteration
                #    self.f = self.fijii_np(self.subroot_data + 'Data/initialization/' + config["f_init"] + '.img',shape=(self.PETImage_shape),type='<f') # enable to avoid pre iteration
                self.x_label, self.x = self.castor_reconstruction(classResults.writer, self.global_it, i_init, self.subroot, config["nb_outer_iteration"], self.experiment, config, self.method, self.phantom, self.replicate, self.suffix, classResults.image_gt, self.f, self.mu, self.PETImage_shape, self.PETImage_shape_str, self.alpha, self.image_init_path_without_extension) # without ADMMLim file
                # Write corrupted image over ADMM iterations
                classResults.writeCorruptedImage(self.global_it,config["nb_outer_iteration"],self.x_label,self.suffix,pet_algo=config["method"])

            # Block 2 - CNN
            start_time_block2= time.time()
            if (self.global_it == i_init and not config["unnested_1st_global_iter"]): # Gong at first epoch -> only pre train the network
                # Create label corresponding to initial reconstructed image to start with
                '''
                #x_label = self.fijii_np(self.subroot_data + 'Data/initialization/' + self.image_init_path_without_extension + '.img',shape=(self.PETImage_shape),type='<f')
                if ( 'nested' in config["method"]): # Nested needs 1 to not add any prior information at the beginning, and to initialize x computation to uniform with 1
                    x_label = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'ADMMLim_100it/replicate_' + str(self.replicate) + '/ADMMLim_it100.img',shape=(self.PETImage_shape),type='<d')
                    x_label = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'ADMMLim_1000it/replicate_' + str(self.replicate) + '/ADMMLim_it1000.img',shape=(self.PETImage_shape),type='<d')
                    x_label = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'BSREM_30it/replicate_' + str(self.replicate) + '/BSREM_it30.img',shape=(self.PETImage_shape),type='<d')
                elif ( 'Gong' in config["method"]): # Fit MLEM 60it for first global iteration
                    x_label = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'MLEM_60it/replicate_' + str(self.replicate) + '/MLEM_it60.img',shape=(self.PETImage_shape),type='<d')
                '''
                if config["image_init_path_without_extension"] == "ADMMLim_it100":
                    x_label = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'ADMMLim_100it/replicate_' + str(self.replicate) + '/ADMMLim_it100.img',shape=(self.PETImage_shape),type='<d')
                elif config["image_init_path_without_extension"] == "ADMMLim_it1000":
                    x_label = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'ADMMLim_1000it/replicate_' + str(self.replicate) + '/ADMMLim_it1000.img',shape=(self.PETImage_shape),type='<d')
                elif config["image_init_path_without_extension"] == "MLEM_it60":
                    x_label = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'MLEM_60it/replicate_' + str(self.replicate) + '/MLEM_it60.img',shape=(self.PETImage_shape),type='<d')
                elif config["image_init_path_without_extension"] == "BSREM_it30":
                    x_label = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'BSREM_30it/replicate_' + str(self.replicate) + '/BSREM_it30.img',shape=(self.PETImage_shape),type='<d')
                

                elif config["image_init_path_without_extension"] == "BSREM_3D_it30":
                    x_label = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'BSREM_3D_it30.img',shape=(self.PETImage_shape),type='<d')
                

                self.save_img(x_label,self.subroot+'Block2/' + self.suffix + '/x_label/' + format(self.experiment)+'/'+ format(i_init) +'_x_label' + self.suffix + '.img')
                #classDenoising.sub_iter_DIP = classDenoising.self.sub_iter_DIP_initial

            if (self.global_it == i_init): # Initialize some variables at first global iteration only
            #if (self.global_it == i_init + 1): # enable to avoid pre iteration


                print("Denoising in reconstruction")
                classDenoising = vDenoising(config,self.global_it)
                classDenoising.global_it = self.global_it
                classDenoising.fixed_hyperparameters_list = self.fixed_hyperparameters_list
                classDenoising.hyperparameters_list = self.hyperparameters_list
                classDenoising.debug = self.debug
                classDenoising.config = self.config
                classDenoising.root = self.root
                classDenoising.method = self.method
                classDenoising.initializeGeneralVariables(config,root)
                classDenoising.initializeSpecific(config,root)
            


            # Loading DIP x_label (corrupted image) from block1
            classDenoising.image_corrupt = self.fijii_np(self.subroot+'Block2/' + self.suffix + '/x_label/' + format(self.experiment)+'/'+ format(self.global_it) +'_x_label' + self.suffix + '.img',shape=(self.PETImage_shape))
            classDenoising.net_outputs_path = self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + '' + format(self.global_it) + self.suffix + '.img'
            classDenoising.checkpoint_simple_path = self.subroot+'Block2/' + self.suffix + '/checkpoint/'
            classDenoising.name_run = ""
            # Train network at current global iteration
            classDenoising.sub_iter_DIP = config["sub_iter_DIP"]
            classDenoising.sub_iter_DIP_initial = config["sub_iter_DIP_initial"]
            classDenoising.global_it = self.global_it

            if (self.global_it == i_init and ((i_init == -1 and not config["unnested_1st_global_iter"]) or (i_init == 0 and config["unnested_1st_global_iter"]))): # TESTCT_random
                classDenoising.global_it = self.global_it
                config["DIP_early_stopping"] = True # WMV for pre iteration, instead of 300 iterations of Gong
                config["all_images_DIP"] = "True"
                classDenoising.sub_iter_DIP = 1000
                classDenoising.sub_iter_DIP_initial = 1000
                print("Denoising in reconstruction")
                classDenoising.initializeSpecific(config,root)

            if (self.global_it == i_init + 1 and ((i_init == -1 and not config["unnested_1st_global_iter"]) or (i_init == 0 and config["unnested_1st_global_iter"]))): # TESTCT_random , put back random input
                classDenoising.global_it = self.global_it
                config["all_images_DIP"] = "Last" # TESTCT_random , put back all images to last to save space
                print("Denoising in reconstruction")
                classDenoising.initializeSpecific(config,root)

            if (self.global_it == self.max_iter - 1): # TESTCT_random    
                classDenoising.global_it = self.global_it
                config["DIP_early_stopping"] = True # WMV for last iteration, instead of 300 iterations of Gong
                config["all_images_DIP"] = "True"
                classDenoising.sub_iter_DIP = 1000
                classDenoising.initializeSpecific(config,root)
                print("Denoising in reconstruction")

            classDenoising.runComputation(config,root)
            
            
            #config["DIP_early_stopping"] = False



            ## Variables for WMV ##
            self.DIP_early_stopping = classDenoising.DIP_early_stopping
            if classDenoising.DIP_early_stopping:
                self.epochStar = classDenoising.epochStar
                self.windowSize = classDenoising.windowSize
                self.patienceNumber = classDenoising.patienceNumber
                self.VAR_recon = classDenoising.VAR_recon
                self.MSE_WMV = classDenoising.MSE_WMV
                self.PSNR_WMV = classDenoising.PSNR_WMV
                self.SSIM_WMV = classDenoising.SSIM_WMV
                self.SUCCESS = classDenoising.SUCCESS
            #if (self.epochStar != classDenoising.sub_iter_DIP - 1): # ES point is reached
                #classDenoising.sub_iter_DIP = self.epochStar + self.patienceNumber + 1
            #    classDenoising.sub_iter_DIP = self.epochStar + 1
            
            print("--- %s seconds - DIP block ---" % (time.time() - start_time_block2))
            
            self.f_before = self.f
            self.f = self.fijii_np(self.subroot+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + classDenoising.net + '' + format(self.global_it) + "_epoch=" + format(classDenoising.sub_iter_DIP - 1) + '.img',shape=(self.PETImage_shape),type='<f') # loading DIP output
            # Saving Final DIP output with name without epochs
            self.save_img(self.f,self.subroot+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + classDenoising.net + '' + format(self.global_it) + "_FINAL" + '.img')
            subroot_output_path = (self.subroot + 'Block2/' + self.suffix)
            # Write hdr with float precision because output of network
            original_FLTNB = self.FLTNB
            self.FLTNB = 'float'
            self.write_hdr(self.subroot,[self.global_it],'out_cnn/' + str(self.experiment),self.phantom,'FINAL',subroot_output_path,additional_name='out_' + self.net)
            self.FLTNB = original_FLTNB

            if config["FLTNB"] == "double":
                self.f = self.f.astype(np.float64)
            
            if (self.global_it != i_init or config["unnested_1st_global_iter"]): # Gong after pre iteration
                # Block 3 - mu update
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
                # Write output image and metrics to tensorboard
                classResults.writeEndImagesAndMetrics(self.global_it,config["nb_outer_iteration"],self.PETImage_shape,self.f,self.suffix,self.phantom,classDenoising.net,pet_algo=config["method"])

            if (self.global_it != i_init): # Do not update rho at Gong pre iteration or at first iteration when rho was 0, and set back to inital value for next iteration
                # Adaptive rho update
                primal_residual_norm = np.linalg.norm((self.x - self.f) / max(np.linalg.norm(self.x),np.linalg.norm(self.f)))
                dual_residual_norm = np.linalg.norm((self.f - self.f_before)) / np.linalg.norm(self.mu)
                if (config["adaptive_parameters_DIP"] == "tau"):
                    new_tau = 1 / config["xi_DIP"] * np.sqrt(primal_residual_norm / dual_residual_norm)
                    if (new_tau >= 1 and new_tau < config["tau_DIP"]):
                        self.tau_DIP = new_tau
                    elif (new_tau < 1 and new_tau > 1 / config["tau_DIP"]):
                        self.tau_DIP = 1 / new_tau
                    else:
                        self.tau_DIP = config["tau_DIP"]
                else:
                    self.tau_DIP = config["tau_DIP"]
                if (config["adaptive_parameters_DIP"] == "rho" or config["adaptive_parameters_DIP"] == "tau"):
                    previous_rho = self.rho
                    if (primal_residual_norm > config["mu_DIP"] * dual_residual_norm):
                        self.rho *= self.tau_DIP
                    elif (dual_residual_norm > config["mu_DIP"] * primal_residual_norm):
                        self.rho /= self.tau_DIP
                    else:
                        print("Keeping rho for next global iteration.")

                    # Do the same scaling for mu
                    coeff_rho = self.rho / previous_rho
                    self.mu /= coeff_rho

                text_file = open(self.subroot + 'Block2/' + self.suffix + '/adaptive_it' + str(self.global_it) + '.log', "a")
                text_file.write("adaptive rho :" + "\n")
                text_file.write(str(self.rho) + "\n")
                text_file.write("adaptive tau :" + "\n")
                text_file.write(str(self.tau_DIP) + "\n")
                text_file.write("adaptive rho :" + "\n")
                text_file.write(str(self.rho) + "\n")
                text_file.write("relPrimal :" + "\n")
                text_file.write(str(primal_residual_norm) + "\n")
                text_file.write("relDual :" + "\n")
                text_file.write(str(dual_residual_norm) + "\n")
                text_file.write("norm of mu(n+1) :" + "\n")
                text_file.write(str(np.linalg.norm(self.mu)) + "\n")
                text_file.close()
                    
            # WMV
            '''
            #if self.DIP_early_stopping:
            if (self.DIP_early_stopping and self.global_it != i_init and self.global_it != self.max_iter - 1):
                classResults.epochStar = self.epochStar
                classResults.VAR_recon = self.VAR_recon
                classResults.MSE_WMV = self.MSE_WMV
                classResults.PSNR_WMV = self.PSNR_WMV
                classResults.SSIM_WMV = self.SSIM_WMV
                classResults.windowSize = self.windowSize
                classResults.patienceNumber = self.patienceNumber
                classResults.SUCCESS = self.SUCCESS
        
                classResults.WMV_plot(config)
            '''
        # Saving final image output
        self.save_img(self.f, self.subroot+'Images/out_final/final_out' + self.suffix + '.img')
        
        ## Averaging for VAE
        if (classDenoising.net == 'DIP_VAE'):
            print('Need to code back this part with abstract classes')