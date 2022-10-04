## Python libraries

# Useful
import time

import numpy as np

# Local files to import
from vReconstruction import vReconstruction
from iDenoisingInReconstruction import iDenoisingInReconstruction

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

        for i in range(i_init, self.max_iter):
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Global iteration !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', i)
            start_time_outer_iter = time.time()
            
            if (i != i_init or config["unnested_1st_global_iter"]): # Gong at first epoch -> only pre train the network
                # Block 1 - Reconstruction with CASToR (tomographic reconstruction part of ADMM)
                self.x_label, self.x = self.castor_reconstruction(classResults.writer, i, self.subroot, config["nb_outer_iteration"], self.experiment, config, self.method, self.phantom, self.replicate, self.suffix, classResults.image_gt, self.f, self.mu, self.PETImage_shape, self.PETImage_shape_str, self.alpha, self.image_init_path_without_extension) # without ADMMLim file
                # Write corrupted image over ADMM iterations
                classResults.writeCorruptedImage(i,config["nb_outer_iteration"],self.x_label,self.suffix,pet_algo="nested ADMM")

            # Block 2 - CNN
            start_time_block2= time.time()
            if (i == i_init and not config["unnested_1st_global_iter"]): # Gong at first epoch -> only pre train the network
                # Create label corresponding to initial reconstructed image to start with
                #x_label = self.fijii_np(self.subroot_data + 'Data/initialization/' + self.image_init_path_without_extension + '.img',shape=(self.PETImage_shape),type='<f')
                if (config["method"] == "nested"): # Nested needs 1 to not add any prior information at the beginning, and to initialize x computation to uniform with 1
                    x_label = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'MLEM_it60_REF_cropped.img',shape=(self.PETImage_shape),type='<f')
                    #x_label = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'ADMMLim_it10000.img',shape=(self.PETImage_shape),type='<d')
                elif (config["method"] == "Gong"): # Fit MLEM 60it for first global iteration
                    x_label = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'MLEM_it60_REF_cropped.img',shape=(self.PETImage_shape),type='<f')
                self.save_img(x_label,self.subroot+'Block2/' + self.suffix + '/x_label/' + format(self.experiment)+'/'+ format(i_init) +'_x_label' + self.suffix + '.img')
                #classDenoising.sub_iter_DIP = classDenoising.self.sub_iter_DIP_initial

            classDenoising = iDenoisingInReconstruction(config,i)
            classDenoising.fixed_hyperparameters_list = self.fixed_hyperparameters_list
            classDenoising.hyperparameters_list = self.hyperparameters_list
            classDenoising.debug = self.debug
            classDenoising.config = self.config
            classDenoising.root = self.root
            classDenoising.do_everything(config,root)

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
            #if (self.epochStar != self.total_nb_iter - 1): # ES point is reached
            #    self.total_nb_iter = self.epochStar + self.patienceNumber + 1
            
            print("--- %s seconds - DIP block ---" % (time.time() - start_time_block2))
            
            self.f_before = self.f
            self.f = self.fijii_np(self.subroot+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + classDenoising.net + '' + format(i) + "_epoch=" + format(classDenoising.sub_iter_DIP - 1) + '.img',shape=(self.PETImage_shape),type='<f') # loading DIP output
            # Saving Final DIP output with name without epochs
            self.save_img(self.f,self.subroot+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + classDenoising.net + '' + format(i) + "FINAL" + '.img')
            
            if config["FLTNB"] == "double":
                self.f = self.f.astype(np.float64)
            
            if (i != i_init or config["unnested_1st_global_iter"]): # Gong at first epoch -> only pre train the network
                # Block 3 - mu update
                self.mu = self.x_label - self.f
                self.save_img(self.mu,self.subroot+'Block2/' + self.suffix + '/mu/'+ format(self.experiment)+'/mu_' + format(i) + self.suffix + '.img') # saving mu
                # Write corrupted image over ADMM iterations
                classResults.writeCorruptedImage(i,config["nb_outer_iteration"],self.mu,self.suffix,pet_algo="mmmmmuuuuuuu")
                print("--- %s seconds - outer_iteration ---" % (time.time() - start_time_outer_iter))
                # Compute IR metric (different from others with several replicates)
                classResults.compute_IR_bkg(self.PETImage_shape,self.f,i,classResults.IR_bkg_recon,self.phantom)
                classResults.writer.add_scalar('Image roughness in the background (best : 0)', classResults.IR_bkg_recon[i], i+1)
                # Write output image and metrics to tensorboard
                classResults.writeEndImagesAndMetrics(i,config["nb_outer_iteration"],self.PETImage_shape,self.f,self.suffix,self.phantom,classDenoising.net,pet_algo=config["method"])

            if (i != i_init or config["unnested_1st_global_iter"]): # Gong at first epoch -> only pre train the network
                # Adaptive rho update
                primal_residual_norm = np.linalg.norm((self.x - self.f) / max(np.linalg.norm(self.x),np.linalg.norm(self.f)))
                dual_residual_norm = np.linalg.norm((self.f - self.f_before) / np.linalg.norm(self.mu))
                if (config["adaptive_parameters_Gong"] == "tau"):
                    new_tau = 1 / config["xi_Gong"] * np.sqrt(primal_residual_norm / dual_residual_norm)
                    if (new_tau >= 1 and new_tau < config["tau_Gong"]):
                        self.tau = new_tau
                    elif (new_tau < 1 and new_tau > 1 / config["tau_Gong"]):
                        self.tau = 1 / new_tau
                    else:
                        self.tau = config["tau_Gong"]
                elif (config["adaptive_parameters_Gong"] == "rho"):
                    if (primal_residual_norm > config["mu_Gong"] * dual_residual_norm):
                        self.rho *= self.tau
                    elif (dual_residual_norm > config["mu_Gong"] * primal_residual_norm):
                        self.rho /= self.tau
                    else:
                        print("Keeping rho for next global iteration.")
                else:
                    print("Keeping rho for next global iteration.")
                    
            # WMV
            if self.DIP_early_stopping:
                classResults.epochStar = self.epochStar
                classResults.VAR_recon = self.VAR_recon
                classResults.MSE_WMV = self.MSE_WMV
                classResults.PSNR_WMV = self.PSNR_WMV
                classResults.SSIM_WMV = self.SSIM_WMV
                classResults.windowSize = self.windowSize
                classResults.patienceNumber = self.patienceNumber
                classResults.SUCCESS = self.SUCCESS
        
                classResults.WMV_plot(config)

        # Saving final image output
        self.save_img(self.f, self.subroot+'Images/out_final/final_out' + self.suffix + '.img')
        
        ## Averaging for VAE
        if (classDenoising.net == 'DIP_VAE'):
            print('Need to code back this part with abstract classes')