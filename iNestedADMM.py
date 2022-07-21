## Python libraries

# Useful
import time

import numpy

# Local files to import
from vReconstruction import vReconstruction
from iDenoisingInReconstruction import iDenoisingInReconstruction

class iNestedADMM(vReconstruction):
    def __init__(self,config):
        print('__init__')

    def runComputation(self,config,fixed_config,hyperparameters_config,root):
        print("Nested ADMM reconstruction")

        # Initializing DIP output with f_init
        self.f = self.f_init

        # Initializing results class
        if ((fixed_config["average_replicates"] and self.replicate == 1) or (fixed_config["average_replicates"] == False)):
            from iResults import iResults
            classResults = iResults(config)
            classResults.nb_replicates = self.nb_replicates
            classResults.rho = self.rho
            classResults.debug = self.debug
            classResults.initializeSpecific(fixed_config,hyperparameters_config,root)
        
        if (fixed_config["method"] == "Gong"):
            i_init = 0 #-1 after MIC
        else:
            i_init = 0

        for i in range(i_init, self.max_iter):
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Global iteration !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', i)
            start_time_outer_iter = time.time()
            
            if (i != 0 or fixed_config["method"] != "Gong"): # Gong at first epoch -> only pre train the network
                # Block 1 - Reconstruction with CASToR (tomographic reconstruction part of ADMM)
                self.x_label = self.castor_reconstruction(classResults.writer, i, self.subroot, self.sub_iter_PLL, self.experiment, hyperparameters_config, self.method, self.phantom, self.replicate, self.suffix, classResults.image_gt, self.f, self.mu, self.PETImage_shape, self.PETImage_shape_str, self.rho, self.alpha, self.image_init_path_without_extension) # without ADMMLim file
                # Write corrupted image over ADMM iterations
                classResults.writeCorruptedImage(i,hyperparameters_config["sub_iter_PLL"],self.x_label,self.suffix,pet_algo="nested ADMM")

            # Block 2 - CNN
            start_time_block2= time.time()
            if (i == 0 and fixed_config["method"] == "Gong"): # Gong at first epoch -> only pre train the network
                # Create label corresponding to initial value of image_init
                #x_label = self.fijii_np(self.subroot_data + 'Data/initialization/' + self.image_init_path_without_extension + '.img',shape=(self.PETImage_shape),type='<f')
                # Fit MLEM 60it for first global iteration
                #x_label = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'MLEM_it60_REF_cropped.img',shape=(self.PETImage_shape),type='<f')
                x_label = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'MLEM_it300_REF_cropped.img',shape=(self.PETImage_shape),type='<d')
                self.save_img(x_label,self.subroot+'Block2/x_label/' + format(self.experiment)+'/'+ format(-1) +'_x_label' + self.suffix + '.img')

                # For first epoch, change number of epochs to 300
                sub_iter_DIP = hyperparameters_config["sub_iter_DIP"]
                hyperparameters_config["sub_iter_DIP"] = 300
                self.sub_iter_DIP = 300
                classDenoising = iDenoisingInReconstruction(hyperparameters_config,i-1)
                # Set sub_iter_DIP back to initial value
                hyperparameters_config["sub_iter_DIP"] = sub_iter_DIP
            else:
                classDenoising = iDenoisingInReconstruction(hyperparameters_config,i)
            classDenoising.hyperparameters_list = self.hyperparameters_list
            classDenoising.debug = self.debug
            classDenoising.do_everything(config,root)
            print("--- %s seconds - DIP block ---" % (time.time() - start_time_block2))
            
            if (i == 0 and fixed_config["method"] == "Gong"): # Gong at first epoch -> only pre train the network
                self.f = self.fijii_np(self.subroot+'Block2/out_cnn/'+ format(self.experiment)+'/out_' + classDenoising.net + '' + format(i-1) + "_epoch=" + format(hyperparameters_config["sub_iter_DIP"] - 1) + self.suffix + '.img',shape=(self.PETImage_shape),type='<f') # loading DIP output
            else:
                self.f = self.fijii_np(self.subroot+'Block2/out_cnn/'+ format(self.experiment)+'/out_' + classDenoising.net + '' + format(i) + "_epoch=" + format(hyperparameters_config["sub_iter_DIP"] - 1) + self.suffix + '.img',shape=(self.PETImage_shape),type='<f') # loading DIP output
            self.f.astype(numpy.float64)
            
            if (i != 0 or fixed_config["method"] != "Gong"): # Gong at first epoch -> only pre train the network
                # Block 3 - mu update
                self.mu = self.x_label - self.f
                self.save_img(self.mu,self.subroot+'Block2/mu/'+ format(self.experiment)+'/mu_' + format(i) + self.suffix + '.img') # saving mu
                # Write corrupted image over ADMM iterations
                classResults.writeCorruptedImage(i,hyperparameters_config["sub_iter_PLL"],self.mu,self.suffix,pet_algo="mmmmmuuuuuuu")
                print("--- %s seconds - outer_iteration ---" % (time.time() - start_time_outer_iter))
                # Compute IR metric (different from others with several replicates)
                classResults.compute_IR_bkg(self.PETImage_shape,self.f,i,classResults.IR_bkg_recon,self.phantom)
                classResults.writer.add_scalar('Image roughness in the background (best : 0)', classResults.IR_bkg_recon[i], i+1)
                # Write output image and metrics to tensorboard
                classResults.writeEndImagesAndMetrics(i,hyperparameters_config["sub_iter_PLL"],self.PETImage_shape,self.f,self.suffix,self.phantom,classDenoising.net,pet_algo=fixed_config["method"])

        # Saving final image output
        self.save_img(self.f, self.subroot+'Images/out_final/final_out' + self.suffix + '.img')
        
        ## Averaging for VAE
        if (classDenoising.net == 'DIP_VAE'):
            print('Need to code back this part with abstract classes')