## Python libraries

# Useful
import time

import numpy as np

# Local files to import
from vReconstruction import vReconstruction
from iDenoisingInReconstruction import iDenoisingInReconstruction

class iNestedADMM(vReconstruction):
    def __init__(self,config):
        print('__init__')

    def runComputation(self,config,settings_config,fixed_config,hyperparameters_config,root):
        print("Nested ADMM reconstruction")

        '''
        # Initializing DIP output with f_init
        self.f = self.f_init
        '''
        self.f = np.ones((self.PETImage_shape[0],self.PETImage_shape[1],self.PETImage_shape[2]))
        
        # Initializing results class
        if ((settings_config["average_replicates"] and self.replicate == 1) or (settings_config["average_replicates"] == False)):
            from iResults import iResults
            classResults = iResults(config)
            classResults.nb_replicates = self.nb_replicates
            classResults.rho = self.rho
            classResults.debug = self.debug
            classResults.initializeSpecific(settings_config,fixed_config,hyperparameters_config,root)
        
        if (fixed_config["unnested_1st_global_iter"]):
            i_init = 0
        else:
            i_init = -1

        for i in range(i_init, self.max_iter):
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Global iteration !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', i)
            start_time_outer_iter = time.time()
            
            if (i != i_init or fixed_config["unnested_1st_global_iter"]): # Gong at first epoch -> only pre train the network
                # Block 1 - Reconstruction with CASToR (tomographic reconstruction part of ADMM)
                self.x_label = self.castor_reconstruction(classResults.writer, i, self.subroot, hyperparameters_config["nb_outer_iteration"], self.experiment, hyperparameters_config, self.method, self.phantom, self.replicate, self.suffix, classResults.image_gt, self.f, self.mu, self.PETImage_shape, self.PETImage_shape_str, self.alpha, self.image_init_path_without_extension) # without ADMMLim file
                # Write corrupted image over ADMM iterations
                classResults.writeCorruptedImage(i,hyperparameters_config["nb_outer_iteration"],self.x_label,self.suffix,pet_algo="nested ADMM")

            # Block 2 - CNN
            start_time_block2= time.time()
            if (i == i_init and not fixed_config["unnested_1st_global_iter"]): # Gong at first epoch -> only pre train the network
                # Create label corresponding to initial value of image_init
                #x_label = self.fijii_np(self.subroot_data + 'Data/initialization/' + self.image_init_path_without_extension + '.img',shape=(self.PETImage_shape),type='<f')
                if (settings_config["method"] == "nested"): # Nested needs 1 to not add any prior information at the beginning, and to initialize x computation to uniform with 1
                    x_label = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'MLEM_it60_REF_cropped.img',shape=(self.PETImage_shape),type='<f')
                elif (settings_config["method"] == "Gong"): # Fit MLEM 60it for first global iteration
                    x_label = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'MLEM_it60_REF_cropped.img',shape=(self.PETImage_shape),type='<f')
                self.save_img(x_label,self.subroot+'Block2/x_label/' + format(self.experiment)+'/'+ format(i_init) +'_x_label' + self.suffix + '.img')
                #classDenoising.sub_iter_DIP = classDenoising.self.sub_iter_DIP_initial


            '''
            if (not fixed_config["unnested_1st_global_iter"]): # Rho is not zero, so initialize the network with f_init
                # Ininitializing DIP output and first image x with f_init and image_init
                if (self.method == "nested"): # Nested needs 1 to not add any prior information at the beginning, and to initialize x computation to uniform with 1
                    self.f_init = np.ones((self.PETImage_shape[0],self.PETImage_shape[1],self.PETImage_shape[2]))
                elif (self.method == "Gong"): # Gong initialization with 60th iteration of MLEM (normally, DIP trained with this image as label...)
                    #self.f_init = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'BSREM_it30_REF_cropped.img',shape=(self.PETImage_shape),type='<f')
                    self.f_init = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'MLEM_it60_REF_cropped.img',shape=(self.PETImage_shape),type='<f')
            '''

            classDenoising = iDenoisingInReconstruction(hyperparameters_config,i)
            classDenoising.fixed_hyperparameters_list = self.fixed_hyperparameters_list
            classDenoising.hyperparameters_list = self.hyperparameters_list
            classDenoising.debug = self.debug
            classDenoising.do_everything(config,root)
            
            print("--- %s seconds - DIP block ---" % (time.time() - start_time_block2))
            
            self.f = self.fijii_np(self.subroot+'Block2/out_cnn/'+ format(self.experiment)+'/out_' + classDenoising.net + '' + format(i) + "_epoch=" + format(classDenoising.sub_iter_DIP - 1) + self.suffix + '.img',shape=(self.PETImage_shape),type='<f') # loading DIP output
            self.f.astype(np.float64)
            
            if (i != i_init or fixed_config["unnested_1st_global_iter"]): # Gong at first epoch -> only pre train the network
                # Block 3 - mu update
                self.mu = self.x_label - self.f
                self.save_img(self.mu,self.subroot+'Block2/mu/'+ format(self.experiment)+'/mu_' + format(i) + self.suffix + '.img') # saving mu
                # Write corrupted image over ADMM iterations
                classResults.writeCorruptedImage(i,hyperparameters_config["nb_outer_iteration"],self.mu,self.suffix,pet_algo="mmmmmuuuuuuu")
                print("--- %s seconds - outer_iteration ---" % (time.time() - start_time_outer_iter))
                # Compute IR metric (different from others with several replicates)
                classResults.compute_IR_bkg(self.PETImage_shape,self.f,i,classResults.IR_bkg_recon,self.phantom)
                classResults.writer.add_scalar('Image roughness in the background (best : 0)', classResults.IR_bkg_recon[i], i+1)
                # Write output image and metrics to tensorboard
                classResults.writeEndImagesAndMetrics(i,hyperparameters_config["nb_outer_iteration"],self.PETImage_shape,self.f,self.suffix,self.phantom,classDenoising.net,pet_algo=settings_config["method"])

        # Saving final image output
        self.save_img(self.f, self.subroot+'Images/out_final/final_out' + self.suffix + '.img')
        
        ## Averaging for VAE
        if (classDenoising.net == 'DIP_VAE'):
            print('Need to code back this part with abstract classes')