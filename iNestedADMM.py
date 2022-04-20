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
        
        for i in range(self.max_iter):
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Global iteration !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', i)
            start_time_outer_iter = time.time()
            
            # Reconstruction with CASToR (tomographic reconstruction part of ADMM)
            self.x_label = self.castor_reconstruction(classResults.writer, i, self.subroot, self.sub_iter_PLL, self.experiment, hyperparameters_config, self.method, self.phantom, self.replicate, self.suffix, classResults.image_gt, self.f, self.mu, self.PETImage_shape, self.PETImage_shape_str, self.rho, self.alpha, self.image_init_path_without_extension) # without ADMMLim file

            # Write corrupted image over ADMM iterations
            classResults.writeCorruptedImage(i,hyperparameters_config["sub_iter_PLL"],self.x_label,self.suffix,pet_algo="nested ADMM")

            # Block 2 - CNN         
            start_time_block2= time.time()            
            classDenoising = iDenoisingInReconstruction(hyperparameters_config,i)
            classDenoising.hyperparameters_list = self.hyperparameters_list
            classDenoising.debug = self.debug
            classDenoising.do_everything(config,root)
            print("--- %s seconds - DIP block ---" % (time.time() - start_time_block2))
            self.f = self.fijii_np(self.subroot+'Block2/out_cnn/'+ format(self.experiment)+'/out_' + classDenoising.net + '' + format(i) + self.suffix + '.img',shape=(self.PETImage_shape),type='<f') # loading DIP output
            self.f.astype(numpy.float64)
            
            # Block 3 - mu update
            self.mu += self.x_label - self.f
            self.save_img(self.mu,self.subroot+'Block2/mu/'+ format(self.experiment)+'/mu_' + format(i) + self.suffix + '.img') # saving mu
            # Write corrupted image over ADMM iterations
            classResults.writeCorruptedImage(i,hyperparameters_config["sub_iter_PLL"],self.mu,self.suffix,pet_algo="mmmmmuuuuuuu")

            print("--- %s seconds - outer_iteration ---" % (time.time() - start_time_outer_iter))

            # Write output image and metrics to tensorboard
            classResults.writeEndImagesAndMetrics(i,hyperparameters_config["sub_iter_PLL"],self.PETImage_shape,self.f,self.suffix,self.phantom,classDenoising.net,pet_algo="nested ADMM")

        # Saving final image output
        self.save_img(self.f, self.subroot+'Images/out_final/final_out' + self.suffix + '.img')
        
        ## Averaging for VAE
        if (classDenoising.net == 'DIP_VAE'):
            print('Need to code back this part with abstract classes')