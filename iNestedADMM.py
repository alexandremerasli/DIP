## Python libraries

# Useful
import time

# Local files to import
from utils.utils_func import *
from vReconstruction import vReconstruction
from iDenoisingInReconstruction import iDenoisingInReconstruction

class iNestedADMM(vReconstruction):
    def __init__(self,config,args,root):
        vReconstruction.__init__(self,config,args,root)
        # Initializing DIP output with f_init
        self.f = self.f_init

    def runReconstruction(self,config,args,root):
        print("Nested ADMM reconstruction")

        # Initializing results class
        from Results import Results
        classResults = Results(config,args,root,self.max_iter,self.PETImage_shape,self.phantom,self.subroot)

        for i in range(self.max_iter):
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Outer iteration !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', i)
            start_time_outer_iter = time.time()
            
            # Reconstruction with CASToR (first equation of ADMM)
            if (config["method"] == 'Gong'):
                subroot_output_path = self.subroot + 'Block1/' + self.suffix # + '/' # Output path for CASTOR framework
                input_path = ' -img ' + self.subroot + 'Block1/' + self.suffix + '/out_eq22/' # Input path for CASTOR framework
                self.x_label = castor_reconstruction_OPTITR(i, self.subroot, self.sub_iter_MAP, self.test, subroot_output_path, input_path, config, self.suffix, self.f, self.mu, self.PETImage_shape, self.image_init_path_without_extension)
            else: # Nested ADMM
                self.x_label = castor_reconstruction(classResults.writer, i, self.subroot, self.sub_iter_MAP, self.test, config, self.suffix, classResults.image_gt, self.f, self.mu, self.PETImage_shape, self.PETImage_shape_str, self.rho, self.alpha, self.image_init_path_without_extension) # without ADMMLim file

            # Write corrupted image over ADMM iterations
            classResults.writeCorruptedImage(i,self.max_iter,self.x_label,pet_algo="nested ADMM")

            # Block 2 - CNN
            start_time_block2= time.time()            
            classDenoising = iDenoisingInReconstruction(config,args,root,i)
            classDenoising.do_everything(config,args,root)
            print("--- %s seconds - DIP block ---" % (time.time() - start_time_block2))
            self.f = fijii_np(self.subroot+'Block2/out_cnn/'+ format(self.test)+'/out_' + classDenoising.net + '' + format(i) + self.suffix + '.img',shape=(self.PETImage_shape)) # loading DIP output

            # Block 3 - mu update
            self.mu += self.x_label - self.f
            save_img(self.mu,self.subroot+'Block2/mu/'+ format(self.test)+'/mu_' + format(i) + self.suffix + '.img') # saving mu
            # Write corrupted image over ADMM iterations
            classResults.writeCorruptedImage(i,self.max_iter,self.mu,pet_algo="mmmmmuuuuuuu")

            print("--- %s seconds - outer_iteration ---" % (time.time() - start_time_outer_iter))

            # Write output image and metrics to tensorboard
            classResults.writeEndImages(i,self.max_iter,self.PETImage_shape,self.f,self.phantom,classDenoising.net,pet_algo="nested ADMM")

        # Saving final image output
        save_img(self.f, self.subroot+'Images/out_final/final_out' + self.suffix + '.img')
        
        ## Averaging for VAE
        if (classDenoising.net == 'DIP_VAE'):
            print('Need to code back this part with abstract classes')