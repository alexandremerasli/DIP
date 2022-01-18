## Python libraries

# Useful
from datetime import datetime

# Local files to import
from utils.utils_func import *
from vDenoising import vDenoising

class iPostReconstruction(vDenoising):
    def __init__(self,config):
        self.admm_it = 1 # Set it to 1, 0 is for ADMM reconstruction with hard coded values
    
    def initializeSpecific(self,fixed_config,hyperparameters_config,root):
        print("Denoising in post reconstruction")
        vDenoising.initializeSpecific(self,fixed_config,hyperparameters_config,root)
        # Loading DIP x_label (corrupted image) from block1
        #self.image_corrupt = fijii_np(self.subroot+'Comparison/im_corrupt_beginning.img',shape=(self.PETImage_shape))
        self.image_corrupt = fijii_np(self.subroot+'Comparison/im_corrupt_beginning_optitr.img',shape=(self.PETImage_shape))
        self.net_outputs_path = self.subroot+'Block2/out_cnn/' + format(self.experiment) + '/out_' + self.net + '_post_reco_epoch=' + format(0) + suffix_func(hyperparameters_config) + '.img'
        self.checkpoint_simple_path = 'runs/' # To log loss in tensorboard thanks to Logger
        self.name_run = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.sub_iter_DIP = 1 # For first iteration, then everything is in for loop with max_iter variable
        '''
        ckpt_file_path = self.subroot+'Block2/checkpoint/'+format(self.experiment)  + '/' + suffix_func(hyperparameters_config) + '/' + '/last.ckpt'
        my_file = Path(ckpt_file_path)
        if (my_file.is_file()):
            os.system('rm -rf ' + ckpt_file_path) # Otherwise, pl will use checkpoint from other run
        '''

    def runComputation(self,config,fixed_config,hyperparameters_config,root):

        # Initializing results class
        from iResults import iResults
        classResults = iResults(fixed_config,hyperparameters_config,self.max_iter,self.PETImage_shape,self.phantom,self.subroot,self.suffix,self.net,self.experiment)
        classResults.initializeSpecific(fixed_config,hyperparameters_config,root)

        self.finetuning = 'False' # to ignore last.ckpt file
        vDenoising.runComputation(self,config,fixed_config,hyperparameters_config,root)
        self.admm_it = 0 # Set it to 0, to ignore last.ckpt file
        # Squeeze image by loading it
        out_descale = fijii_np(self.net_outputs_path,shape=(self.PETImage_shape)) # loading DIP output
        #writer = model.logger.experiment # Assess to new variable, otherwise error : weakly-referenced object ...
     
        classResults.writeBeginningImages(self.image_net_input,self.suffix)
        classResults.writeCorruptedImage(0,self.max_iter,self.image_corrupt,self.suffix,pet_algo="to fit",iteration_name="(post reconstruction)")
        
        for epoch in range(0,self.max_iter,self.max_iter//10):      
            if (epoch > 0):
                # Train model using previously trained network (at iteration before)
                model = self.train_process(hyperparameters_config, self.finetuning, self.processing_unit, self.max_iter//10, self.method, self.admm_it, self.image_net_input_torch, self.image_corrupt_torch, self.net, self.PETImage_shape, self.experiment, self.checkpoint_simple_path, self.name_run, self.subroot)
                # Do finetuning now
                self.admm_it = 1 # Set it to 1, to take last.ckpt file into account
                self.finetuning = 'last' # Put finetuning back to 'last' as if we did not split network training

                # Saving variables
                if (self.net == 'DIP_VAE'):
                    out, mu, logvar, z = model(self.image_net_input_torch)
                else:
                    out = model(self.image_net_input_torch)

                # Descale like at the beginning
                out_descale = descale_imag(out,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)
                # Saving image output
                net_outputs_path = self.subroot+'Block2/out_cnn/' + format(self.experiment) + '/out_' + self.net + '_post_reco_epoch=' + format(epoch) + suffix_func(hyperparameters_config) + '.img'
                save_img(out_descale, net_outputs_path)
                # Squeeze image by loading it
                out_descale = fijii_np(net_outputs_path,shape=(self.PETImage_shape)) # loading DIP output
                # Saving (now DESCALED) image output
                save_img(out_descale, net_outputs_path)

            # Write images over epochs
            classResults.writeEndImages(epoch,self.max_iter,self.PETImage_shape,out_descale,self.suffix,self.phantom,self.net,pet_algo="to fit",iteration_name="(post reconstruction)")
