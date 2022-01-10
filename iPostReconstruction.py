## Python libraries

# Pytorch
import torch

# Useful
import os
from datetime import datetime

# Math
import numpy as np
import matplotlib.pyplot as plt

# Local files to import
from utils.utils_func import *
from vDenoising import vDenoising
#from vReconstruction import vReconstruction

class iPostReconstruction(vDenoising):
    def __init__(self,config,args,root):
        self.admm_it = 1 # Set it to 1, 0 is for ADMM reconstruction with hard coded values
        vDenoising.__init__(self,config,args,root)
    
    def initializeSpecific(self, config, args, root):
        print("Denoising in post reconstruction")
        vDenoising.initializeSpecific(self,config,args,root)
        # Loading DIP x_label (corrupted image) from block1
        self.image_corrupt = fijii_np(self.subroot+'Comparison/im_corrupt_beginning.img',shape=(self.PETImage_shape))
        self.net_outputs_path = self.subroot+'Block2/out_cnn/' + format(self.test) + '/out_' + self.net + '_post_reco_epoch=' + format(0) + suffix_func(config) + '.img'
        self.checkpoint_simple_path = 'runs/' # To log loss in tensorboard thanks to Logger
        self.name_run = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.sub_iter_DIP = 1 # For first iteration, then everything is in for loop with max_iter variable
        '''
        ckpt_file_path = self.subroot+'Block2/checkpoint/'+format(self.test)  + '/' + suffix_func(self.config) + '/' + '/last.ckpt'
        my_file = Path(ckpt_file_path)
        if (my_file.is_file()):
            os.system('rm -rf ' + ckpt_file_path) # Otherwise, pl will use checkpoint from other run
        '''

    def runDenoiser(self,root):

        # Initializing results class
        from Results import Results
        classResults = Results(self.config,self.args,root,self.max_iter,self.PETImage_shape,self.phantom)

        self.finetuning = 'False' # to ignore last.ckpt file
        vDenoising.runDenoiser(self,root)
        self.admm_it = 0 # Set it to 0, to ignore last.ckpt file
        # Squeeze image by loading it
        out_descale = fijii_np(self.net_outputs_path,shape=(self.PETImage_shape)) # loading DIP output
        #writer = model.logger.experiment # Assess to new variable, otherwise error : weakly-referenced object ...
     
        classResults.writeBeginningImages(self.image_net_input)
        classResults.writeCorruptedImage(0,self.max_iter,self.image_corrupt,pet_algo="to fit",iteration_name="(post reconstruction)")
        
        for epoch in range(0,self.max_iter,self.max_iter//10):      
            if (epoch > 0):
                # Train model using previously trained network (at iteration before)
                model = self.train_process(self.config, self.finetuning, self.processing_unit, self.max_iter//10, self.admm_it, self.image_net_input_torch, self.image_corrupt_torch, self.net, self.PETImage_shape, self.test, self.checkpoint_simple_path, self.name_run, self.subroot)
                # Do finetuning now
                self.admm_it = 1 # Set it to 1, to take last.ckpt file into account
                self.finetuning = 'last' # Put finetuning back to 'last' as if we did not split network training
                #write_image_tensorboard(self.writer,self.image_net_input,"TEST",suffix_func(self.config),self.image_gt,epoch,full_contrast=True) # DIP input in tensorboar

                # Saving variables
                if (self.net == 'DIP_VAE'):
                    out, mu, logvar, z = model(self.image_net_input_torch)
                else:
                    out = model(self.image_net_input_torch)

                # Descale like at the beginning
                out_descale = descale_imag(out,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)
                # Saving image output
                net_outputs_path = self.subroot+'Block2/out_cnn/' + format(self.test) + '/out_' + self.net + '_post_reco_epoch=' + format(epoch) + suffix_func(self.config) + '.img'
                save_img(out_descale, net_outputs_path)
                # Squeeze image by loading it
                out_descale = fijii_np(net_outputs_path,shape=(self.PETImage_shape)) # loading DIP output
                # Saving (now DESCALED) image output
                save_img(out_descale, net_outputs_path)

            # Write images over epochs
            classResults.writeEndImages(epoch,self.max_iter,self.PETImage_shape,out_descale,self.phantom,self.net,pet_algo="to fit",iteration_name="(post reconstruction)")
