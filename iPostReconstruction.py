## Python libraries

# Useful
from datetime import datetime
import numpy as np
import torch

# Local files to import
from vDenoising import vDenoising

class iPostReconstruction(vDenoising):
    def __init__(self,config):
        self.admm_it = 1 # Set it to 1, 0 is for ADMM reconstruction with hard coded values
    
    def initializeSpecific(self,fixed_config,hyperparameters_config,root):
        print("Denoising in post reconstruction")
        vDenoising.initializeSpecific(self,fixed_config,hyperparameters_config,root)
        # Loading DIP x_label (corrupted image) from block1
        self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/' + 'im_corrupt_beginning.img',shape=(self.PETImage_shape),type='<d') # ADMMLim for nested
        #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/' + 'initialization/MLEM_it60_REF_cropped.img',shape=(self.PETImage_shape),type='<f') # MLEM for Gong
        #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/' + 'im_corrupt_beginning_10.img',shape=(self.PETImage_shape),type='<d') # ADMMLim for nested
        self.net_outputs_path = self.subroot+'Block2/out_cnn/' + format(self.experiment) + '/out_' + self.net + '_post_reco_epoch=' + format(0) + self.suffix + '.img'
        self.checkpoint_simple_path = 'runs/' # To log loss in tensorboard thanks to Logger
        self.name_run = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.total_nb_iter = hyperparameters_config["sub_iter_DIP"]
        self.sub_iter_DIP = 1 # For first iteration, then everything is in for loop with total_nb_iter variable
        '''
        ckpt_file_path = self.subroot+'Block2/checkpoint/'+format(self.experiment)  + '/' + suffix_func(hyperparameters_config) + '/' + '/last.ckpt'
        my_file = Path(ckpt_file_path)
        if (my_file.is_file()):
            os.system('rm -rf ' + ckpt_file_path) # Otherwise, pl will use checkpoint from other run
        '''

    def runComputation(self,config,fixed_config,hyperparameters_config,root):

        # Option to store only 10 images like in tensorboard (quicker, for visualization, set it to True by default)
        #all_images = "False" # Only 10 images
        all_images = "True" # Images for all iterations
        #all_images = 1 # Only last image

        # Initializing results class
        if ((fixed_config["average_replicates"] and self.replicate == 1) or (fixed_config["average_replicates"] == False)):
            from iResults import iResults
            classResults = iResults(config)
            classResults.nb_replicates = self.nb_replicates
            classResults.debug = self.debug
            classResults.initializeSpecific(fixed_config,hyperparameters_config,root)

        self.finetuning = 'False' # to ignore last.ckpt file
        self.admm_it = 0 # Set it to 0, to ignore last.ckpt file

        # Initialize variables
        # Scaling of x_label image
        image_corrupt_input_scale,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt = self.rescale_imag(self.image_corrupt,self.scaling_input) # Scaling of x_label image

        # Corrupted image x_label, numpy --> torch
        self.image_corrupt_torch = torch.Tensor(image_corrupt_input_scale)
        # Adding dimensions to fit network architecture
        self.image_corrupt_torch = self.image_corrupt_torch.view(1,1,self.PETImage_shape[0],self.PETImage_shape[1],self.PETImage_shape[2])
        if (len(self.image_corrupt_torch.shape) == 5): # if 3D but with dim3 = 1 -> 2D
            self.image_corrupt_torch = self.image_corrupt_torch[:,:,:,:,0]

        # Squeeze image by loading it
        out_descale = self.fijii_np(self.net_outputs_path,shape=(self.PETImage_shape),type='<f') # loading DIP output
        #writer = model.logger.experiment # Assess to new variable, otherwise error : weakly-referenced object ...
     
        classResults.writeBeginningImages(self.suffix,self.image_net_input)
        classResults.writeCorruptedImage(0,self.total_nb_iter,self.image_corrupt,self.suffix,pet_algo="to fit",iteration_name="(post reconstruction)")
        
        # Train model using previously trained network (at iteration before)
        model = self.train_process(self.suffix,hyperparameters_config, self.finetuning, self.processing_unit, self.total_nb_iter, self.method, self.admm_it, self.image_net_input_torch, self.image_corrupt_torch, self.net, self.PETImage_shape, self.experiment, self.checkpoint_simple_path, self.name_run, self.subroot, all_images = all_images)

        # Saving variables
        if (self.net == 'DIP_VAE'):
            out, mu, logvar, z = model(self.image_net_input_torch)
        else:
            out = model(self.image_net_input_torch)

        # Write descaled images in files
        if (all_images == "True"):
            epoch_values = np.arange(0,self.total_nb_iter)
        elif (all_images == "False"):
            if (self.total_nb_iter < 10):
                epoch_values = np.arange(0,self.total_nb_iter)
            else:
                epoch_values = np.arange(0,self.total_nb_iter,self.total_nb_iter//10)
        elif (all_images == 1):
            epoch_values = np.array([self.total_nb_iter-1])

        for epoch in epoch_values:
            net_outputs_path = self.subroot+'Block2/out_cnn/' + format(self.experiment) + '/out_' + self.net + '_post_reco_epoch=' + format(epoch) + '.img'
            out = self.fijii_np(net_outputs_path,shape=(self.PETImage_shape),type='<f')
            out = torch.from_numpy(out)
            # Descale like at the beginning
            out_descale = self.descale_imag(out,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)
            #'''
            # Saving image output
            net_outputs_path = self.subroot+'Block2/out_cnn/' + format(self.experiment) + '/out_' + self.net + '_post_reco_epoch=' + format(epoch) + self.suffix + '.img'
            self.save_img(out_descale, net_outputs_path)
            # Squeeze image by loading it
            out_descale = self.fijii_np(net_outputs_path,shape=(self.PETImage_shape),type='<f') # loading DIP output
            # Saving (now DESCALED) image output
            self.save_img(out_descale, net_outputs_path)

            # Compute IR metric (different from others with several replicates)
            classResults.compute_IR_bkg(self.PETImage_shape,out_descale,epoch,classResults.IR_bkg_recon,self.phantom)
            classResults.writer.add_scalar('Image roughness in the background (best : 0)', classResults.IR_bkg_recon[epoch], epoch+1)
            # Write images over epochs
            classResults.writeEndImagesAndMetrics(epoch,self.total_nb_iter,self.PETImage_shape,out_descale,self.suffix,self.phantom,self.net,pet_algo="to fit",iteration_name="(post reconstruction)",all_images=all_images)
