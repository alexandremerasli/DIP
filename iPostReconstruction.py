## Python libraries

# Useful
from datetime import datetime
import numpy as np
import torch

# Local files to import
from vDenoising import vDenoising

class iPostReconstruction(vDenoising):
    def __init__(self,config):
        self.finetuning = 'False' # to ignore last.ckpt file
        self.global_it = 0 # Set it to 0, to ignore last.ckpt file

    def initializeSpecific(self,settings_config,fixed_config,hyperparameters_config,root):
        print("Denoising in post reconstruction")
        vDenoising.initializeSpecific(self,settings_config,fixed_config,hyperparameters_config,root)
        # Loading DIP x_label (corrupted image) from block1
        self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/' + 'im_corrupt_beginning.img',shape=(self.PETImage_shape),type='<d') # ADMMLim for nested
        #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/' + 'initialization/MLEM_it60_REF_cropped.img',shape=(self.PETImage_shape),type='<f') # MLEM for Gong
        self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/' + 'im_corrupt_beginning_it100.img',shape=(self.PETImage_shape),type='<d') # ADMMLim for nested
        self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'ADMMLim_it10000.img',shape=(self.PETImage_shape),type='<d') # ADMMLim for nested
        self.net_outputs_path = self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + '_epoch=' + format(0) + '.img'
        self.checkpoint_simple_path = 'runs/' # To log loss in tensorboard thanks to Logger
        self.name_run = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.total_nb_iter = hyperparameters_config["sub_iter_DIP"]

        '''
        ## Variables for WMV ##
        self.epochStar = -1
        self.windowSize = hyperparameters_config["windowSize"]
        self.patienceNumber = hyperparameters_config["patienceNumber"]
        self.SUCCESS = False
        self.VAR_recon = []
        '''

    def runComputation(self,config,settings_config,fixed_config,hyperparameters_config,root):
        # Initializing results class
        if ((settings_config["average_replicates"] and self.replicate == 1) or (settings_config["average_replicates"] == False)):
            from iResults import iResults
            classResults = iResults(config)
            classResults.nb_replicates = self.nb_replicates
            classResults.debug = self.debug
            classResults.initializeSpecific(settings_config,fixed_config,hyperparameters_config,root)

        # Initialize variables
        # Scaling of x_label image
        image_corrupt_input_scale,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt = self.rescale_imag(self.image_corrupt,self.scaling_input) # Scaling of x_label image

        # Corrupted image x_label, numpy --> torch float32
        self.image_corrupt_torch = torch.Tensor(image_corrupt_input_scale)
        # Adding dimensions to fit network architecture
        self.image_corrupt_torch = self.image_corrupt_torch.view(1,1,self.PETImage_shape[0],self.PETImage_shape[1],self.PETImage_shape[2])
        if (len(self.image_corrupt_torch.shape) == 5): # if 3D but with dim3 = 1 -> 2D
            self.image_corrupt_torch = self.image_corrupt_torch[:,:,:,:,0]

        classResults.writeBeginningImages(self.suffix,self.image_net_input)
        classResults.writeCorruptedImage(0,self.total_nb_iter,self.image_corrupt,self.suffix,pet_algo="to fit",iteration_name="(post reconstruction)")
        
        # Train model using previously trained network (at iteration before)
        model = self.train_process(self.param1_scale_im_corrupt, self.param2_scale_im_corrupt, self.scaling_input, self.suffix,hyperparameters_config, self.finetuning, self.processing_unit, self.total_nb_iter, self.method, self.global_it, self.image_net_input_torch, self.image_corrupt_torch, self.net, self.PETImage_shape, self.experiment, self.checkpoint_simple_path, self.name_run, self.subroot, all_images_DIP = self.all_images_DIP)

        ## Variables for WMV ##
        self.epochStar = model.epochStar
        self.windowSize = model.windowSize
        self.patienceNumber = model.patienceNumber
        self.VAR_recon = model.VAR_recon
        self.MSE_WMV = model.MSE_WMV
        self.PSNR_WMV = model.PSNR_WMV
        self.SSIM_WMV = model.SSIM_WMV
        self.SUCCESS = model.SUCCESS
        if (self.SUCCESS): # ES point is reached
            self.total_nb_iter = self.epochStar + self.patienceNumber + 1

        # Saving variables
        if (self.net == 'DIP_VAE'):
            out, mu, logvar, z = model(self.image_net_input_torch)
        else:
            out = model(self.image_net_input_torch)

        # Write descaled images in files
        if (self.all_images_DIP == "True"):
            epoch_values = np.arange(0,self.total_nb_iter)
        elif (self.all_images_DIP == "False"):
            epoch_values = np.arange(0,self.total_nb_iter,max(self.total_nb_iter//10,1))
        elif (self.all_images_DIP == "Last"):
            epoch_values = np.array([self.total_nb_iter-1])

        ## Variables for WMV ##
        queueQ = []
        VAR_min = np.inf
        #model.SUCCESS = False
        stagnate = 0

        for epoch in epoch_values:
            net_outputs_path = self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + format(self.global_it) + '_epoch=' + format(epoch) + '.img'
            out = self.fijii_np(net_outputs_path,shape=(self.PETImage_shape),type='<f')
            out_torch = torch.from_numpy(out)
            # Descale like at the beginning
            out_descale = self.descale_imag(out_torch,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)
            #'''
            # Saving image output
            net_outputs_path = self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + format(self.global_it) + '_epoch=' + format(epoch) + '.img'
            self.save_img(out_descale, net_outputs_path)
            # Squeeze image by loading it
            out_descale = self.fijii_np(net_outputs_path,shape=(self.PETImage_shape),type='<f') # loading DIP output
            # Saving (now DESCALED) image output
            self.save_img(out_descale, net_outputs_path)

            # Compute IR metric (different from others with several replicates)
            classResults.compute_IR_bkg(self.PETImage_shape,out_descale,epoch,classResults.IR_bkg_recon,self.phantom)
            classResults.writer.add_scalar('Image roughness in the background (best : 0)', classResults.IR_bkg_recon[epoch], epoch+1)
            # Write images over epochs
            classResults.writeEndImagesAndMetrics(epoch,self.total_nb_iter,self.PETImage_shape,out_descale,self.suffix,self.phantom,self.net,pet_algo="to fit",iteration_name="(post reconstruction)")
            #classResults.writeEndImagesAndMetrics(epoch,self.total_nb_iter,self.PETImage_shape,out,self.suffix,self.phantom,self.net,pet_algo="to fit",iteration_name="(post reconstruction)")

            '''
            self.SUCCESS,VAR_min,stagnate = self.WMV(out_descale,epoch,queueQ,model.SUCCESS,VAR_min,stagnate)
            if(model.SUCCESS):
                break
            '''

        classResults.epochStar = self.epochStar
        classResults.VAR_recon = self.VAR_recon
        classResults.MSE_WMV = self.MSE_WMV
        classResults.PSNR_WMV = self.PSNR_WMV
        classResults.SSIM_WMV = self.SSIM_WMV
        classResults.windowSize = self.windowSize
        classResults.patienceNumber = self.patienceNumber
        classResults.SUCCESS = self.SUCCESS

        classResults.WMV_plot(fixed_config,hyperparameters_config)