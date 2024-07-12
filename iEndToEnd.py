## Python libraries

# Useful
from datetime import datetime
import numpy as np
import torch
import os

# Local files to import
from vDenoising import vDenoising

class iEndToEnd(vDenoising):
    def __init__(self,config, *args, **kwargs):
        self.finetuning = 'False' # to ignore last.ckpt file
        self.outer_it = -100 # Set it to -100, to ignore last.ckpt file

    def initializeSpecific(self,config,root, *args, **kwargs):
        print("DNA - End to end reconstruction")
        
        # Initialize variables related to DIP optimization and path to store images
        self.override_input = False
        self.sub_iter_DIP_already_done = 0
        config["sub_iter_DIP"] = config["max_iter"] # Override sub_iter_DIP to max_iter because end to end mode
        self.total_nb_iter = config["sub_iter_DIP"]

        # Initialize specific variables from parent class
        vDenoising.initializeSpecific(self,config,root)
        
        # Initialize paths and metadata to store images
        self.net_outputs_path = self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + '_epoch=' + format(0) + '.img'
        self.checkpoint_simple_path = 'runs/' # To log loss in tensorboard thanks to Logger
        self.name_run = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        ## Variables for MV ##
        self.epochStar = -1
        self.windowSize = config["windowSize"]
        self.patienceNumber = config["patienceNumber"]
        self.SUCCESS = False
        self.VAR_recon = []

        # Initializing results class
        self.initializeClassResults(config,root)

        # Initializing corrupted sinogram for DIP label
        self.initializeCorruptedSinogram(config,root)

    def initializeCorruptedSinogram(self,config,root):
        # Loading DIP y_label (corrupted sinogram, prompts sinogram here)
        self.sinogram_corrupt = self.fijii_np(self.subroot+'Data/database_v2/' + self.phantom + '/' + "simu0"  + '_' + str(config["replicates"]) + '/simu0_' + str(config["replicates"])+  '_pt.s',shape=self.sinogram_shape_transpose,type_im=np.dtype('int16')).astype(np.float32)
        # Scaling of y_label sinogram
        sinogram_corrupt_input_scale,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt = self.rescale_imag(self.sinogram_corrupt,self.scaling_input) # Scaling of y_label sinogram
        # Corrupted sinogram (prompt) y_label, numpy --> torch float32
        self.sinogram_corrupt_torch = torch.Tensor(self.several_DIP_inputs*[sinogram_corrupt_input_scale])
        # Adding dimensions to fit network architecture
        if (self.sinogram_shape[2] == 1): # if 3D but with dim3 = 1 -> 2D
            self.sinogram_corrupt_torch = self.sinogram_corrupt_torch.view(self.several_DIP_inputs,1,self.sinogram_shape[0],self.sinogram_shape[1],self.sinogram_shape[2])
            self.sinogram_corrupt_torch = self.sinogram_corrupt_torch[:,:,:,:,0]
        else: #3D
            self.sinogram_corrupt_torch = self.sinogram_corrupt_torch.view(1,1,self.sinogram_shape[2],self.sinogram_shape[1],self.sinogram_shape[0])
        self.classResults.writeBeginningImages(self.suffix,self.image_net_input)
        self.classResults.writeCorruptedImage(0,self.total_nb_iter,self.sinogram_corrupt,self.suffix,pet_algo="to fit",iteration_name="(post reconstruction)")
        self.classResults.sinogram_corrupt = self.sinogram_corrupt
        
    def runComputation(self,config,root):
        # Before training, list all images already saved to resume computation
        # folder_sub_path = self.subroot_phantom + 'Block2/' + self.suffix + '/out_cnn/' + str(self.experiment)
        # sorted_files = [filename*(self.has_numbers(filename)) for filename in os.listdir(folder_sub_path) if os.path.splitext(filename)[1] == '.img']
        last_iter = -1 # Means start DIP optimization from scratch
        
        # Set binary images and to save locally and in tensorboard
        self.set_when_to_save_DIP_outputs(config, algo_state="outer")

        # Train model
        model = self.train_process(self.param1_scale_im_corrupt, self.param2_scale_im_corrupt, self.scaling_input, self.suffix,config, self.finetuning, self.processing_unit, self.total_nb_iter, self.method, self.outer_it, self.image_net_input_torch, self.sinogram_corrupt_torch, self.net, self.PETImage_shape, self.experiment, self.checkpoint_simple_path, self.name_run, self.subroot_phantom, all_images_DIP = self.all_images_DIP)
        
        # Get MV attributes from model
        self.getMVAttributesFromModel(model)
        # Initialize MV class
        model.initialize_MV(config,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input,self.suffix,self.outer_it,self.sub_iter_DIP,root, self.subroot,self.scanner, self.simulation, self.hyperparameters_list)
        # Descale DIP outputs and save them as binary files
        self.descale_and_save_images(model,config,last_iter)
        # Set MV attributes to classResults
        self.setMVAttributesInResults(model)

    def descale_and_save_images(self, model, config, last_iter):
        # Choose DIP iterations to be descaled
        if (self.all_images_DIP == "True"):
            epoch_values = np.arange(last_iter+1,self.total_nb_iter)
        elif (self.all_images_DIP == "False"):
            epoch_values = np.arange(last_iter+self.total_nb_iter//10,self.total_nb_iter+self.total_nb_iter//10,max((self.total_nb_iter-last_iter+1)//10,1)) - 1
        elif (self.all_images_DIP == "Unique"):
            epoch_values = np.array([self.total_nb_iter-1])

        # Write descaled images in binary files for each iterations
        for epoch in epoch_values:
            # Path to images
            if (self.all_images_DIP == "Unique"):
                net_outputs_path = self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + "/ES_out_" + self.net + format(self.outer_it) + '_epoch=' + format(epoch) + '.img'
            else:
                net_outputs_path = self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + format(self.outer_it) + '_epoch=' + format(epoch) + '.img'
            
            # Load DIP output for current iteration
            out = self.fijii_np(net_outputs_path,shape=(self.PETImage_shape),type_im='<f')

            # Compute MV value for current iteration
            if (model.DIP_early_stopping):
                model.classMV.SUCCESS,model.classMV.VAR_min,model.classMV.stagnate = model.classMV.compute_MV_value(np.copy(out),epoch,model.sub_iter_DIP,model.classMV.queueQ,model.classMV.SUCCESS,model.classMV.VAR_min,model.classMV.stagnate)
                self.VAR_recon = model.classMV.VAR_recon
                self.MSE_MV = model.classMV.MSE_MV
                self.PSNR_MV = model.classMV.PSNR_MV
                self.SSIM_MV = model.classMV.SSIM_MV
                self.epochStar = model.classMV.epochStar
                
                self.patienceNumber = model.classMV.patienceNumber
                self.SUCCESS = model.classMV.SUCCESS
                print(self.VAR_recon)
                if self.SUCCESS:
                    print("SUCCESS MVVVVVVVVVVVVVVVVVV")
            
            # Descale DIP output
            out_descale = self.descale_DIP_output(out,epoch)

            # Compute metrics and add images to tensorboard for current iteration
            if (self.simulation):
                if ("post_reco" not in config["task"]):
                    # Compute IR metric (different from others with several replicates)
                    self.classResults.compute_IR_bkg(self.PETImage_shape,out_descale,epoch,self.classResults.IR_bkg_recon,self.phantom)
                    self.classResults.writer.add_scalar('Image roughness in the background (best : 0)', self.classResults.IR_bkg_recon[epoch], epoch+1)
                    # Compute IR in whole phantom (different from others with several replicates)
                    self.classResults.compute_IR_whole(self.PETImage_shape,out_descale,self.outer_it,self.classResults.IR_whole_recon,self.phantom)
                    self.classResults.writer.add_scalar('Image roughness in the phantom', self.classResults.IR_whole_recon[self.outer_it], self.outer_it+1)
                # Write images over epochs
            self.classResults.writeEndImagesAndMetrics(epoch,self.total_nb_iter,self.PETImage_shape,out_descale,self.suffix,self.phantom,self.net,pet_algo="to fit",iteration_name="(post reconstruction)")

            # Break loop if ES point reached
            if (self.DIP_early_stopping):
                if (model.classMV.SUCCESS):
                    break

    def getMVAttributesFromModel(self,model):
        if (model.DIP_early_stopping):
            self.epochStar = model.classMV.epochStar
            self.patienceNumber = model.classMV.patienceNumber
            self.VAR_recon = model.classMV.VAR_recon
            self.MSE_MV = model.classMV.MSE_MV
            self.PSNR_MV = model.classMV.PSNR_MV
            self.SSIM_MV = model.classMV.SSIM_MV
            self.SUCCESS = model.classMV.SUCCESS
            # Override total_nb_iter if ES point reached
            if (self.SUCCESS): 
                self.total_nb_iter = self.epochStar + self.patienceNumber

    def setMVAttributesInResults(self,model):
        if (model.DIP_early_stopping):
            self.classResults.epochStar = self.epochStar
            self.classResults.VAR_recon = self.VAR_recon
            self.classResults.MSE_MV = self.MSE_MV
            self.classResults.PSNR_MV = self.PSNR_MV
            self.classResults.SSIM_MV = self.SSIM_MV
            self.classResults.patienceNumber = self.patienceNumber
            self.classResults.SUCCESS = self.SUCCESS