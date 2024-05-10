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
        self.global_it = -100 # Set it to -100, to ignore last.ckpt file

    def initializeSpecific(self,config,root, *args, **kwargs):
        print("DNA - End to end reconstruction")

        self.override_input = False
        self.sub_iter_DIP_already_done = 0
        vDenoising.initializeSpecific(self,config,root)
        # Loading DIP y_label (corrupted sinogram, prompts)
        # self.sinogram_corrupt = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "simu0"  + '_' + str(config["replicates"]) + '/simu0_' + str(config["replicates"])+  '_pt.s',shape=(self.sinogram_shape),type_im=np.dtype('int16'))
        self.sinogram_corrupt = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "simu0"  + '_' + str(config["replicates"]) + '/simu0_' + str(config["replicates"])+  '_pt.s',shape=(336,336,1),type_im=np.dtype('int16'))
        self.sinogram_corrupt = np.resize(self.sinogram_corrupt,(344,252,1)) # to be removed

        self.net_outputs_path = self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + '_epoch=' + format(0) + '.img'
        self.checkpoint_simple_path = 'runs/' # To log loss in tensorboard thanks to Logger
        self.name_run = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.total_nb_iter = config["sub_iter_DIP"]

        ## Variables for WMV ##
        self.epochStar = -1
        self.windowSize = config["windowSize"]
        self.patienceNumber = config["patienceNumber"]
        self.SUCCESS = False
        self.VAR_recon = []
        
    def remove_cold_corrupted(self,config):
        addon = "remove_cold" # mu_DIP = 5
        addon = "remove_cold_already_in_corrupt" # mu_DIP = 5
        if (addon == "remove_cold"):
            self.sinogram_corrupt[35:59,35:59] = config["mu_DIP"]
            from pathlib import Path
            Path("/home/meraslia/workspace_reco/nested_admm/data/Algo/image40_0/replicate_1/" + str(config["mu_DIP"])).mkdir(parents=True, exist_ok=True)
            self.save_img(self.sinogram_corrupt,"/home/meraslia/workspace_reco/nested_admm/data/Algo/image40_0/replicate_1/" + str(config["mu_DIP"]) + "/corrupt.raw")
        
        # import matplotlib.pyplot as plt
        # plt.imshow(self.sinogram_corrupt,vmin=np.min(self.sinogram_corrupt),vmax=np.max(self.sinogram_corrupt),cmap='gray')
        # plt.show()
        
    def runComputation(self,config,root):
        # Initializing results class
        # if ((config["average_replicates"] and self.replicate == 1) or (config["average_replicates"] == False)):
        #     from iResults import iResults
        #     classResults = iResults(config)
        #     classResults.nb_replicates = self.nb_replicates
        #     classResults.debug = self.debug
        #     classResults.fixed_hyperparameters_list = self.fixed_hyperparameters_list
        #     classResults.hyperparameters_list = self.hyperparameters_list
        #     classResults.scanner = self.scanner
        #     if ("3D" not in self.phantom):
        #         classResults.bkg_ROI = self.bkg_ROI
        #         classResults.hot_TEP_ROI = self.hot_TEP_ROI
        #         if (self.phantom == "image50_1"):
        #             classResults.hot_TEP_ROI_ref = self.hot_TEP_ROI_ref
        #         classResults.hot_TEP_match_square_ROI = self.hot_TEP_match_square_ROI
        #         classResults.hot_perfect_match_ROI = self.hot_perfect_match_ROI
        #         classResults.hot_MR_recon = self.hot_MR_recon
        #         classResults.hot_ROI = self.hot_ROI
        #         classResults.cold_ROI = self.cold_ROI
        #         classResults.cold_inside_ROI = self.cold_inside_ROI
        #         classResults.cold_edge_ROI = self.cold_edge_ROI

        #     classResults.initializeSpecific(config,root)



        # Initialize variables
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
        # classResults.writeBeginningImages(self.suffix,self.image_net_input)
        # classResults.writeCorruptedImage(0,self.total_nb_iter,self.sinogram_corrupt,self.suffix,pet_algo="to fit",iteration_name="(post reconstruction)")
        # classResults.sinogram_corrupt = self.sinogram_corrupt
        # Before training, list all images already saved
        folder_sub_path = self.subroot + 'Block2/' + self.suffix + '/out_cnn/' + str(self.experiment)
        sorted_files = [filename*(self.has_numbers(filename)) for filename in os.listdir(folder_sub_path) if os.path.splitext(filename)[1] == '.img']
        # Train model using previously trained network (at iteration before)
        model = self.train_process(self.param1_scale_im_corrupt, self.param2_scale_im_corrupt, self.scaling_input, self.suffix,config, self.finetuning, self.processing_unit, self.total_nb_iter, self.method, self.global_it, self.image_net_input_torch, self.sinogram_corrupt_torch, self.net, self.PETImage_shape, self.experiment, self.checkpoint_simple_path, self.name_run, self.subroot, all_images_DIP = self.all_images_DIP,end_to_end=True)
        # model = self.train_process_end_to_end(self.param1_scale_im_corrupt, self.param2_scale_im_corrupt, self.scaling_input, self.suffix,config, self.finetuning, self.processing_unit, self.total_nb_iter, self.method, self.global_it, self.image_net_input_torch, self.sinogram_corrupt_torch, self.net, self.PETImage_shape, self.experiment, self.checkpoint_simple_path, self.name_run, self.subroot, all_images_DIP = self.all_images_DIP,end_to_end=True)
        ## Variables for WMV ##
        if (model.DIP_early_stopping):
            self.epochStar = model.classWMV.epochStar
            # if (config["EMV_or_WMV"] == "WMV"):
            #     classResults.windowSize = self.windowSize
            self.patienceNumber = model.classWMV.patienceNumber
            self.VAR_recon = model.classWMV.VAR_recon
            self.MSE_WMV = model.classWMV.MSE_WMV
            self.PSNR_WMV = model.classWMV.PSNR_WMV
            self.SSIM_WMV = model.classWMV.SSIM_WMV
            self.SUCCESS = model.classWMV.SUCCESS
            if (self.SUCCESS): # ES point is reached
                self.total_nb_iter = self.epochStar + self.patienceNumber + 1
                self.total_nb_iter = self.epochStar + 1

        # Saving variables
        if (self.net == 'DIP_VAE'):
            out, mu, logvar, z = model(self.image_net_input_torch)
        else:
            out = model(self.image_net_input_torch)

        last_iter = -1
        
        # Override total number of iterations if ES point found
        if (model.DIP_early_stopping):
            if (model.SUCCESS):
                self.total_nb_iter = model.epochStar + self.patienceNumber

        # Initialize WMV class
        model.initialize_WMV(config,self.fixed_hyperparameters_list,self.hyperparameters_list,self.debug,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input,self.suffix,self.global_it,root,self.scanner)

        # Iterations to be descaled
        if (self.all_images_DIP == "True"):
            epoch_values = np.arange(last_iter+1,self.total_nb_iter)
        elif (self.all_images_DIP == "False"):
            #epoch_values = np.arange(0,self.total_nb_iter,max(self.total_nb_iter//10,1))
            epoch_values = np.arange(last_iter+self.total_nb_iter//10,self.total_nb_iter+self.total_nb_iter//10,max((self.total_nb_iter-last_iter+1)//10,1)) - 1
        elif (self.all_images_DIP == "Last"):
            epoch_values = np.array([self.total_nb_iter-1])


        # Write descaled images in files
        for epoch in epoch_values:
            if (self.all_images_DIP == "Last"):
                net_outputs_path = self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + "/ES_out_" + self.net + format(self.global_it) + '_epoch=' + format(epoch) + '.img'
            else:
                net_outputs_path = self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + format(self.global_it) + '_epoch=' + format(epoch) + '.img'
            
            out = self.fijii_np(net_outputs_path,shape=(self.PETImage_shape),type_im='<f')



            # WMV
            # self.log("SUCCESS", int(model.classWMV.SUCCESS))
            if (model.DIP_early_stopping):
                model.classWMV.SUCCESS,model.classWMV.VAR_min,model.classWMV.stagnate = model.classWMV.WMV(np.copy(out),epoch,model.sub_iter_DIP,model.classWMV.queueQ,model.classWMV.SUCCESS,model.classWMV.VAR_min,model.classWMV.stagnate)
                self.VAR_recon = model.classWMV.VAR_recon
                self.MSE_WMV = model.classWMV.MSE_WMV
                self.PSNR_WMV = model.classWMV.PSNR_WMV
                self.SSIM_WMV = model.classWMV.SSIM_WMV
                self.epochStar = model.classWMV.epochStar
                '''
                if self.EMV_or_WMV == "EMV":
                    self.alpha_EMV = model.classWMV.alpha_EMV
                else:
                    self.windowSize = model.classWMV.windowSize
                '''
                self.patienceNumber = model.classWMV.patienceNumber
                self.SUCCESS = model.classWMV.SUCCESS
                print(self.VAR_recon)
                if self.SUCCESS:
                    print("SUCCESS WMVVVVVVVVVVVVVVVVVV")

            out_descale = out



            out_torch = torch.from_numpy(out)
            # Descale like at the beginning
            out_descale = self.descale_imag(out,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)
            #'''
            # Saving image output
            net_outputs_path = self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + format(self.global_it) + '_epoch=' + format(epoch) + '.img'
            os.system("mv " + "'" + net_outputs_path + "' '" + self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + format(self.global_it) + '_epoch=' + format(epoch)  + 'scaled.img' + "'")
            self.save_img(out_descale, net_outputs_path)
            # Squeeze image by loading it
            out_descale = self.fijii_np(net_outputs_path,shape=(self.PETImage_shape),type_im='<f') # loading DIP output
            # Saving (now DESCALED) image output
            self.save_img(out_descale, net_outputs_path)

            # if ("3D" not in self.phantom):
            #     if ("post_reco" not in config["task"]):
            #         # Compute IR metric (different from others with several replicates)
            #         classResults.compute_IR_bkg(self.PETImage_shape,out_descale,epoch,classResults.IR_bkg_recon,self.phantom)
            #         classResults.writer.add_scalar('Image roughness in the background (best : 0)', classResults.IR_bkg_recon[epoch], epoch+1)
            #         # Compute IR in whole phantom (different from others with several replicates)
            #         classResults.compute_IR_whole(self.PETImage_shape,out_descale,self.global_it,classResults.IR_whole_recon,self.phantom)
            #         classResults.writer.add_scalar('Image roughness in the phantom', classResults.IR_whole_recon[self.global_it], self.global_it+1)
            #     # Write images over epochs
            # classResults.writeEndImagesAndMetrics(epoch,self.total_nb_iter,self.PETImage_shape,out_descale,self.suffix,self.phantom,self.net,pet_algo="to fit",iteration_name="(post reconstruction)")

            if (config["DIP_early_stopping"]):
                if (model.classWMV.SUCCESS):
                    break

        # if (model.DIP_early_stopping):
        #     classResults.epochStar = self.epochStar
        #     classResults.VAR_recon = self.VAR_recon
        #     classResults.MSE_WMV = self.MSE_WMV
        #     classResults.PSNR_WMV = self.PSNR_WMV
        #     classResults.SSIM_WMV = self.SSIM_WMV
        #     classResults.patienceNumber = self.patienceNumber
        #     classResults.SUCCESS = self.SUCCESS