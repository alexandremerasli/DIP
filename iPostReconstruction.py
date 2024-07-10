## Python libraries

# Useful
from datetime import datetime
import numpy as np
import torch
import os

# Local files to import
from vDenoising import vDenoising

class iPostReconstruction(vDenoising):
    def __init__(self,config, *args, **kwargs):
        self.finetuning = 'False' # to ignore last.ckpt file
        self.outer_it = -100 # Set it to -100, to ignore last.ckpt file

    def initializeSpecific(self,config,root, *args, **kwargs):
        print("Denoising in post reconstruction")
        # Delete previous ckpt files from previous runs
        # if (self.finetuning == "ES"):
        os.system("rm -rf " + self.subroot_phantom+'Block2/' + self.suffix + '/checkpoint/'+format(self.experiment) + "*")

        self.override_input = False
        self.sub_iter_DIP_already_done = 0
        vDenoising.initializeSpecific(self,config,root)
        # Loading DIP x_label (corrupted image) from block1
        
        self.image_corrupt = self.fijii_np("/home/MEDECINE/mera1140/sherbrooke_workspace/wakusuteshon/24_06_07_iecFirstTestAlgoOnUHR/Datasets/MLEM_UHR_sens_scaled/MLEM_UHR_sens_scaled_it20_192.img",shape=(self.PETImage_shape)) # 
        
        
        # modify input with line on the edge of the phantom (DIP input tests)
        # self.remove_cold_corrupted(config)
        
        self.net_outputs_path = self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + '_epoch=' + format(0) + '.img'
        self.checkpoint_simple_path = 'runs/' # To log loss in tensorboard thanks to Logger
        self.name_run = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.total_nb_iter = config["sub_iter_DIP"]
        
    def runComputation(self,config,root):
        # Initializing results class
        if ((self.average_replicates and self.replicate == 1) or (self.average_replicates == False)):
            from iResults import iResults
            classResults = iResults(config)
            self.assignVariablesFromResults(classResults)
            self.assignROI(classResults)
            classResults.initializeSpecific(config,root)



        # Initialize variables
        # Scaling of x_label image
        if ("3D" in self.phantom):
            #self.image_corrupt = self.image_corrupt.reshape(self.image_corrupt.shape[::-1])
            #self.image_corrupt = np.transpose(self.image_corrupt,axes=(1,2,0)) # imshow ok
            #self.image_corrupt = np.transpose(self.image_corrupt,axes=(1,0,2)) #bug
            #self.image_corrupt = np.transpose(self.image_corrupt,axes=(0,1,2)) #nope
            #self.image_corrupt = np.transpose(self.image_corrupt,axes=(0,2,1)) #bug
            #self.image_corrupt = np.transpose(self.image_corrupt,axes=(2,0,1)) #nope
            #self.image_corrupt = np.transpose(self.image_corrupt,axes=(2,1,0)) #nope
            #self.image_corrupt = self.image_corrupt.reshape(self.image_corrupt.shape[::-1])
            
            print("ok")


        image_corrupt_input_scale,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt = self.rescale_imag(self.image_corrupt,self.scaling_input) # Scaling of x_label image


        # '''
        # import matplotlib.pyplot as plt
        # plt.imshow(image_corrupt_input_scale[30,:,:],cmap='gray')
        # plt.colorbar()
        # plt.show()
        # '''

        # Corrupted image x_label, numpy --> torch float32
        self.image_corrupt_torch = torch.Tensor(self.several_DIP_inputs*[image_corrupt_input_scale])
        # Adding dimensions to fit network architecture
        if (self.nb_dimensions == 2): # if 3D but with dim3 = 1 -> 2D
            self.image_corrupt_torch = self.image_corrupt_torch.view(self.several_DIP_inputs,1,self.PETImage_shape[0],self.PETImage_shape[1],self.PETImage_shape[2])
            self.image_corrupt_torch = self.image_corrupt_torch[:,:,:,:,0]
        else: #3D
            self.image_corrupt_torch = self.image_corrupt_torch.view(1,1,self.PETImage_shape[2],self.PETImage_shape[1],self.PETImage_shape[0])
        classResults.writeBeginningImages(self.suffix,self.image_net_input)
        classResults.writeCorruptedImage(0,self.total_nb_iter,self.image_corrupt,self.suffix,pet_algo="to fit",iteration_name="(post reconstruction)")
        classResults.image_corrupt = self.image_corrupt
        # Before training, list all images already saved
        folder_sub_path = self.subroot_phantom + 'Block2/' + self.suffix + '/out_cnn/' + str(self.experiment)
        sorted_files = [filename*(self.has_numbers(filename)) for filename in os.listdir(folder_sub_path) if os.path.splitext(filename)[1] == '.img']
        # Train model using previously trained network (at iteration before)
        model = self.train_process(self.param1_scale_im_corrupt, self.param2_scale_im_corrupt, self.scaling_input, self.suffix,config, self.finetuning, self.processing_unit, self.total_nb_iter, self.method, self.outer_it, self.image_net_input_torch, self.image_corrupt_torch, self.net, self.PETImage_shape, self.experiment, self.checkpoint_simple_path, self.name_run, self.subroot_phantom, all_images_DIP = self.all_images_DIP)

        ## Variables for MV ##
        if (model.DIP_early_stopping):
            self.epochStar = model.classMV.epochStar
            # if (config["EMV_or_WMV"] == "WMV"):
            #     classResults.windowSize = self.windowSize
            self.patienceNumber = model.classMV.patienceNumber
            self.VAR_recon = model.classMV.VAR_recon
            self.MSE_MV = model.classMV.MSE_MV
            self.PSNR_MV = model.classMV.PSNR_MV
            self.SSIM_MV = model.classMV.SSIM_MV
            self.SUCCESS = model.classMV.SUCCESS
            if (self.SUCCESS): # ES point is reached
                self.total_nb_iter = self.epochStar + self.patienceNumber + 1
                self.total_nb_iter = self.epochStar + 1

        # Saving variables
        if (self.net == 'DIP_VAE'):
            out, mu, logvar, z = model(self.image_net_input_torch)
        else:
            out = model(self.image_net_input_torch)

        # Check if previous computation was already done to only scale last computed images
        # if len(sorted_files) > 0:
        #     initial_image_not_used, it_not_used, last_iter = self.ImageAndItToResumeComputation(sorted_files,"",folder_sub_path)
        # else:
        #     last_iter = -1

        # if (last_iter > 0):
        #     nb_iter_train = self.total_nb_iter - (last_iter + 1)
        # else:
        #     nb_iter_train = self.total_nb_iter
        last_iter = -1
        
        # Override total number of iterations if ES point found
        if (model.DIP_early_stopping):
            if (model.SUCCESS):
                self.total_nb_iter = model.epochStar + self.patienceNumber

        # Initialize WMV class
        model.initialize_MV(config,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input,self.suffix,self.outer_it,self.sub_iter_DIP,self.root, self.subroot,self.scanner, self.simulation, self.hyperparameters_list)

        # Iterations to be descaled
        if (self.all_images_DIP == "True"):
            epoch_values = np.arange(last_iter+1,self.total_nb_iter)
        elif (self.all_images_DIP == "False"):
            #epoch_values = np.arange(0,self.total_nb_iter,max(self.total_nb_iter//10,1))
            epoch_values = np.arange(last_iter+self.total_nb_iter//10,self.total_nb_iter+self.total_nb_iter//10,max((self.total_nb_iter-last_iter+1)//10,1)) - 1
        elif (self.all_images_DIP == "Unique"):
            epoch_values = np.array([self.total_nb_iter-1])


        # Write descaled images in files
        for epoch in epoch_values:
            if (self.all_images_DIP == "Unique"):
                net_outputs_path = self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + "/ES_out_" + self.net + format(self.outer_it) + '_epoch=' + format(epoch) + '.img'
            else:
                net_outputs_path = self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + format(self.outer_it) + '_epoch=' + format(epoch) + '.img'
            
            out = self.fijii_np(net_outputs_path,shape=(self.PETImage_shape),type_im='<f')



            # MV
            # self.log("SUCCESS", int(model.classMV.SUCCESS))
            if (model.DIP_early_stopping):
                model.classMV.SUCCESS,model.classMV.VAR_min,model.classMV.stagnate = model.classMV.compute_MV_value(np.copy(out),epoch,model.sub_iter_DIP,model.classMV.queueQ,model.classMV.SUCCESS,model.classMV.VAR_min,model.classMV.stagnate)
                self.VAR_recon = model.classMV.VAR_recon
                self.MSE_MV = model.classMV.MSE_MV
                self.PSNR_MV = model.classMV.PSNR_MV
                self.SSIM_MV = model.classMV.SSIM_MV
                self.epochStar = model.classMV.epochStar
                '''
                if self.EMV_or_WMV == "EMV":
                    self.alpha_EMV = model.classMV.alpha_EMV
                else:
                    self.windowSize = model.classMV.windowSize
                '''
                self.patienceNumber = model.classMV.patienceNumber
                self.SUCCESS = model.classMV.SUCCESS
                print(self.VAR_recon)
                if self.SUCCESS:
                    print("SUCCESS MVVVVVVVVVVVVVVVVVV")

            out_descale = out



            out_torch = torch.from_numpy(out)
            # Descale like at the beginning
            out_descale = self.descale_imag(out,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)
            #'''
            # Saving image output
            net_outputs_path = self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + format(self.outer_it) + '_epoch=' + format(epoch) + '.img'
            os.system("mv " + "'" + net_outputs_path + "' '" + self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + format(self.outer_it) + '_epoch=' + format(epoch)  + 'scaled.img' + "'")
            self.save_img(out_descale, net_outputs_path)
            # Squeeze image by loading it
            out_descale = self.fijii_np(net_outputs_path,shape=(self.PETImage_shape),type_im='<f') # loading DIP output
            # Saving (now DESCALED) image output
            self.save_img(out_descale, net_outputs_path)

            if (self.simulation):
                if ("post_reco" not in config["task"]):
                    # Compute IR metric (different from others with several replicates)
                    classResults.compute_IR_bkg(self.PETImage_shape,out_descale,epoch,classResults.IR_bkg_recon,self.phantom)
                    classResults.writer.add_scalar('Image roughness in the background (best : 0)', classResults.IR_bkg_recon[epoch], epoch+1)
                    # Compute IR in whole phantom (different from others with several replicates)
                    classResults.compute_IR_whole(self.PETImage_shape,out_descale,self.outer_it,classResults.IR_whole_recon,self.phantom)
                    classResults.writer.add_scalar('Image roughness in the phantom', classResults.IR_whole_recon[self.outer_it], self.outer_it+1)
                # Write images over epochs
            classResults.writeEndImagesAndMetrics(epoch,self.total_nb_iter,self.PETImage_shape,out_descale,self.suffix,self.phantom,self.net,pet_algo="to fit",iteration_name="(post reconstruction)")
            #classResults.writeEndImagesAndMetrics(epoch,self.total_nb_iter,self.PETImage_shape,out,self.suffix,self.phantom,self.net,pet_algo="to fit",iteration_name="(post reconstruction)")

            if (self.DIP_early_stopping):
                if (model.classMV.SUCCESS):
                    break

        if (model.DIP_early_stopping):
            classResults.epochStar = self.epochStar
            classResults.VAR_recon = self.VAR_recon
            classResults.MSE_MV = self.MSE_MV
            classResults.PSNR_MV = self.PSNR_MV
            classResults.SSIM_MV = self.SSIM_MV
            # if (config["EMV_or_WMV"] == "WMV"):
            #     classResults.windowSize = self.windowSize
            classResults.patienceNumber = self.patienceNumber
            classResults.SUCCESS = self.SUCCESS
            # if (config["EMV_or_WMV"] == "WMV"):
            #     classResults.MV_plot(config)