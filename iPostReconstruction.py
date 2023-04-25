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
        self.global_it = -100 # Set it to -100, to ignore last.ckpt file

    def initializeSpecific(self,config,root, *args, **kwargs):
        print("Denoising in post reconstruction")
        # Delete previous ckpt files from previous runs
        if (config["finetuning"] == "ES"):
            os.system("rm -rf " + self.subroot+'Block2/' + self.suffix + '/checkpoint/'+format(self.experiment) + "*")

        self.override_input = False
        vDenoising.initializeSpecific(self,config,root)
        # Loading DIP x_label (corrupted image) from block1
        #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/' + 'im_corrupt_beginning.img',shape=(self.PETImage_shape),type_im='<d') # ADMMLim for nested
        #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'ADMMLim_it10000.img',shape=(self.PETImage_shape),type_im='<d') # ADMMLim for nested
        #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'ADMMLim_blurred_it10000.img',shape=(self.PETImage_shape),type_im='<d') # ADMMLim for nested
        #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'ADMMLim_it100.img',shape=(self.PETImage_shape),type_im='<d') # ADMMLim for nested
        #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/' + 'initialization/MLEM_it60_REF_cropped.img',shape=(self.PETImage_shape),type_im='<f') # MLEM for Gong
        #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/' + 'initialization/MLEM_it60.img',shape=(self.PETImage_shape),type_im='<d') # MLEM for Gong
        #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'MLEM_60it/replicate_' + str(self.replicate) + '/MLEM_it60.img',shape=(self.PETImage_shape),type_im='<d')
        #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/initialization/' + self.phantom + '/BSREM_30it' + '/replicate_' + str(self.replicate) + '/BSREM_it30.img',shape=(self.PETImage_shape),type_im='<d')
        #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'snail_noisy.img',shape=((256,384)),type_im='<f')
        #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'snail_noisy.img',shape=(self.PETImage_shape),type_im='<f')
        
        
        #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'F16_GT_' + str(self.PETImage_shape[0]) + '.img',shape=(self.PETImage_shape),type_im='<f')
        self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/initialization/' + self.phantom + '/BSREM_30it' + '/replicate_' + str(self.replicate) + '/BSREM_it30.img',shape=(self.PETImage_shape),type_im='<d')
        #self.image_corrupt = self.fijii_np("/home/meraslia/workspace_reco/nested_admm/data/Algo/image4_0/replicate_10/nested/Block2/config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=14_tau_D=2_lr=0.01_sub_i=100_opti_=Adam_skip_=3_scali=standardization_input=random_nb_ou=1_mlem_=False_A_AML=-100/x_label/24/" + "-1_x_labelconfig_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=14_tau_D=2_lr=0.01_sub_i=100_opti_=Adam_skip_=3_scali=standardization_input=random_nb_ou=1_mlem_=False_A_AML=-100.img",shape=(self.PETImage_shape))
        
        #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'random_1.img',shape=(self.PETImage_shape),type_im='<d')
        #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'MLEM_3D_it2.img',shape=(self.PETImage_shape),type_im='<f')
        #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'BSREM_3D_it30.img',shape=(self.PETImage_shape),type_im='<d')
        #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.raw',shape=(self.PETImage_shape),type_im='<f')
        #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/' + 'im_corrupt_beginning_it100.img',shape=(self.PETImage_shape),type_im='<d') # ADMMLim for nested
        #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'OPTITR_2it.img',shape=(self.PETImage_shape),type_im='<d') # ADMMLim for nested
        #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/database_v2/' + 'image2_3D/image2_3D.img',shape=(self.PETImage_shape),type_im='<f') # ADMMLim for nested
        #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/database_v2/' + 'image0/image0.img',shape=(self.PETImage_shape),type_im='<f') # ADMMLim for nested
        self.net_outputs_path = self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + '_epoch=' + format(0) + '.img'
        self.checkpoint_simple_path = 'runs/' # To log loss in tensorboard thanks to Logger
        self.name_run = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.total_nb_iter = config["sub_iter_DIP"]

        '''
        ## Variables for WMV ##
        self.epochStar = -1
        self.windowSize = config["windowSize"]
        self.patienceNumber = config["patienceNumber"]
        self.SUCCESS = False
        self.VAR_recon = []
        '''

    def runComputation(self,config,root):
        # Initializing results class
        if ((config["average_replicates"] and self.replicate == 1) or (config["average_replicates"] == False)):
            from iResults import iResults
            classResults = iResults(config)
            classResults.nb_replicates = self.nb_replicates
            classResults.debug = self.debug
            classResults.hyperparameters_list = self.hyperparameters_list
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
            
            self.save_img(self.image_corrupt,"/disk/workspace_reco/nested_admm/data/Algo/image2_3D/replicate_1/nested/Block2/post_reco config_image=BSREM_3D_it30_rho=0.003_adapt=nothing_mu_DI=10_tau_D=2_lr=0.0001_sub_i=10_opti_=Adam_skip_=3_scali=standardization_input=random_nb_ou=3_alpha=1_adapt=tau_mu_ad=2_tau=100_mlem_=False/out_cnn/24/corrupt.raw")

            print("ok")


        image_corrupt_input_scale,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt = self.rescale_imag(self.image_corrupt,self.scaling_input) # Scaling of x_label image


        '''
        import matplotlib.pyplot as plt
        plt.imshow(self.image_corrupt[30,:,:],cmap='gray')
        plt.colorbar()
        plt.show()
        '''

        # Corrupted image x_label, numpy --> torch float32
        self.image_corrupt_torch = torch.Tensor(image_corrupt_input_scale)
        # Adding dimensions to fit network architecture
        if (self.PETImage_shape[2] == 1): # if 3D but with dim3 = 1 -> 2D
            self.image_corrupt_torch = self.image_corrupt_torch.view(1,1,self.PETImage_shape[0],self.PETImage_shape[1],self.PETImage_shape[2])
            self.image_corrupt_torch = self.image_corrupt_torch[:,:,:,:,0]
        else: #3D
            self.image_corrupt_torch = self.image_corrupt_torch.view(1,1,self.PETImage_shape[2],self.PETImage_shape[1],self.PETImage_shape[0])
        classResults.writeBeginningImages(self.suffix,self.image_net_input)
        classResults.writeCorruptedImage(0,self.total_nb_iter,self.image_corrupt,self.suffix,pet_algo="to fit",iteration_name="(post reconstruction)")
        classResults.image_corrupt = self.image_corrupt
        # Train model using previously trained network (at iteration before)
       
       
        folder_sub_path = self.subroot + 'Block2/' + self.suffix + '/out_cnn/' + str(self.experiment)
        sorted_files = [filename*(self.has_numbers(filename)) for filename in os.listdir(folder_sub_path) if os.path.splitext(filename)[1] == '.img']
        
        # Check if previous computation was already done
        if len(sorted_files) > 0:
            initialimage_not_used, it_not_used, last_iter = self.ImageAndItToResumeComputation(sorted_files,"",folder_sub_path)
        else:
            last_iter = -1

        if (last_iter > 0):
            nb_iter_train = self.total_nb_iter - (last_iter + 1)
        else:
            nb_iter_train = self.total_nb_iter
        
        if (nb_iter_train > 0):
            model = self.train_process(self.param1_scale_im_corrupt, self.param2_scale_im_corrupt, self.scaling_input, self.suffix,config, self.finetuning, self.processing_unit, nb_iter_train, self.method, self.global_it, self.image_net_input_torch, self.image_corrupt_torch, self.net, self.PETImage_shape, self.experiment, self.checkpoint_simple_path, self.name_run, self.subroot, all_images_DIP = self.all_images_DIP, last_iter=last_iter)
        else:
            raise ValueError("need to select a higher number of iterations, because the " + str(self.total_nb_iter) + " first iterations were already computed")


        ## Variables for WMV ##
        if (model.DIP_early_stopping):
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
                self.total_nb_iter = self.epochStar + 1

        # Saving variables
        if (self.net == 'DIP_VAE'):
            out, mu, logvar, z = model(self.image_net_input_torch)
        else:
            out = model(self.image_net_input_torch)

        # Write descaled images in files
        if (self.all_images_DIP == "True"):
            epoch_values = np.arange(last_iter+1,self.total_nb_iter)
        elif (self.all_images_DIP == "False"):
            #epoch_values = np.arange(0,self.total_nb_iter,max(self.total_nb_iter//10,1))
            epoch_values = np.arange(last_iter+self.total_nb_iter//10,self.total_nb_iter+self.total_nb_iter//10,max((self.total_nb_iter-last_iter+1)//10,1)) - 1
        elif (self.all_images_DIP == "Last"):
            epoch_values = np.array([self.total_nb_iter-1])

        ## Variables for WMV ##
        queueQ = []
        VAR_min = np.inf
        #model.SUCCESS = False
        stagnate = 0

        for epoch in epoch_values:
            if (self.all_images_DIP == "Last"):
                net_outputs_path = self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + "/ES_out_" + self.net + format(self.global_it) + '_epoch=' + format(epoch) + '.img'
            else:
                net_outputs_path = self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + format(self.global_it) + '_epoch=' + format(epoch) + '.img'
            
            out = self.fijii_np(net_outputs_path,shape=(self.PETImage_shape),type_im='<f')
            #out_torch = torch.from_numpy(out)
            # Descale like at the beginning
            out_descale = self.descale_imag(out,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)
            #'''
            # Saving image output
            net_outputs_path = self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + format(self.global_it) + '_epoch=' + format(epoch) + '.img'
            self.save_img(out_descale, net_outputs_path)
            # Squeeze image by loading it
            out_descale = self.fijii_np(net_outputs_path,shape=(self.PETImage_shape),type_im='<f') # loading DIP output
            # Saving (now DESCALED) image output
            self.save_img(out_descale, net_outputs_path)

            if ("3D" not in self.phantom):
                # Compute IR metric (different from others with several replicates)
                classResults.compute_IR_bkg(self.PETImage_shape,out_descale,epoch,classResults.IR_bkg_recon,self.phantom)
                classResults.writer.add_scalar('Image roughness in the background (best : 0)', classResults.IR_bkg_recon[epoch], epoch+1)
                # Compute IR in whole phantom (different from others with several replicates)
                classResults.compute_IR_whole(self.PETImage_shape,out_descale,self.global_it,classResults.IR_whole_recon,self.phantom)
                classResults.writer.add_scalar('Image roughness in the phantom', classResults.IR_whole_recon[self.global_it], self.global_it+1)
                # Write images over epochs
            classResults.writeEndImagesAndMetrics(epoch,self.total_nb_iter,self.PETImage_shape,out_descale,self.suffix,self.phantom,self.net,pet_algo="to fit",iteration_name="(post reconstruction)")
            #classResults.writeEndImagesAndMetrics(epoch,self.total_nb_iter,self.PETImage_shape,out,self.suffix,self.phantom,self.net,pet_algo="to fit",iteration_name="(post reconstruction)")

        if (model.DIP_early_stopping):
            classResults.epochStar = self.epochStar
            classResults.VAR_recon = self.VAR_recon
            classResults.MSE_WMV = self.MSE_WMV
            classResults.PSNR_WMV = self.PSNR_WMV
            classResults.SSIM_WMV = self.SSIM_WMV
            classResults.windowSize = self.windowSize
            classResults.patienceNumber = self.patienceNumber
            classResults.SUCCESS = self.SUCCESS

            classResults.WMV_plot(config)