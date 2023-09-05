## Python libraries

# Pytorch
from torch.utils.tensorboard import SummaryWriter

# Math
import numpy as np
import matplotlib.pyplot as plt

# Useful
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from os.path import isfile
from csv import writer as writer_csv
from csv import reader as reader_csv

# Local files to import
#from vGeneral import vGeneral
from vDenoising import vDenoising
from iWMV import iWMV

class iResults(vDenoising):
    def __init__(self,config, *args, **kwargs):
        print("__init__")

    def initializeSpecific(self,config,root, *args, **kwargs):
        # Initialize general variables
        self.initializeGeneralVariables(config,root)
        self.config = config
        # Specific hyperparameters for reconstruction module (Do it here to have raytune config hyperparameters selection)
        if (config["net"] == "DD" or config["net"] == "DD_AE"):
            self.d_DD = config["d_DD"]
            self.k_DD = config["k_DD"]
        # if ("nested" in config["method"] or "Gong" in config["method"]):
        #     self.DIP_early_stopping = config["DIP_early_stopping"]
        #vDenoising.initializeSpecific(self,config,root)
        # Initialize early stopping method if asked for
        if ("nested" in config["method"] or "Gong" in config["method"]):
            self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/initialization/' + self.phantom + '/BSREM_30it' + '/replicate_' + str(self.replicate) + '/BSREM_it30.img',shape=(self.PETImage_shape),type_im='<d')
            image_corrupt_input_scale,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt = self.rescale_imag(self.image_corrupt,config["scaling"]) # Scaling of x_label image
            if ("post_reco_in_suffix" in config):
                if (config["post_reco_in_suffix"]):
                    self.global_it = -100
                else:
                    self.global_it = -1
            else:
                self.global_it = -100
            if (config["DIP_early_stopping"] and "show_results_post_reco" in config["task"]):
                self.initialize_WMV(config,self.fixed_hyperparameters_list,self.hyperparameters_list,self.debug,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,config["scaling"],self.suffix,self.global_it,root,self.scanner)
                self.lr = config['lr']

        if ('ADMMLim' in config["method"]):
            self.i_init = 30 # Remove first iterations
            self.i_init = 1 # Remove first iterations
        else:
            self.i_init = 1

        self.defineTotalNbIter_beta_rho(config["method"], config, config["task"],stopping_criterion=False) # Compute metrics for every iterations, stopping_criterion will be used in final curves


        # Create summary writer from tensorboard
        self.tensorboard = config["tensorboard"]
        self.writer = SummaryWriter()
        
        #Loading Ground Truth image to compute metrics
        self.image_gt = self.fijii_np(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.raw',shape=(self.PETImage_shape),type_im='<f')
        if config["FLTNB"] == "double":
            self.image_gt = self.image_gt.astype(np.float64)

        # # Loading attenuation map
        # image_atn = self.fijii_np(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '_atn.raw',shape=(self.PETImage_shape),type_im='<f')
        # self.write_image_tensorboard(self.writer,image_atn,"Attenuation map (FULL CONTRAST)",self.suffix,self.image_gt,0,full_contrast=True) # Attenuation map in tensorboard
    
        # # Loading MR-like image
        # image_mr = self.fijii_np(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '_mr.raw',shape=(self.PETImage_shape),type_im='<f')
        # self.write_image_tensorboard(self.writer,image_atn,"Attenuation map (FULL CONTRAST)",self.suffix,self.image_gt,0,full_contrast=True) # Attenuation map in tensorboard

        '''
        image = self.image_gt
        image = image[20,:,:]
        plt.imshow(image, cmap='gray_r',vmin=0,vmax=np.max(image)) # Showing all images with same contrast
        plt.colorbar()
        #os.system('rm -rf' + self.subroot + 'Images/tmp/' + suffix + '/*')
        plt.savefig(self.subroot_data + 'Data/database_v2/' + 'image_gt.png')
        '''

        # Defining ROIs
        if (not hasattr(self,"phantom_ROI")):
            if ("3D" not in self.phantom):
                self.phantom_ROI = self.get_phantom_ROI(self.phantom)
                # self.bkg_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "background_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
                if (self.phantom == "image4_0" or self.phantom == "image400_0" or self.phantom == "image40_0" or self.phantom == "image40_1"):
                    self.hot_TEP_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "tumor_TEP_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
                    self.hot_TEP_match_square_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "tumor_TEP_match_square_ROI_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
                    self.hot_perfect_match_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "tumor_perfect_match_ROI_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
                    # This ROIs has already been defined, but is computed for the sake of simplicity
                    self.hot_ROI = self.hot_TEP_ROI
                else:
                    self.hot_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "tumor_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
                    # These ROIs do not exist, so put them equal to hot ROI for the sake of simplicity
                    self.hot_TEP_ROI = np.array(self.hot_ROI)
                    self.hot_TEP_match_square_ROI = np.array(self.hot_ROI)
                    self.hot_perfect_match_ROI = np.array(self.hot_ROI)
                self.cold_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "cold_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
                self.cold_inside_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "cold_inside_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
                self.cold_edge_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "cold_edge_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
        
        if ("3D" not in self.phantom):
            # Metrics arrays
            self.PSNR_recon = np.zeros(int(self.total_nb_iter ) + 1)
            self.PSNR_norm_recon = np.zeros(int(self.total_nb_iter ) + 1)
            self.MSE_recon = np.zeros(int(self.total_nb_iter ) + 1)
            self.SSIM_recon = np.zeros(int(self.total_nb_iter ) + 1)
            self.MA_cold_recon = np.zeros(int(self.total_nb_iter ) + 1)
            self.AR_hot_recon = np.zeros(int(self.total_nb_iter ) + 1)
            
            self.AR_hot_TEP_recon = np.zeros(int(self.total_nb_iter ) + 1)
            self.AR_hot_TEP_match_square_recon = np.zeros(int(self.total_nb_iter ) + 1)
            self.AR_hot_perfect_match_recon = np.zeros(int(self.total_nb_iter ) + 1)
            
            self.loss_DIP_recon = np.zeros(int(self.total_nb_iter ) + 1)
            self.CRC_hot_recon = np.zeros(int(self.total_nb_iter ) + 1)
            self.AR_bkg_recon = np.zeros(int(self.total_nb_iter ) + 1)
            self.IR_bkg_recon = np.zeros(int(self.total_nb_iter ) + 1)
            self.IR_whole_recon = np.empty(int(self.total_nb_iter ) + 1)
            self.IR_whole_recon[:] = 0 #np.nan

            self.mean_inside_recon = np.zeros(int(self.total_nb_iter ) + 1)

            self.likelihoods = [] # Will be appended

            self.MA_cold_inside = np.zeros(int(self.total_nb_iter ) + 1)
            self.MA_cold_edge = np.zeros(int(self.total_nb_iter ) + 1)

        if ( 'nested' in self.method or  'Gong' in self.method):
            #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'MLEM_60it/replicate_' + str(self.replicate) + '/MLEM_it60.img',shape=(self.PETImage_shape),type_im='<d')
            #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'random_1.img',shape=(self.PETImage_shape),type_im='<d')
            #self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/initialization/' + 'F16_GT_' + str(self.PETImage_shape[0]) + '.img',shape=(self.PETImage_shape),type_im='<f')
            if ("3_" not in self.phantom):
                try:
                    self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/initialization/' + self.phantom + '/BSREM_30it' + '/replicate_' + str(self.replicate) + '/BSREM_it30.img',shape=(self.PETImage_shape),type_im='<f')
                except:
                    self.image_corrupt = self.fijii_np(self.subroot_data + 'Data/initialization/' + self.phantom + '/BSREM_30it' + '/replicate_' + str(self.replicate) + '/BSREM_it30.img',shape=(self.PETImage_shape),type_im='<d')
            else:
                self.image_corrupt = self.fijii_np(self.subroot_data + "/Data/database_v2/" + self.phantom + '/' + self.phantom + '.img',shape=(self.PETImage_shape),type_im='<d')
            
            #self.image_corrupt = self.fijii_np("/home/meraslia/workspace_reco/nested_admm/data/Algo/image4_0/replicate_10/nested/Block2/config_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=14_tau_D=2_lr=0.01_sub_i=100_opti_=Adam_skip_=3_scali=standardization_input=random_nb_ou=1_mlem_=False_A_AML=-100/x_label/24/" + "-1_x_labelconfig_image=BSREM_it30_rho=0.003_adapt=nothing_mu_DI=14_tau_D=2_lr=0.01_sub_i=100_opti_=Adam_skip_=3_scali=standardization_input=random_nb_ou=1_mlem_=False_A_AML=-100.img",shape=(self.PETImage_shape))

        # self.image_gt = self.image_gt / np.max(self.image_gt) * 255
        # self.image_gt = self.image_gt.astype(np.int8)
        # if (hasattr(self,'image_corrupt')):
        #     self.image_corrupt = self.image_corrupt / np.max(self.image_corrupt) * 255
        #     self.image_corrupt = self.image_corrupt.astype(np.int8)
        
        # PSNR_corrupt = peak_signal_noise_ratio(self.image_gt, self.image_corrupt, data_range=np.amax(self.image_corrupt) - np.amin(self.image_corrupt)) # PSNR with true values
        # SSIM_corrupt = structural_similarity(np.squeeze(self.image_gt), np.squeeze(self.image_corrupt), data_range=(self.image_corrupt).max() - (self.image_corrupt).min())
        # from utils.mssim import ssimc
        # SSIM_corrupt = ssimc(np.squeeze(self.image_gt), np.squeeze(self.image_corrupt),1, 1, 1)
        if ('Gong' in config["method"] or 'nested' in config["method"]):
            self.i_init = 0

    def writeBeginningImages(self,suffix,image_net_input=None):
        if (self.tensorboard):
            self.write_image_tensorboard(self.writer,self.image_gt,"Ground Truth (emission map)",suffix,self.image_gt,0,full_contrast=True) # Ground truth in tensorboard
            if (image_net_input is not None):
                # image_net_input_reversed = np.max(image_net_input) - image_net_input
                image_net_input_reversed = image_net_input # only colorbar will be reversed in write_image_tensorboard()
                self.write_image_tensorboard(self.writer,image_net_input_reversed,"DIP input (FULL CONTRAST)",suffix,image_net_input_reversed,0,full_contrast=True) # DIP input in tensorboard

    def writeCorruptedImage(self,i,max_iter,x_label,suffix,pet_algo,iteration_name='iterations'):
        if (self.tensorboard):
            if (self.all_images_DIP == "Last"):
                self.write_image_tensorboard(self.writer,x_label,"Corrupted image (x_label) over " + pet_algo + " " + iteration_name,suffix,self.image_gt,i) # Showing all corrupted images with same contrast to compare them together
                self.write_image_tensorboard(self.writer,x_label,"Corrupted image (x_label) over " + pet_algo + " " + iteration_name + " (FULL CONTRAST)",suffix,self.image_gt,i,full_contrast=True) # Showing each corrupted image with contrast = 1
            else:       
                if (((max_iter>=10) and (i%(max_iter // 10) == 0)) or (max_iter<10)):
                    self.write_image_tensorboard(self.writer,x_label,"Corrupted image (x_label) over " + pet_algo + " " + iteration_name,suffix,self.image_gt,i) # Showing all corrupted images with same contrast to compare them together
                    self.write_image_tensorboard(self.writer,x_label,"Corrupted image (x_label) over " + pet_algo + " " + iteration_name + " (FULL CONTRAST)",suffix,self.image_gt,i,full_contrast=True) # Showing each corrupted image with contrast = 1

    def writeEndImagesAndMetrics(self,i,max_iter,PETImage_shape,f,suffix,phantom,net,pet_algo,iteration_name='iterations'):       
        # Metrics for NN output
        if ("3D" not in phantom):
            self.compute_metrics(PETImage_shape,f,self.image_gt,i,self.PSNR_recon,self.PSNR_norm_recon,self.MSE_recon,self.SSIM_recon,self.MA_cold_recon,self.AR_hot_recon,self.AR_hot_TEP_recon,self.AR_hot_TEP_match_square_recon,self.AR_hot_perfect_match_recon,self.loss_DIP_recon,self.CRC_hot_recon,self.AR_bkg_recon,self.IR_bkg_recon,self.IR_whole_recon,self.mean_inside_recon,phantom,writer=self.writer)

        if (self.tensorboard):
            # Write image over ADMM iterations
            if (self.all_images_DIP == "Last"):
                self.write_image_tensorboard(self.writer,f,"Image over " + pet_algo + " " + iteration_name + "(" + net + "output)",suffix,self.image_gt,i) # Showing all images with same contrast to compare them together
                self.write_image_tensorboard(self.writer,f,"Image over " + pet_algo + " " + iteration_name + "(" + net + "output, FULL CONTRAST)",suffix,self.image_gt,i,full_contrast=True) # Showing each image with contrast = 1
                self.write_image_tensorboard(self.writer,f*self.phantom_ROI,"Image over " + pet_algo + " " + iteration_name + "(" + net + "output, FULL CONTRAST CROPPED)",suffix,self.image_gt,i,full_contrast=True) # Showing each image with contrast = 1
            else:
                if (((max_iter>=10) and (i%(max_iter // 10) == 0)) or (max_iter<10)):
                    self.write_image_tensorboard(self.writer,f,"Image over " + pet_algo + " " + iteration_name + "(" + net + "output)",suffix,self.image_gt,i) # Showing all images with same contrast to compare them together
                    self.write_image_tensorboard(self.writer,f,"Image over " + pet_algo + " " + iteration_name + "(" + net + "output, FULL CONTRAST)",suffix,self.image_gt,i,full_contrast=True) # Showing each image with contrast = 1
                    self.write_image_tensorboard(self.writer,f*self.phantom_ROI,"Image over " + pet_algo + " " + iteration_name + "(" + net + "output, FULL CONTRAST CROPPED)",suffix,self.image_gt,i,full_contrast=True) # Showing each image with contrast = 1

    def runComputation(self,config,root):
        if ("read_only_MV_csv" in config):
            if (config["read_only_MV_csv"]):
                if ("nested" in config["method"] or "Gong" in config["method"]):
                    self.MV_several_alphas_plot(config)
        else:
            if (hasattr(self,'beta')):
                self.beta_string = ', beta = ' + str(self.beta)

            if (('nested' in config["method"] or  'Gong' in config["method"]) and "results" not in config["task"]):
                self.writeBeginningImages(self.suffix,self.image_net_input) # Write GT and DIP input
                self.writeCorruptedImage(0,self.total_nb_iter,self.image_corrupt,self.suffix,pet_algo="to fit",iteration_name="(post reconstruction)")
            else:
                # self.writeBeginningImages(self.suffix) # Write GT
                if ('nested' in config["method"] or  'Gong' in config["method"]):
                    if (not hasattr(self,"image_net_input")):
                        self.input = config["input"]
                        self.override_input = False
                        self.image_net_input = self.load_input(self.net,self.PETImage_shape,self.subroot_data) # Scaling of network input. DO NOT CREATE RANDOM INPUT IN BLOCK 2 !!! ONLY AT THE BEGINNING, IN BLOCK 1    

                        # modify input with line on the edge of the phantom, or to remove a region (DIP input tests)
                        # self.modify_input_line_edge(config)            

            if (self.FLTNB == 'float'):
                type_im = '<f'
            elif (self.FLTNB == 'double'):
                type_im = '<d'

            self.f = np.zeros(self.PETImage_shape,dtype=type_im)
            f_p = np.zeros(self.PETImage_shape,dtype=type_im)

            # Nested ADMM stopping criterion
            if ("3_" not in self.phantom):
                if ('nested' in config["method"]):
                    # Compute IR for BSREM initialization image
                    im_BSREM = self.fijii_np(self.subroot_data + 'Data/initialization/' + self.phantom + '/BSREM_30it' + '/replicate_' + str(self.replicate) + '/BSREM_it30.img',shape=(self.PETImage_shape),type_im='<d') # loading BSREM initialization image
                    self.IR_ref = [np.NaN]
                    self.compute_IR_whole(self.PETImage_shape,im_BSREM,0,self.IR_ref,self.phantom)
                    # Add 1 to number of iterations before stopping criterion
                    self.total_nb_iter += 1

            for i in range(self.i_init,self.total_nb_iter+self.i_init):
                self.IR = 0
                self.IR_whole = 0
                if (self.loop_on_replicates(config,i)):
                    break    

                if ("3D" not in self.phantom):
                    self.IR_bkg_recon[int((i-self.i_init))] = self.IR
                    self.IR_whole_recon[int((i-self.i_init))] = self.IR_whole
                    if (self.tensorboard):
                        #print("IR saved in tensorboard")
                        self.writer.add_scalar('Image roughness in the background (best : 0)', self.IR_bkg_recon[int((i-self.i_init))], i)
                        self.writer.add_scalar('Image roughness in whole phantom', self.IR_whole_recon[int((i-self.i_init))], i)

                # Show images and metrics in tensorboard (averaged images if asked in config)
                print('Metrics for iteration',int((i-self.i_init)))
                self.writeEndImagesAndMetrics(int((i-self.i_init)),self.total_nb_iter,self.PETImage_shape,self.f,self.suffix,self.phantom,self.net,self.pet_algo,self.iteration_name)

            print("loop over")

            if ("nested" in config["method"] or "Gong" in config["method"]):
                if(config["DIP_early_stopping"]):# WMV
                    self.WMV_plot(config)

    def WMV_plot(self,config):

        self.VAR_recon_original = np.copy(self.VAR_recon)

        # 2.2 plot window moving variance
        plt.figure(1)
        if (config["EMV_or_WMV"] == "WMV"):
            var_x = np.arange(self.windowSize-1, self.windowSize + len(self.VAR_recon)-1)  # define x axis of WMV
        else:
            var_x = np.arange(len(self.VAR_recon))  # define x axis of EMV
            # var_x = np.arange(self.patienceNumber + self.epochStar + 1)  # define x axis of EMV
        
        # Save computed variance from WMV/EMV in csv
        with open(self.MV_csv_path(self.alpha_EMV,config), 'w', newline='') as myfile:
            wr = writer_csv(myfile,delimiter=';')
            wr.writerow(self.VAR_recon)        
        
        # Remove first iterations 
        remove_first_iterations = 0
        # remove_first_iterations = 200
        # Remove last iterations
        last_iteration = self.total_nb_iter
        last_iteration = 1000
        
        var_x = var_x[remove_first_iterations:last_iteration+1]
        self.VAR_recon = self.VAR_recon[remove_first_iterations:last_iteration+1]
        self.MSE_WMV = self.MSE_WMV[remove_first_iterations:last_iteration+1]
        self.PSNR_WMV = self.PSNR_WMV[remove_first_iterations:last_iteration+1]
        self.SSIM_WMV = self.SSIM_WMV[remove_first_iterations:last_iteration+1]

        plt.plot(var_x, self.VAR_recon, 'r')
        plt.title('Window Moving Variance,epoch*=' + str(self.epochStar) + ',lr=' + str(self.lr))
        plt.axvline(self.epochStar, c='g')  # plot a vertical line at self.epochStar(detection point)
        # plt.xticks([self.epochStar, 0, self.total_nb_iter-1], [self.epochStar, 0, self.total_nb_iter-1], color='green')
        plt.axhline(y=np.min(self.VAR_recon), c="black", linewidth=0.5)
        if (config["EMV_or_WMV"] == "WMV"):
            plt.savefig(self.mkdir(self.subroot + '/self.VAR_recon/' + self.suffix + '/w' + str(self.windowSize) + 'p' + str(self.patienceNumber)) + '/' + str(
            self.lr) + '-lr' + str(self.lr) + '+self.VAR_recon-w' + str(self.windowSize) + 'p' + str(self.patienceNumber) + '_' + str(remove_first_iterations) + '.png')
        else:
            plt.savefig(self.mkdir(self.subroot + '/self.VAR_recon/' + self.suffix + '/w' + str(self.alpha_EMV) + 'p' + str(self.patienceNumber)) + '/' + str(
            self.lr) + '-lr' + str(self.lr) + '+self.VAR_recon-w' + str(self.alpha_EMV) + 'p' + str(self.patienceNumber) + '_' + str(remove_first_iterations) + '.png')

        # Save WMV in tensorboard
        #print("WMV saved in tensorboard")
        for i in range(len(self.VAR_recon)):
            self.writer.add_scalar('WMV in the phantom (should follow MSE trend to find peak)', self.VAR_recon[i], var_x[i])

        # 2.3 plot MSE
        plt.figure(2)
        plt.plot(var_x,self.MSE_WMV, 'y')
        plt.title('MSE,epoch*=' + str(self.epochStar) + ',lr=' + str(self.lr))
        plt.axvline(self.epochStar, c='g')
        # plt.xticks([self.epochStar, 0, self.total_nb_iter-1], [self.epochStar, 0, self.total_nb_iter-1], color='green')
        plt.axhline(y=np.min(self.MSE_WMV), c="black", linewidth=0.5)
        if (config["EMV_or_WMV"] == "WMV"):
            plt.savefig(self.mkdir(self.subroot + '/self.MSE_WMV/' + self.suffix + '/w' + str(self.windowSize) + 'p' + str(self.patienceNumber)) + '/' + str(
                self.lr) + '-lr' + str(self.lr) + '+self.MSE_WMV-w' + str(self.windowSize) + 'p' + str(self.patienceNumber) + '_' + str(remove_first_iterations) + '.png')
        else:
            plt.savefig(self.mkdir(self.subroot + '/self.MSE_WMV/' + self.suffix + '/w' + str(self.alpha_EMV) + 'p' + str(self.patienceNumber)) + '/' + str(
            self.lr) + '-lr' + str(self.lr) + '+self.MSE_WMV-w' + str(self.alpha_EMV) + 'p' + str(self.patienceNumber) + '_' + str(remove_first_iterations) + '.png')

        # 2.4 plot PSNR
        plt.figure(3)
        plt.plot(var_x,self.PSNR_WMV)
        plt.title('PSNR,epoch*=' + str(self.epochStar) + ',lr=' + str(self.lr))
        plt.axvline(self.epochStar, c='g')
        # plt.xticks([self.epochStar, 0, self.total_nb_iter - 1], [self.epochStar, 0, self.total_nb_iter - 1], color='green')
        plt.axhline(y=np.max(self.PSNR_WMV), c="black", linewidth=0.5)
        if (config["EMV_or_WMV"] == "WMV"):
            plt.savefig(self.mkdir(self.subroot + '/self.PSNR_WMV/' + self.suffix + '/w' + str(self.windowSize) + 'p' + str(self.patienceNumber)) + '/' + str(
                self.lr) + '-lr' + str(self.lr) + '+self.PSNR_WMV-w' + str(self.windowSize) + 'p' + str(self.patienceNumber) + '_' + str(remove_first_iterations) + '.png')
        else:
            plt.savefig(self.mkdir(self.subroot + '/self.PSNR_WMV/' + self.suffix + '/w' + str(self.alpha_EMV) + 'p' + str(self.patienceNumber)) + '/' + str(
            self.lr) + '-lr' + str(self.lr) + '+self.PSNR_WMV-w' + str(self.alpha_EMV) + 'p' + str(self.patienceNumber) + '_' + str(remove_first_iterations) + '.png')
        #'''
        # 2.5 plot SSIM
        plt.figure(4)
        plt.plot(var_x,self.SSIM_WMV, c='orange')
        plt.title('SSIM,epoch*=' + str(self.epochStar) + ',lr=' + str(self.lr))
        plt.axvline(self.epochStar, c='g')
        # plt.xticks([self.epochStar, 0, self.total_nb_iter - 1], [self.epochStar, 0, self.total_nb_iter - 1], color='green')
        plt.axhline(y=np.max(self.SSIM_WMV), c="black", linewidth=0.5)
        if (config["EMV_or_WMV"] == "WMV"):
            plt.savefig(self.mkdir(self.subroot + '/self.SSIM_WMV/' + self.suffix + '/w' + str(self.windowSize) + 'p' + str(self.patienceNumber)) + '/' + str(
                self.lr) + '-lr' + str(self.lr) + '+self.SSIM_WMV-w' + str(self.windowSize) + 'p' + str(self.patienceNumber) + '_' + str(remove_first_iterations) + '.png')
        else:
            plt.savefig(self.mkdir(self.subroot + '/self.SSIM_WMV/' + self.suffix + '/w' + str(self.alpha_EMV) + 'p' + str(self.patienceNumber)) + '/' + str(
            self.lr) + '-lr' + str(self.lr) + '+self.SSIM_WMV-w' + str(self.alpha_EMV) + 'p' + str(self.patienceNumber) + '_' + str(remove_first_iterations) + '.png')
        
        #'''
        
        # 2.6 plot all the curves together
        fig, ax1 = plt.subplots()
        fig.subplots_adjust(right=0.8, left=0.1, bottom=0.12)
        ax2 = ax1.twinx()  # creat other y-axis for different scale
        ax3 = ax1.twinx()  # creat other y-axis for different scale
        ax4 = ax1.twinx()  # creat other y-axis for different scale
        if (config["EMV_or_WMV"] == "WMV"):
            ax2.spines.right.set_position(("axes", 1.18))
        p4, = ax4.plot(var_x,self.MSE_WMV, "y", label="MSE")
        p1, = ax1.plot(var_x,self.PSNR_WMV, label="PSNR")
        p2, = ax2.plot(var_x, self.VAR_recon, "r", label="WMV")
        p3, = ax3.plot(var_x,self.SSIM_WMV, "orange", label="SSIM")
        #ax1.set_xlim(0, self.total_nb_iter-1)
        ax1.set_xlim(0, min(self.epochStar+self.patienceNumber,self.total_nb_iter-1))
        plt.title('skip : ' + str(config["skip_connections"]) + ' lr=' + str(self.lr))
        ax1.set_ylabel("Peak Signal-Noise ratio")
        ax2.set_ylabel("Window-Moving variance")
        ax3.set_ylabel("Structural similarity")
        ax4.yaxis.set_visible(False)
        ax1.yaxis.label.set_color(p1.get_color())
        ax2.yaxis.label.set_color(p2.get_color())
        ax3.yaxis.label.set_color(p3.get_color())
        tkw = dict(size=3, width=1)
        ax1.tick_params(axis='y', colors=p1.get_color(), **tkw)
        ax1.tick_params(axis='x', colors="green", **tkw)
        ax2.tick_params(axis='y', colors=p2.get_color(), **tkw)
        ax3.tick_params(axis='y', colors=p3.get_color(), **tkw)
        ax1.tick_params(axis='x', **tkw)
        ax1.legend(handles=[p1, p3, p2, p4])
        ax1.axvline(self.epochStar, c='g', linewidth=1, ls='--')
        if (config["EMV_or_WMV"] == "WMV"):
            ax1.axvline(self.windowSize-1, c='g', linewidth=1, ls=':')
        ax1.axvline(self.epochStar+self.patienceNumber, c='g', lw=1, ls=':')
        if (config["EMV_or_WMV"] == "WMV"):
            if self.epochStar+self.patienceNumber > self.epochStar:
                plt.xticks([self.epochStar, self.windowSize-1, self.epochStar+self.patienceNumber], ['\n' + str(self.epochStar) + '\nES point', str(self.windowSize), '+' + str(self.patienceNumber)], color='green')
            else:
                plt.xticks([self.epochStar, self.windowSize-1], ['\n' + str(self.epochStar) + '\nES point', str(self.windowSize)], color='green')
            
            plt.savefig(self.mkdir(self.subroot + '/combined/' + self.suffix + '/w' + str(self.windowSize) + 'p' + str(self.patienceNumber)) + '/' + str(
                self.lr) + '-lr' + str(self.lr) + '+combined-w' + str(self.windowSize) + 'p' + str(self.patienceNumber) + '_' + str(remove_first_iterations) + '.png')
        else:
            plt.savefig(self.mkdir(self.subroot + '/combined/' + self.suffix + '/w' + str(self.alpha_EMV) + 'p' + str(self.patienceNumber)) + '/' + str(
                self.lr) + '-lr' + str(self.lr) + '+combined-w' + str(self.alpha_EMV) + 'p' + str(self.patienceNumber) + '_' + str(remove_first_iterations) + '.png')

        # 2.4 plot PSNR
        plt.figure(3)
        plt.plot(self.PSNR_WMV)

        '''
        N = 100
        moving_average_PSNR = self.moving_average(self.PSNR_WMV,N)
        plt.plot(np.arange(N-1,len(self.PSNR_WMV)), moving_average_PSNR)
        '''


        plt.title('PSNR,epoch*=' + str(self.epochStar) + ',lr=' + str(self.lr))
        plt.axvline(self.epochStar, c='g')
        # plt.xticks([self.epochStar, 0, self.total_nb_iter - 1], [self.epochStar, 0, self.total_nb_iter - 1], color='green')
        plt.axhline(y=np.max(self.PSNR_WMV), c="black", linewidth=0.5)
        if (config["EMV_or_WMV"] == "WMV"):
            plt.savefig(self.mkdir(self.subroot + '/self.PSNR_WMV/' + self.suffix + '/w' + str(self.windowSize) + 'p' + str(self.patienceNumber)) + '/' + str(
                self.lr) + '-lr' + str(self.lr) + '+self.PSNR_WMV-w' + str(self.windowSize) + 'p' + str(self.patienceNumber) + '_' + str(remove_first_iterations) + '.png')
        else:
            plt.savefig(self.mkdir(self.subroot + '/self.PSNR_WMV/' + self.suffix + '/w' + str(self.alpha_EMV) + 'p' + str(self.patienceNumber)) + '/' + str(
            self.lr) + '-lr' + str(self.lr) + '+self.PSNR_WMV-w' + str(self.alpha_EMV) + 'p' + str(self.patienceNumber) + '_' + str(remove_first_iterations) + '.png')
    
    def MV_csv_path(self,alpha_EMV,config):
        if ("read_only_MV_csv" in config):
            return self.subroot_metrics + self.method + '/' + self.suffix_metrics + '/' + config["EMV_or_WMV"] + '_alpha=' + str(alpha_EMV) + "_nb_it=" + str(self.total_nb_iter) + '.csv'
        else:
            return self.subroot_metrics + self.method + '/' + self.suffix_metrics + '/' + config["EMV_or_WMV"] + '_alpha=' + str(alpha_EMV) + "_nb_it=" + str(self.total_nb_iter-1) + '.csv'

    def MV_several_alphas_plot(self,config):

        # Remove first iterations 
        remove_first_iterations = 100
        # remove_first_iterations = 300
        # remove_first_iterations = 400
        # remove_first_iterations = 1500
        # Remove last iterations
        last_iteration = self.total_nb_iter
        last_iteration = 3000

        for smooth in [True,False]:
            plt.figure()
            if (config["EMV_or_WMV"] == "EMV"):
                # alpha_list = [0.01,0.0251,0.1,0.5,0.99]
                alpha_list = [0.0251,0.1,0.5,0.9]
                # alpha_list = [0.1,0.5,0.9]
                # alpha_list = [1]
                self.MV = len(alpha_list) * [0]
                self.MV_original = len(alpha_list) * [0]
                alpha_list = sorted(alpha_list,reverse=True)
                for alpha_idx in range(len(alpha_list)):
                    alpha_EMV = alpha_list[alpha_idx]
                    with open(self.MV_csv_path(alpha_EMV,config), 'r') as myfile:
                        spamreader = reader_csv(myfile,delimiter=';')
                        rows_csv = list(spamreader)
                        self.MV[alpha_idx] = [float(rows_csv[0][i]) for i in range(int(self.i_init) - 1,min(len(rows_csv[0]),self.total_nb_iter))]
                        self.MV[alpha_idx] = self.MV[alpha_idx][remove_first_iterations:last_iteration+1]
                        self.MV_original[alpha_idx] = self.MV[alpha_idx]

                    
                    # Smooth MV curve
                    alpha_tmp = 0.025
                    # smooth = False
                    if (smooth):
                        self.MV[alpha_idx][0] = 0
                        for i in range(1,len(self.MV[alpha_idx])):
                            self.MV[alpha_idx][i] = (1-alpha_tmp) * self.MV[alpha_idx][i-1] + alpha_tmp * self.MV_original[alpha_idx][i]


                    plt.plot(np.arange(remove_first_iterations,min(last_iteration+1,self.total_nb_iter+1)),self.MV[alpha_idx],label=alpha_EMV)
                    plt.legend()
                    print("MV min for alpha_EMV=",alpha_EMV,"at it= (10 first removed, after ",1000+remove_first_iterations," also)",np.argmin(self.MV[alpha_idx][10:1000]) + remove_first_iterations)
                plt.savefig(self.mkdir(self.subroot + '/several_alphas/' + self.suffix) + '/' + str(
                self.lr) + '-lr' + str(self.lr) + '+several_alphas' '_' + str(alpha_list) + '_' + str(remove_first_iterations) + '_' + str(last_iteration) + '_smooth=' + str(smooth) + '.png')

    def loop_on_replicates(self,config,i):
        for p in range(1,self.nb_replicates+1):
            if (config["average_replicates"] or (config["average_replicates"] == False and p == self.replicate)):
                self.subroot_p = self.subroot_data + 'debug/'*self.debug + '/' + self.phantom + '/' + 'replicate_' + str(p) + '/' + self.method + '/' # Directory root

                # Take NNEPPS images if NNEPPS is asked for this run
                if (config["NNEPPS"]):
                    NNEPPS_string = "_NNEPPS"
                else:
                    NNEPPS_string = ""
                if ( 'Gong' in config["method"] or  'nested' in config["method"]):
                    if ('post_reco' in config["task"]):
                        self.pet_algo=config["method"]+"to fit"
                        self.iteration_name="(post reconstruction)"
                    else:
                        self.pet_algo=config["method"]
                        self.iteration_name="iterations"
                    if ('post_reco' in config["task"]):
                        try:
                            f_p = self.fijii_np(self.subroot_p+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.net + '' + format(self.global_it) + '_epoch=' + format(i-self.i_init) + NNEPPS_string + '.img',shape=(self.PETImage_shape),type_im='<f') # loading DIP output



                            # out = f_p
                            # # Descale like at the beginning
                            # out_descale = self.descale_imag(out,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,config["scaling"])
                            # #'''
                            # # Saving image output
                            # net_outputs_path = self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + format(self.global_it) + '_epoch=' + format(i) + '.img'
                            # self.save_img(out_descale, net_outputs_path)
                            # # Squeeze image by loading it
                            # out_descale = self.fijii_np(net_outputs_path,shape=(self.PETImage_shape),type_im='<f') # loading DIP output
                            # # Saving (now DESCALED) image output
                            # self.save_img(out_descale, net_outputs_path)


                        except:
                            print("!!!!! failed to read image")
                            break
                    else:
                        f_p = self.fijii_np(self.subroot_p+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.net + '' + format(i-1) + "_FINAL" + NNEPPS_string + '.img',shape=(self.PETImage_shape),type_im='<f') # loading DIP output
                    if config["FLTNB"] == "double":
                        f_p = f_p.astype(np.float64)

                elif ('ADMMLim' in config["method"] or config["method"] == 'MLEM' or config["method"] == 'OPTITR' or config["method"] == 'OSEM' or config["method"] == 'BSREM' or config["method"] == 'AML' or 'APGMAP' in config["method"]):
                    self.pet_algo=config["method"]
                    self.iteration_name = "iterations"
                    if (hasattr(self,'beta')):
                        self.iteration_name += self.beta_string
                    if ('ADMMLim' in config["method"]):
                        subdir = 'ADMM' + '_' + str(config["nb_threads"])
                        subdir = ''
                        #f_p = self.fijii_np(self.subroot_p + self.suffix + '/' + subdir + '/0_' + format(i) + '_it' + str(config["nb_inner_iteration"]) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                        #f_p = self.fijii_np(self.subroot_p + self.suffix + '/' + subdir + '/0_' + format(i) + '_it1' + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                        #f_p = self.fijii_np(self.subroot_p + self.suffix + '/' + subdir + '/0_1'  + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                        f_p = self.fijii_np(self.subroot_p + self.suffix + '/' + subdir + '/0'  + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                    #elif (config["method"] == 'BSREM'):
                    #    f_p = self.fijii_np(self.subroot_p + self.suffix + '/' +  config["method"] + '_beta_' + str(self.beta) + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                    else:
                        if ('APGMAP' in config["method"]):
                            f_p = self.fijii_np(self.subroot_p + self.suffix + '/' +  "APGMAP" + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                        else:
                            f_p = self.fijii_np(self.subroot_p + self.suffix + '/' +  config["method"] + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output

                # Compute IR metric (different from others with several replicates)
                if ("3D" not in self.phantom):
                    self.compute_IR_bkg(self.PETImage_shape,f_p,int((i-self.i_init)),self.IR_bkg_recon,self.phantom)
                    self.compute_IR_whole(self.PETImage_shape,f_p,int((i-self.i_init)),self.IR_whole_recon,self.phantom)

                    # # Nested ADMM stopping criterion
                    # if('nested' in config["method"]):
                    #     if (self.IR_whole_recon[int((i-self.i_init))]> self.IR_ref[0]): # > 1.604):# > self.IR_ref[0]):
                    #         print("Nested ADMM stopping criterion reached")
                    #         self.path_stopping_criterion = self.subroot + 'Block2/' + self.suffix + '/' + 'IR_stopping_criteria.log'
                    #         stopping_criterion_file = open(self.path_stopping_criterion, "w")
                    #         stopping_criterion_file.write("stopping iteration :" + "\n")
                    #         stopping_criterion_file.write(str(i) + "\n")
                    #         stopping_criterion_file.close()
                    #         return 1

                    # Specific average for IR
                    if (config["average_replicates"] == False and p == self.replicate):
                        self.IR = self.IR_bkg_recon[int((i-self.i_init))]
                        self.IR_whole = self.IR_whole_recon[int((i-self.i_init))]
                    elif (config["average_replicates"]):
                        self.IR += self.IR_bkg_recon[int((i-self.i_init))] / self.nb_replicates
                        self.IR_whole += self.IR_whole_recon[int((i-self.i_init))]

                if (config["average_replicates"]): # Average images across replicates (for metrics except IR)
                    self.f += f_p / self.nb_replicates
                elif (config["average_replicates"] == False and p == self.replicate):
                    self.f = f_p
            
                # WMV
                if ("nested" in config["method"] or "Gong" in config["method"]):
                    # self.run_WMV(f_p,self.config,self.fixed_hyperparameters_list,self.hyperparameters_list,self.debug,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,config["scaling"],self.suffix,self.global_it,self.root,self.scanner,i)
                    if(config["DIP_early_stopping"]):# WMV
                        self.run_WMV(f_p,self.config,self.fixed_hyperparameters_list,self.hyperparameters_list,self.debug,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,config["scaling"],self.suffix,self.global_it,self.root,self.scanner,i)
                        if (self.SUCCESS):
                            return 1
                del f_p


                # Normal end
                return 0
            
    def moving_average(self, series, n):
        # MVA
        ret = np.cumsum(series, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        # EMA
        import pandas as pd
        pd_series = pd.DataFrame(series)
        return pd_series.ewm(com=0.4).mean()

    def compute_IR_bkg(self, PETImage_shape, image_recon,i,IR_bkg_recon,image):
        # radius - 1 is to remove partial volume effect in metrics computation / radius + 1 must be done on cold and hot ROI when computing background ROI, because we want to exclude those regions from big cylinder
        bkg_ROI_act = image_recon[self.bkg_ROI==1]
        IR_bkg_recon[i] = (np.std(bkg_ROI_act) / np.mean(bkg_ROI_act))
        #print("IR_bkg_recon",IR_bkg_recon)
        #print('Image roughness in the background', IR_bkg_recon[i],' , must be as small as possible')

    def compute_IR_whole(self, PETImage_shape, image_recon,i,IR_whole_recon,image):
        ROI_act = image_recon
        IR_whole_recon[i] = (np.std(ROI_act) / np.mean(ROI_act))
        print("IR_whole_recon",IR_whole_recon)
        #print('Image roughness in the background', IR_whole_recon[i],' , must be as small as possible')


    def compute_metrics(self, PETImage_shape, image_recon,image_gt,i,PSNR_recon,PSNR_norm_recon,MSE_recon,SSIM_recon,MA_cold_recon,AR_hot_recon,AR_hot_TEP_recon,AR_hot_TEP_match_square_recon,AR_hot_perfect_match_recon,loss_DIP_recon,CRC_hot_recon,AR_bkg_recon,IR_bkg_recon,IR_whole_recon,mean_inside_recon,image,writer=None):
        # radius - 1 is to remove partial volume effect in metrics computation / radius + 1 must be done on cold and hot ROI when computing background ROI, because we want to exclude those regions from big cylinder
        image_recon_cropped = image_recon*self.phantom_ROI
        image_recon_norm = self.norm_imag(image_recon_cropped)[0] # normalizing DIP output
        image_gt_cropped = image_gt * self.phantom_ROI
        image_gt_norm = self.norm_imag(image_gt_cropped)[0]

        # PSNR calculation
        PSNR_whole = peak_signal_noise_ratio(image_gt, image_recon, data_range=np.amax(image_recon_cropped) - np.amin(image_recon_cropped)) # PSNR with true values
        PSNR_recon[i] = peak_signal_noise_ratio(image_gt_cropped, image_recon_cropped, data_range=np.amax(image_recon_cropped) - np.amin(image_recon_cropped)) # PSNR with true values
        PSNR_norm_recon[i] = peak_signal_noise_ratio(image_gt_norm,image_recon_norm) # PSNR with scaled values [0-1]

        # MSE calculation
        MSE_recon[i] = np.mean((image_gt_cropped - image_recon_cropped)**2)
        
        # SSIM calculation
        '''
        To match the implementation of Wang et al. [1]_, set `gaussian_weights`
        to True, `sigma` to 1.5, `use_sample_covariance` to False, and
        specify the `data_range` argument.
        '''
        if (PETImage_shape[0] < 11): # SSIM cannot be calculated with this window size for a small image
            SSIM_recon[i] = np.NaN
        else:
            SSIM_recon[i] = structural_similarity(np.squeeze(image_gt_cropped), np.squeeze(image_recon_cropped), data_range=(image_recon_cropped).max() - (image_recon_cropped).min())
        # SSIM_recon[i] = structural_similarity(np.squeeze(image_gt_cropped), np.squeeze(image_recon_cropped), data_range=(image_recon_cropped).max() - (image_recon_cropped).min(), sigma=1.5, gaussian_weights=True, use_sample_covariance=False)

        # Contrast Recovery Coefficient calculation    
        # Mean activity in cold cylinder calculation (-c -40. -40. 0. 40. 4. 0.)
        cold_ROI_act = image_recon[self.cold_ROI==1]
        MA_cold_recon[i] = np.mean(cold_ROI_act)
        cold_inside_ROI_act = image_recon[self.cold_inside_ROI==1]
        self.MA_cold_inside[i] = np.mean(cold_inside_ROI_act)
        cold_edge_ROI_act = image_recon[self.cold_edge_ROI==1]
        self.MA_cold_edge[i] = np.mean(cold_edge_ROI_act)

        # DIP loss function
        if ( 'nested' in self.method or  'Gong' in self.method):
            loss_DIP_recon[i] = np.mean((self.image_corrupt * self.phantom_ROI - image_recon_cropped)**2)

        # Mean Activity Recovery (ARmean) in hot cylinder calculation (-c 50. 10. 0. 20. 4. 400)
        hot_ROI_act = image_recon[self.hot_ROI==1]
        AR_hot_recon[i] = np.mean(hot_ROI_act)
        CRC_hot_recon[i] = np.mean(hot_ROI_act)
        
        ### Only useful for new phantom with 3 hot ROIs, but compute it for every phantom for the sake of simplicity ###
        # Mean Activity Recovery (ARmean) in hot cylinder calculation (-c 50. 10. 0. 20. 4. 400)
        hot_TEP_ROI_act = image_recon[self.hot_TEP_ROI==1]
        AR_hot_TEP_recon[i] = np.mean(hot_TEP_ROI_act)
        
        # Mean Activity Recovery (ARmean) in hot cylinder calculation (-c -20. 70. 0. 20. 4. 400)
        hot_TEP_match_square_ROI_act = image_recon[self.hot_TEP_match_square_ROI==1]
        AR_hot_TEP_match_square_recon[i] = np.mean(hot_TEP_match_square_ROI_act)
        
        # Mean Activity Recovery (ARmean) in hot cylinder calculation (-c 50. 90. 0. 20. 4. 400)
        hot_perfect_match_ROI_act = image_recon[self.hot_perfect_match_ROI==1]
        AR_hot_perfect_match_recon[i] = np.mean(hot_perfect_match_ROI_act)

        # Mean Activity Recovery (ARmean) in background calculation (-c 0. 0. 0. 150. 4. 100)
        bkg_ROI_act = image_recon[self.bkg_ROI==1]
        AR_bkg_recon[i] = np.mean(bkg_ROI_act) / 100.

        # Mean in whole denoised image
        # mean_inside_recon[i] = np.mean(image_recon) / np.mean(image_gt)
        if ("nested" in self.method or "Gong" in self.method):
            mean_inside_recon[i] = np.mean(image_recon * self.phantom_ROI) / np.mean(self.image_corrupt * self.phantom_ROI)
        
        # Likelihood from fake CASToR reconstruction just to compute likelihood of initialization image        
        if 'nested' in self.config["method"] or 'Gong' in self.config["method"]:
            folder_sub_path = self.subroot + 'Block2/' + self.suffix
        else:
            folder_sub_path = self.subroot + '/' + self.suffix
        if 'nested' in self.config["method"] or 'Gong' in self.config["method"]:
            logfile_name = self.config["method"] + '_' + str(i-1) + '.log'
        else:
            logfile_name = self.config["method"] + '_' + str(i+1) + '.log'
        path_log = folder_sub_path + '/' + logfile_name
        if (isfile(path_log)):
            self.extract_likelihood_from_log(path_log)

        del image_recon_cropped
        del image_gt_cropped

        # Save metrics in csv
        Path(self.subroot_metrics + self.method + '/' + self.suffix_metrics).mkdir(parents=True, exist_ok=True) # CASToR path
        with open(self.subroot_metrics + self.method + '/' + self.suffix_metrics + '/metrics.csv', 'w', newline='') as myfile:
            wr = writer_csv(myfile,delimiter=';')
            wr.writerow(PSNR_recon)
            wr.writerow(PSNR_norm_recon)
            wr.writerow(MSE_recon)
            wr.writerow(SSIM_recon)
            wr.writerow(MA_cold_recon)
            wr.writerow(AR_hot_recon)
            wr.writerow(AR_hot_TEP_recon)
            wr.writerow(AR_hot_TEP_match_square_recon)
            wr.writerow(AR_hot_perfect_match_recon)
            wr.writerow(AR_bkg_recon)
            wr.writerow(IR_bkg_recon)
            # wr.writerow(loss_DIP_recon)
            wr.writerow(mean_inside_recon)
            wr.writerow(CRC_hot_recon)
            wr.writerow(IR_whole_recon)
            if ("results" in self.config["task"]):
                wr.writerow(self.likelihoods)

        # Save cold inside and edge
        with open(self.subroot_metrics + self.method + '/' + self.suffix_metrics + '/metrics_cold.csv', 'w', newline='') as myfile:
            wr = writer_csv(myfile,delimiter=';')
            wr.writerow(MA_cold_recon)
            wr.writerow(self.MA_cold_inside)
            wr.writerow(self.MA_cold_edge)
        
        # Show metrics in tensorboard
        if (self.tensorboard):
            print("Metrics saved in tensorboard")
            writer.add_scalar('PSNR gt (best : inf)', PSNR_recon[i], i)
            writer.add_scalar('MSE gt (best : 0)', MSE_recon[i], i)
            writer.add_scalar('SSIM gt (best : 0)', SSIM_recon[i], i)
            writer.add_scalar('Mean activity in cold cylinder (best : 0)', MA_cold_recon[i], i)
            writer.add_scalar('Mean Concentration Recovery coefficient in hot cylinder (best : 1)', AR_hot_recon[i], i)
            writer.add_scalar('Mean Concentration Recovery coefficient in hot (only in TEP) cylinder (best : 1)', AR_hot_TEP_recon[i], i)
            writer.add_scalar('Mean Concentration Recovery coefficient in hot (square MR, circle TEP) cylinder (best : 1)', AR_hot_TEP_match_square_recon[i], i)
            writer.add_scalar('Mean Concentration Recovery coefficient in hot (perfect match) cylinder (best : 1)', AR_hot_perfect_match_recon[i], i)
            writer.add_scalar('Mean Concentration Recovery coefficient in background (best : 1)', AR_bkg_recon[i], i)
            writer.add_scalar('DIP loss', loss_DIP_recon[i], i)
            writer.add_scalar('Mean inside phantom', mean_inside_recon[i], i)
            #writer.add_scalar('Image roughness in the background (best : 0)', IR_bkg_recon[i], i)


    def initialize_WMV(self,config,fixed_hyperparameters_list,hyperparameters_list,debug,param1_scale_im_corrupt,param2_scale_im_corrupt,scaling_input,suffix,global_it,root, scanner):
        self.classWMV = iWMV(config)            
        self.classWMV.fixed_hyperparameters_list = fixed_hyperparameters_list
        self.classWMV.hyperparameters_list = hyperparameters_list
        self.classWMV.debug = debug
        self.classWMV.param1_scale_im_corrupt = param1_scale_im_corrupt
        self.classWMV.param2_scale_im_corrupt = param2_scale_im_corrupt
        self.classWMV.scaling_input = scaling_input
        self.classWMV.suffix = suffix
        self.classWMV.global_it = global_it
        self.classWMV.scanner = scanner
        # Initialize variables
        self.classWMV.do_everything(config,root)

    def run_WMV(self,out,config,fixed_hyperparameters_list,hyperparameters_list,debug,param1_scale_im_corrupt,param2_scale_im_corrupt,scaling_input,suffix,global_it,root,scanner,i):
        if (config["DIP_early_stopping"]):
            self.SUCCESS = self.classWMV.SUCCESS

            self.classWMV.SUCCESS,self.classWMV.VAR_min,self.classWMV.stagnate = self.classWMV.WMV(out,i,self.classWMV.queueQ,self.classWMV.SUCCESS,self.classWMV.VAR_min,self.classWMV.stagnate,descale=False)
            self.VAR_recon = self.classWMV.VAR_recon
            self.MSE_WMV = self.classWMV.MSE_WMV
            self.PSNR_WMV = self.classWMV.PSNR_WMV
            self.SSIM_WMV = self.classWMV.SSIM_WMV
            self.epochStar = self.classWMV.epochStar
            if config["EMV_or_WMV"] == "EMV":
                self.alpha_EMV = self.classWMV.alpha_EMV
            else:
                self.windowSize = self.classWMV.windowSize
            self.patienceNumber = self.classWMV.patienceNumber

            if self.SUCCESS:
                print("SUCCESS WMVVVVVVVVVVVVVVVVVV")
                self.initialize_WMV(config,fixed_hyperparameters_list,hyperparameters_list,debug,param1_scale_im_corrupt,param2_scale_im_corrupt,scaling_input,suffix,global_it,root,scanner)
        
        # else:
        #     self.log("SUCCESS", int(False))