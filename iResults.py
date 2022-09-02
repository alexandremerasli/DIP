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

# Local files to import
#from vGeneral import vGeneral
from vDenoising import vDenoising

class iResults(vDenoising):
    def __init__(self,config):
        print("__init__")

    def initializeSpecific(self,settings_config,fixed_config,hyperparameters_config,root):
        # Initialize general variables
        self.initializeGeneralVariables(settings_config,fixed_config,hyperparameters_config,root)
        vDenoising.initializeSpecific(self,settings_config,fixed_config,hyperparameters_config,root)

        if ('ADMMLim' in settings_config["method"]):
            try:
                with open(self.path_stopping_criterion) as f:
                    first_line = f.readline()
                    self.total_nb_iter = int(f.readline().rstrip()) - 1
            except:
                self.total_nb_iter = hyperparameters_config["nb_outer_iteration"]
            self.beta = hyperparameters_config["alpha"]
        elif (settings_config["method"] == 'nested' or settings_config["method"] == 'Gong'):
            if (settings_config["task"] == 'show_results_post_reco'):
                self.total_nb_iter = hyperparameters_config["sub_iter_DIP"]
            else:
                self.total_nb_iter = fixed_config["max_iter"]
        else:
            self.total_nb_iter = self.max_iter

            if (settings_config["method"] == 'AML'):
                self.beta = hyperparameters_config["A_AML"]
            if (settings_config["method"] == 'BSREM' or settings_config["method"] == 'nested' or settings_config["method"] == 'Gong' or settings_config["method"] == 'APGMAP'):
                self.rho = hyperparameters_config["rho"]
                self.beta = self.rho
        # Create summary writer from tensorboard
        self.tensorboard = settings_config["tensorboard"]
        self.writer = SummaryWriter()
        
        #Loading Ground Truth image to compute metrics
        self.image_gt = self.fijii_np(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.raw',shape=(self.PETImage_shape),type='<f').astype(np.float64)
        if settings_config["FLTNB"] == "double":
            self.image_gt.astype(np.float64)

        # Defining ROIs
        self.bkg_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "background_mask" + self.phantom[-1] + '.raw', shape=(self.PETImage_shape),type='<f')
        self.hot_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "tumor_mask" + self.phantom[-1] + '.raw', shape=(self.PETImage_shape),type='<f')
        self.cold_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "cold_mask" + self.phantom[-1] + '.raw', shape=(self.PETImage_shape),type='<f')
        self.phantom_ROI = self.get_phantom_ROI()

        # Metrics arrays
        self.PSNR_recon = np.zeros(self.total_nb_iter)
        self.PSNR_norm_recon = np.zeros(self.total_nb_iter)
        self.MSE_recon = np.zeros(self.total_nb_iter)
        self.SSIM_recon = np.zeros(self.total_nb_iter)
        self.MA_cold_recon = np.zeros(self.total_nb_iter)
        self.AR_hot_recon = np.zeros(self.total_nb_iter)
        self.AR_bkg_recon = np.zeros(self.total_nb_iter)
        self.IR_bkg_recon = np.zeros(self.total_nb_iter)
        
    def writeBeginningImages(self,suffix,image_net_input=None):
        if (self.tensorboard):
            self.write_image_tensorboard(self.writer,self.image_gt,"Ground Truth (emission map)",suffix,self.image_gt,0,full_contrast=True) # Ground truth in tensorboard
            if (image_net_input is not None):
                self.write_image_tensorboard(self.writer,image_net_input,"DIP input (FULL CONTRAST)",suffix,image_net_input,0,full_contrast=True) # DIP input in tensorboard

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
        self.compute_metrics(PETImage_shape,f,self.image_gt,i,self.PSNR_recon,self.PSNR_norm_recon,self.MSE_recon,self.SSIM_recon,self.MA_cold_recon,self.AR_hot_recon,self.AR_bkg_recon,self.IR_bkg_recon,phantom,writer=self.writer)

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

    def runComputation(self,config,settings_config,fixed_config,hyperparameters_config,root):
        if (hasattr(self,'beta')):
            beta_string = ', beta = ' + str(self.beta)

        if (settings_config["method"] == "nested" or settings_config["method"] == "Gong"):
            self.writeBeginningImages(self.suffix,self.image_net_input) # Write GT and DIP input
            #self.writeCorruptedImage(0,self.total_nb_iter,self.image_corrupt,self.suffix,pet_algo="to fit",iteration_name="(post reconstruction)")
        else:
            self.writeBeginningImages(self.suffix) # Write GT

        if (self.FLTNB == 'float'):
            type = '<f'
        elif (self.FLTNB == 'double'):
            type = '<d'

        f = np.zeros(self.PETImage_shape,dtype=type)
        f_p = np.zeros(self.PETImage_shape,dtype=type)

        for i in range(1,self.total_nb_iter+1):
            IR = 0
            for p in range(1,self.nb_replicates+1):
                if (settings_config["average_replicates"] or (settings_config["average_replicates"] == False and p == self.replicate)):
                    self.subroot_p = self.subroot_data + 'debug/'*self.debug + '/' + self.phantom + '/' + 'replicate_' + str(p) + '/' + self.method + '/' # Directory root

                    # Take NNEPPS images if NNEPPS is asked for this run
                    if (hyperparameters_config["NNEPPS"]):
                        NNEPPS_string = "_NNEPPS"
                    else:
                        NNEPPS_string = ""
                    if (config["method"] == 'Gong' or config["method"] == 'nested'):
                        if (settings_config["task"] == "show_results_post_reco"):
                            pet_algo=config["method"]+"to fit"
                            iteration_name="(post reconstruction)"
                        else:
                            pet_algo=config["method"]
                            iteration_name="iterations"
                        if (settings_config["task"] == "show_results_post_reco"):
                            try:
                                f_p = self.fijii_np(self.subroot_p+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.net + '' + format(0) + '_epoch=' + format(i-1) + NNEPPS_string + '.img',shape=(self.PETImage_shape),type='<f') # loading DIP output
                            except: # ES point is reached
                                break
                        else:
                            f_p = self.fijii_np(self.subroot_p+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.net + '' + format(i-1) + "FINAL" + NNEPPS_string + '.img',shape=(self.PETImage_shape),type='<f') # loading DIP output
                        f_p.astype(np.float64)
                    elif ('ADMMLim' in config["method"] or config["method"] == 'MLEM' or config["method"] == 'BSREM' or config["method"] == 'AML' or config["method"] == 'APGMAP'):
                        pet_algo=config["method"]
                        iteration_name = "iterations"
                        if (hasattr(self,'beta')):
                            iteration_name += beta_string
                        if ('ADMMLim' in config["method"]):
                            subdir = 'ADMM' + '_' + str(settings_config["nb_threads"])
                            subdir = ''
                            #f_p = self.fijii_np(self.subroot_p + self.suffix + '/' + subdir + '/0_' + format(i) + '_it' + str(fixed_config["nb_inner_iteration"]) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                            #f_p = self.fijii_np(self.subroot_p + self.suffix + '/' + subdir + '/0_' + format(i) + '_it1' + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                            #f_p = self.fijii_np(self.subroot_p + self.suffix + '/' + subdir + '/0_1'  + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                            f_p = self.fijii_np(self.subroot_p + self.suffix + '/' + subdir + '/0'  + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                        #elif (config["method"] == 'BSREM'):
                        #    f_p = self.fijii_np(self.subroot_p + self.suffix + '/' +  config["method"] + '_beta_' + str(self.beta) + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                        else:
                            f_p = self.fijii_np(self.subroot_p + self.suffix + '/' +  config["method"] + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output

                    # Compute IR metric (different from others with several replicates)
                    self.compute_IR_bkg(self.PETImage_shape,f_p,i-1,self.IR_bkg_recon,self.phantom)

                    # Specific average for IR
                    if (settings_config["average_replicates"] == False and p == self.replicate):
                        IR = self.IR_bkg_recon[i-1]
                    elif (settings_config["average_replicates"]):
                        IR += self.IR_bkg_recon[i-1] / self.nb_replicates
                        
                    if (settings_config["average_replicates"]): # Average images across replicates (for metrics except IR)
                        f += f_p / self.nb_replicates
                    elif (settings_config["average_replicates"] == False and p == self.replicate):
                        f = f_p
                
                    del f_p
                    
                
            self.IR_bkg_recon[i-1] = IR
            if (self.tensorboard):
                #print("IR saved in tensorboard")
                self.writer.add_scalar('Image roughness in the background (best : 0)', self.IR_bkg_recon[i-1], i)

            # Show images and metrics in tensorboard (averaged images if asked in settings_config)           
            self.writeEndImagesAndMetrics(i-1,self.total_nb_iter,self.PETImage_shape,f,self.suffix,self.phantom,self.net,pet_algo,iteration_name)

        #self.WMV_plot(fixed_config)

    def WMV_plot(self,fixed_config, hyperparameters_config):

        self.lr = hyperparameters_config['lr']

        # 2.2 plot window moving variance
        plt.figure(1)
        var_x = np.arange(self.windowSize-1, self.windowSize + len(self.VAR_recon)-1)  # define x axis of WMV
        plt.plot(var_x, self.VAR_recon, 'r')
        plt.title('Window Moving Variance,epoch*=' + str(self.epochStar) + ',lr=' + str(self.lr))
        plt.axvline(self.epochStar, c='g')  # plot a vertical line at self.epochStar(detection point)
        plt.xticks([self.epochStar, 0, self.total_nb_iter-1], [self.epochStar, 0, self.total_nb_iter-1], color='green')
        plt.axhline(y=np.min(self.VAR_recon), c="black", linewidth=0.5)
        plt.savefig(self.mkdir(self.subroot + '/self.VAR_recon' + '/w' + str(self.windowSize) + 'p' + str(self.patienceNumber)) + '/' + str(
            self.lr) + '-lr' + str(self.lr) + '+self.VAR_recon-w' + str(self.windowSize) + 'p' + str(self.patienceNumber) + '.png')

        # Save WMV in tensorboard
        #print("WMV saved in tensorboard")
        for i in range(len(self.VAR_recon)):
            var_x = np.arange(self.windowSize-1, self.windowSize + len(self.VAR_recon)-1)  # define x axis of WMV
            self.writer.add_scalar('WMV in the phantom (should follow MSE trend to find peak)', self.VAR_recon[i], var_x[i])

        # 2.3 plot MSE
        plt.figure(2)
        plt.plot(self.MSE_WMV, 'y')
        plt.title('MSE,epoch*=' + str(self.epochStar) + ',lr=' + str(self.lr))
        plt.axvline(self.epochStar, c='g')
        plt.xticks([self.epochStar, 0, self.total_nb_iter-1], [self.epochStar, 0, self.total_nb_iter-1], color='green')
        plt.axhline(y=np.min(self.MSE_WMV), c="black", linewidth=0.5)
        plt.savefig(self.mkdir(self.subroot + '/self.MSE_WMV' + '/w' + str(self.windowSize) + 'p' + str(self.patienceNumber)) + '/' + str(
            self.lr) + '-lr' + str(self.lr) + '+self.MSE_WMV-w' + str(self.windowSize) + 'p' + str(self.patienceNumber) + '.png')

        # 2.4 plot PSNR
        plt.figure(3)
        plt.plot(self.PSNR_WMV)
        plt.title('PSNR,epoch*=' + str(self.epochStar) + ',lr=' + str(self.lr))
        plt.axvline(self.epochStar, c='g')
        plt.xticks([self.epochStar, 0, self.total_nb_iter - 1], [self.epochStar, 0, self.total_nb_iter - 1], color='green')
        plt.axhline(y=np.max(self.PSNR_WMV), c="black", linewidth=0.5)
        plt.savefig(self.mkdir(self.subroot + '/self.PSNR_WMV' + '/w' + str(self.windowSize) + 'p' + str(self.patienceNumber)) + '/' + str(
            self.lr) + '-lr' + str(self.lr) + '+self.PSNR_WMV-w' + str(self.windowSize) + 'p' + str(self.patienceNumber) + '.png')

        #'''
        # 2.5 plot SSIM
        plt.figure(4)
        plt.plot(self.SSIM_WMV, c='orange')
        plt.title('SSIM,epoch*=' + str(self.epochStar) + ',lr=' + str(self.lr))
        plt.axvline(self.epochStar, c='g')
        plt.xticks([self.epochStar, 0, self.total_nb_iter - 1], [self.epochStar, 0, self.total_nb_iter - 1], color='green')
        plt.axhline(y=np.max(self.SSIM_WMV), c="black", linewidth=0.5)
        plt.savefig(self.mkdir(self.subroot + '/self.SSIM_WMV' + '/w' + str(self.windowSize) + 'p' + str(self.patienceNumber)) + '/' + str(
            self.lr) + '-lr' + str(self.lr) + '+self.SSIM_WMV-w' + str(self.windowSize) + 'p' + str(self.patienceNumber) + '.png')
        #'''
        
        # 2.6 plot all the curves together
        fig, ax1 = plt.subplots()
        fig.subplots_adjust(right=0.8, left=0.1, bottom=0.12)
        ax2 = ax1.twinx()  # creat other y-axis for different scale
        ax3 = ax1.twinx()  # creat other y-axis for different scale
        ax4 = ax1.twinx()  # creat other y-axis for different scale
        ax2.spines.right.set_position(("axes", 1.18))
        p4, = ax4.plot(self.MSE_WMV, "y", label="MSE")
        p1, = ax1.plot(self.PSNR_WMV, label="PSNR")
        p2, = ax2.plot(var_x, self.VAR_recon, "r", label="WMV")
        p3, = ax3.plot(self.SSIM_WMV, "orange", label="SSIM")
        ax1.set_xlim(0, self.total_nb_iter-1)
        plt.title('skip : ' + str(hyperparameters_config["skip_connections"]) + ' lr=' + str(self.lr))
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
        ax1.axvline(self.windowSize-1, c='g', linewidth=1, ls=':')
        ax1.axvline(self.epochStar+self.patienceNumber, c='g', lw=1, ls=':')
        if self.epochStar+self.patienceNumber > self.epochStar:
            plt.xticks([self.epochStar, self.windowSize-1, self.epochStar+self.patienceNumber], ['\n' + str(self.epochStar) + '\nES point', str(self.windowSize), '+' + str(self.patienceNumber)], color='green')
        else:
            plt.xticks([self.epochStar, self.windowSize-1], ['\n' + str(self.epochStar) + '\nES point', str(self.windowSize)], color='green')
        plt.savefig(self.mkdir(self.subroot + '/combined/w' + str(self.windowSize) + 'p' + str(self.patienceNumber)) + '/' + str(
            self.lr) + '-lr' + str(self.lr) + '+combined-w' + str(self.windowSize) + 'p' + str(self.patienceNumber) + '.png')

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
        plt.xticks([self.epochStar, 0, self.total_nb_iter - 1], [self.epochStar, 0, self.total_nb_iter - 1], color='green')
        plt.axhline(y=np.max(self.PSNR_WMV), c="black", linewidth=0.5)
        plt.savefig(self.mkdir(self.subroot + '/self.PSNR_WMV' + '/w' + str(self.windowSize) + 'p' + str(self.patienceNumber)) + '/' + str(
            self.lr) + '-lr' + str(self.lr) + '+self.PSNR_WMV-w' + str(self.windowSize) + 'p' + str(self.patienceNumber) + '.png')

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
        #bkg_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + image + '/' + "background_mask" + image[-1] + '.raw', shape=(PETImage_shape),type='<f')
        bkg_ROI_act = image_recon[self.bkg_ROI==1]
        #IR_bkg_recon[i] += (np.std(bkg_ROI_act) / np.mean(bkg_ROI_act)) / self.nb_replicates
        IR_bkg_recon[i] = (np.std(bkg_ROI_act) / np.mean(bkg_ROI_act))
        #print("IR_bkg_recon",IR_bkg_recon)
        #print('Image roughness in the background', IR_bkg_recon[i],' , must be as small as possible')

    def compute_metrics(self, PETImage_shape, image_recon,image_gt,i,PSNR_recon,PSNR_norm_recon,MSE_recon,SSIM_recon,MA_cold_recon,AR_hot_recon,AR_bkg_recon,IR_bkg_recon,image,writer=None):
        # radius - 1 is to remove partial volume effect in metrics computation / radius + 1 must be done on cold and hot ROI when computing background ROI, because we want to exclude those regions from big cylinder
        image_recon_cropped = image_recon*self.phantom_ROI
        image_recon_norm = self.norm_imag(image_recon_cropped)[0] # normalizing DIP output
        image_gt_cropped = image_gt * self.phantom_ROI
        image_gt_norm = self.norm_imag(image_gt_cropped)[0]

        # Print metrics
        print('Metrics for iteration',i)
        #print('Dif for PSNR calculation',np.amax(image_recon_cropped) - np.amin(image_recon_cropped),' , must be as small as possible')

        # PSNR calculation
        PSNR_recon[i] = peak_signal_noise_ratio(image_gt_cropped, image_recon_cropped, data_range=np.amax(image_recon_cropped) - np.amin(image_recon_cropped)) # PSNR with true values
        #print(image_gt_norm.dtype)
        #print(image_recon_norm.dtype)
        PSNR_norm_recon[i] = peak_signal_noise_ratio(image_gt_norm,image_recon_norm) # PSNR with scaled values [0-1]
        #print('PSNR calculation', PSNR_norm_recon[i],' , must be as high as possible')

        # MSE calculation
        MSE_recon[i] = np.mean((image_gt - image_recon)**2)
        #print('MSE gt', MSE_recon[i],' , must be as small as possible')
        MSE_recon[i] = np.mean((image_gt_cropped - image_recon_cropped)**2)
        #print('MSE phantom gt', MSE_recon[i],' , must be as small as possible')
        
        # SSIM calculation
        SSIM_recon[i] = structural_similarity(np.squeeze(image_gt_cropped), np.squeeze(image_recon_cropped), data_range=(image_recon_cropped).max() - (image_recon_cropped).min())
        #print('SSIM calculation', SSIM_recon[i],' , must be close to 1')

        # Contrast Recovery Coefficient calculation    
        # Mean activity in cold cylinder calculation (-c -40. -40. 0. 40. 4. 0.)
        #cold_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + image + '/' + "cold_mask" + image[-1] + '.raw', shape=(PETImage_shape),type='<f')
        cold_ROI_act = image_recon[self.cold_ROI==1]
        MA_cold_recon[i] = np.mean(cold_ROI_act)
        #IR_cold_recon[i] = np.std(cold_ROI_act) / MA_cold_recon[i]
        #print('Mean activity in cold cylinder', MA_cold_recon[i],' , must be close to 0')
        #print('Image roughness in the cold cylinder', IR_cold_recon[i])

        # Mean Activity Recovery (ARmean) in hot cylinder calculation (-c 50. 10. 0. 20. 4. 400)
        #hot_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + image + '/' + "tumor_mask" + image[-1] + '.raw', shape=(PETImage_shape),type='<f')
        hot_ROI_act = image_recon[self.hot_ROI==1]
        AR_hot_recon[i] = np.mean(hot_ROI_act) / 400.
        #IR_hot_recon[i] = np.std(hot_ROI_act) / np.mean(hot_ROI_act)
        #print('Mean Activity Recovery in hot cylinder', AR_hot_recon[i],' , must be close to 1')
        #print('Image roughness in the hot cylinder', IR_hot_recon[i])

        # Mean Activity Recovery (ARmean) in background calculation (-c 0. 0. 0. 150. 4. 100)
        #m0_bkg = (np.sum(coord_to_value_array(bkg_ROI,image_recon_cropped)) - np.sum([coord_to_value_array(cold_ROI,image_recon_cropped),coord_to_value_array(hot_ROI,image_recon_cropped)])) / (len(bkg_ROI) - (len(cold_ROI) + len(hot_ROI)))
        #AR_bkg_recon[i] = m0_bkg / 100.
        #         
        #bkg_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + image + '/' + "background_mask" + image[-1] + '.raw', shape=(PETImage_shape),type='<f')
        bkg_ROI_act = image_recon[self.bkg_ROI==1]
        AR_bkg_recon[i] = np.mean(bkg_ROI_act) / 100.
        #IR_bkg_recon[i] = np.std(bkg_ROI_act) / np.mean(bkg_ROI_act)
        #print('Mean Activity Recovery in background', AR_bkg_recon[i],' , must be close to 1')
        #print('Image roughness in the background', IR_bkg_recon[i],' , must be as small as possible')

        del image_recon_cropped
        del image_gt_cropped

        # Save metrics in csv
        from csv import writer as writer_csv
        Path(self.subroot_metrics + self.method + '/' + self.suffix_metrics).mkdir(parents=True, exist_ok=True) # CASToR path
        with open(self.subroot_metrics + self.method + '/' + self.suffix_metrics + '/metrics.csv', 'w', newline='') as myfile:
            wr = writer_csv(myfile,delimiter=';')
            wr.writerow(PSNR_recon)
            wr.writerow(PSNR_norm_recon)
            wr.writerow(MSE_recon)
            wr.writerow(SSIM_recon)
            wr.writerow(MA_cold_recon)
            wr.writerow(AR_hot_recon)
            wr.writerow(AR_bkg_recon)
            wr.writerow(IR_bkg_recon)

        '''
        print(PSNR_recon)
        print(PSNR_norm_recon)
        print(MSE_recon)
        print(SSIM_recon)
        print(MA_cold_recon)
        print(AR_hot_recon)
        print(AR_bkg_recon)
        print(IR_bkg_recon)
        '''
        
        if (self.tensorboard):
            print("Metrics saved in tensorboard")
            '''
            writer.add_scalars('MSE gt (best : 0)', {'MSE':  MSE_recon[i], 'best': 0,}, i)
            writer.add_scalars('Mean activity in cold cylinder (best : 0)', {'mean_cold':  MA_cold_recon[i], 'best': 0,}, i)
            writer.add_scalars('Mean Concentration Recovery coefficient in hot cylinder (best : 1)', {'AR_hot':  AR_hot_recon[i], 'best': 1,}, i)
            writer.add_scalars('Mean Concentration Recovery coefficient in background (best : 1)', {'MA_bkg':  AR_bkg_recon[i], 'best': 1,}, i)
            #writer.add_scalars('Image roughness in the background (best : 0)', {'IR':  IR_bkg_recon[i], 'best': 0,}, i)
            '''
            writer.add_scalar('MSE gt (best : 0)', MSE_recon[i], i)
            writer.add_scalar('SSIM gt (best : 0)', SSIM_recon[i], i)
            writer.add_scalar('Mean activity in cold cylinder (best : 0)', MA_cold_recon[i], i)
            writer.add_scalar('Mean Concentration Recovery coefficient in hot cylinder (best : 1)', AR_hot_recon[i], i)
            writer.add_scalar('Mean Concentration Recovery coefficient in background (best : 1)', AR_bkg_recon[i], i)
            #writer.add_scalar('Image roughness in the background (best : 0)', IR_bkg_recon[i], i)