## Python libraries

# Pytorch
from torch.utils.tensorboard import SummaryWriter

# Math
import numpy as np
import matplotlib.pyplot as plt

# Local files to import
from utils.utils_func import *
from vGeneral import vGeneral

class Results(vGeneral):
    def __init__(self,hyperparameters_config,root,max_iter,PETImage_shape,phantom,subroot):
        print("__init__")

        # Create summary writer from tensorboard
        self.writer = SummaryWriter()
        
        #Loading Ground Truth image to compute metrics
        self.image_gt = fijii_np(subroot+'Data/database_v2/' + phantom + '/' + phantom + '.raw',shape=(PETImage_shape))
        
        # Metrics arrays
        self.PSNR_recon = np.zeros(max_iter)
        self.PSNR_norm_recon = np.zeros(max_iter)
        self.MSE_recon = np.zeros(max_iter)
        self.MA_cold_recon = np.zeros(max_iter)
        self.CRC_hot_recon = np.zeros(max_iter)
        self.CRC_bkg_recon = np.zeros(max_iter)
        self.IR_bkg_recon = np.zeros(max_iter)

        #self.initializeGeneralVariables(hyperparameters_config,root)

    def writeBeginningImages(self,image_net_input,suffix):
        write_image_tensorboard(self.writer,image_net_input,"DIP input (FULL CONTRAST)",suffix,self.image_gt,0,full_contrast=True) # DIP input in tensorboard

    def writeCorruptedImage(self,i,max_iter,x_label,suffix,pet_algo,iteration_name='iterations'):
        if (((max_iter>=10) and (i%(max_iter // 10) == 0)) or (max_iter<10)):
            write_image_tensorboard(self.writer,x_label,"Corrupted image (x_label) over " + pet_algo + " " + iteration_name,suffix,self.image_gt,i) # Showing all corrupted images with same contrast to compare them together
            write_image_tensorboard(self.writer,x_label,"Corrupted image (x_label) over " + pet_algo + " " + iteration_name + " (FULL CONTRAST)",suffix,self.image_gt,i,full_contrast=True) # Showing each corrupted image with contrast = 1

    def writeEndImages(self,i,max_iter,PETImage_shape,f,suffix,phantom,net,pet_algo,iteration_name='iterations'):
        # Metrics for NN output
        compute_metrics(PETImage_shape,f,self.image_gt,i,self.PSNR_recon,self.PSNR_norm_recon,self.MSE_recon,self.MA_cold_recon,self.CRC_hot_recon,self.CRC_bkg_recon,self.IR_bkg_recon,phantom,writer=self.writer,write_tensorboard=True)

        # Write image over ADMM iterations
        if (((max_iter>=10) and (i%(max_iter // 10) == 0)) or (max_iter<10)):
            write_image_tensorboard(self.writer,f,"Image over " + pet_algo + " " + iteration_name + "(" + net + "output)",suffix,self.image_gt,i) # Showing all images with same contrast to compare them together
            write_image_tensorboard(self.writer,f,"Image over " + pet_algo + " " + iteration_name + "(" + net + "output, FULL CONTRAST)",suffix,self.image_gt,i,full_contrast=True) # Showing each image with contrast = 1

        # Display CRC vs STD curve in tensorboard
        if (i>max_iter - min(max_iter,10)):
            # Creating matplotlib figure
            plt.plot(self.IR_bkg_recon,self.CRC_hot_recon,linestyle='None',marker='x')
            plt.xlabel('IR')
            plt.ylabel('CRC')
            # Adding this figure to tensorboard
            self.writer.flush()
            self.writer.add_figure('CRC in hot region vs IR in background', plt.gcf(),global_step=i,close=True)
            self.writer.close()
