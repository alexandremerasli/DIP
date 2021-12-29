## Python libraries

# Pytorch
from ray.tune import analysis
import torch
from torch.utils.tensorboard import SummaryWriter

# Useful
import os
from pathlib import Path
import argparse
import time
import subprocess
from functools import partial
from ray import tune

# Math
import numpy as np
import matplotlib.pyplot as plt

# Local files to import
from utils.utils_func import *
from vReconstruction import vReconstruction
from iDenoisingInReconstruction import iDenoisingInReconstruction

class iNestedADMM(vReconstruction):
    def __init__(self,config,args,root):
        vReconstruction.__init__(self,config,args,root)
    def runReconstruction(self,config,args,root):
        print("Nested ADMM reconstruction")

        f = self.f_init  # Initializing DIP output with f_init
        for i in range(self.max_iter):
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Outer iteration !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', i)
            start_time_outer_iter = time.time()
            
            # Reconstruction with CASToR (first equation of ADMM)
            if (config["method"] == 'Gong'):
                subroot_output_path = self.subroot + 'Block1/' + self.suffix # + '/' # Output path for CASTOR framework
                input_path = ' -img ' + self.subroot + 'Block1/' + self.suffix + '/out_eq22/' # Input path for CASTOR framework
                x_label = castor_reconstruction_OPTITR(i, self.subroot, self.sub_iter_MAP, self.test, subroot_output_path, input_path, config, self.suffix, f, self.mu, self.PETImage_shape, self.image_init_path_without_extension)
            else: # Nested ADMM
                x_label = castor_reconstruction(self.writer, i, self.subroot, self.sub_iter_MAP, self.test, config, self.suffix, self.image_gt, f, self.mu, self.PETImage_shape, self.PETImage_shape_str, self.rho, self.alpha, self.image_init_path_without_extension) # without ADMMLim file

            # Write image over ADMM iterations
            if ((self.max_iter>=10) and (i%(self.max_iter // 10) == 0) or True):

                write_image_tensorboard(self.writer,x_label,"Corrupted image (x_label) over ADMM iterations",self.suffix,self.image_gt,i) # Showing all corrupted images with same contrast to compare them together
                write_image_tensorboard(self.writer,x_label,"Corrupted image (x_label) over ADMM iterations (FULL CONTRAST)",self.suffix,self.image_gt,i,full_contrast=True) # Showing each corrupted image with contrast = 1
            
            # Block 2 - CNN - 10 iterations
            start_time_block2= time.time()
            
            
            classDenoising = iDenoisingInReconstruction(config,args,root,i)
            classDenoising.do_everything(config,args,root)
            #classDenoising.initializeSpecific(config,args,root)
            #classDenoising.runDenoiser()
            
            f = fijii_np(self.subroot+'Block2/out_cnn/'+ format(self.test)+'/out_' + self.net + '' + format(i) + self.suffix + '.img',shape=(self.PETImage_shape)) # loading DIP output

            # Metrics for NN output
            compute_metrics(self.PETImage_shape,f,self.image_gt,i,self.PSNR_recon,self.PSNR_norm_recon,self.MSE_recon,self.MA_cold_recon,self.CRC_hot_recon,self.CRC_bkg_recon,self.IR_bkg_recon,self.phantom,writer=self.writer,write_tensorboard=True)

            # Block 3 - equation 15 - mu
            self.mu = x_label- f
            save_img(self.mu,self.subroot+'Block2/mu/'+ format(self.test)+'/mu_' + format(i) + self.suffix + '.img') # saving mu

            write_image_tensorboard(self.writer,self.mu,"mu(FULL CONTRAST)",self.suffix,self.image_gt,i,full_contrast=True) # Showing all corrupted images with same contrast to compare them together
            print("--- %s seconds - outer_iteration ---" % (time.time() - start_time_outer_iter))

            # Write image over ADMM iterations
            if ((self.max_iter>=10) and (i%(self.max_iter // 10) == 0) or (self.max_iter<10)):
                write_image_tensorboard(self.writer,f,"Image over ADMM iterations (" + self.net + "output)",self.suffix,self.image_gt,i) # Showing all images with same contrast to compare them together
                write_image_tensorboard(self.writer,f,"Image over ADMM iterations (" + self.net + "output, FULL CONTRAST)",self.suffix,self.image_gt,i,full_contrast=True) # Showing each image with contrast = 1
            
            # Display CRC vs STD curve in tensorboard
            if (i>self.max_iter - min(self.max_iter,10)):
                # Creating matplotlib figure
                plt.plot(self.IR_bkg_recon,self.CRC_hot_recon,linestyle='None',marker='x')
                plt.xlabel('IR')
                plt.ylabel('CRC')
                # Adding this figure to tensorboard
                self.writer.flush()
                self.writer.add_figure('CRC in hot region vs IR in background', plt.gcf(),global_step=i,close=True)
                self.writer.close()


        """
        Output framework
        """

        # Output of the framework
        self.x_out = f

        # Saving final image output
        save_img(self.x_out, self.subroot+'Images/out_final/final_out' + self.suffix + '.img')
        '''
        #Plot and save output of the framework
        plt.figure()
        plt.plot(STD_recon,TCR_Recon,'--ro', label='Recon Algorithm')
        plt.title('STD vs CRC')
        plt.legend(loc='lower right')
        plt.xlabel('STD')
        plt.ylabel('CRC')
        plt.savefig(self.subroot+'Images/metrics/'+format(self.test)+'/' + format(i) +'-wo-pre.png')
        

        # Display and saved.
        plt.figure()
        plt.imshow(x_out, cmap='gray_r')
        plt.colorbar()
        plt.title('Reconstructed image: %d ' % (i))
        plt.savefig(self.subroot+'Images/out_final/'+format(self.test)+'/' + format(self.test) +'.png')
        '''
        ## Averaging for VAE
        if (self.net == 'DIP_VAE'):
            print('Computing average over VAE ouputs')
            ## Initialize variables
            # Number of posterior samples to use in mean and variance computation
            n_posterior_samples = min(10,self.max_iter)
            print("Number of posterior samples :",n_posterior_samples, '(over', self.max_iter, 'overall iterations)')
            # Averaged and uncertainty images
            x_avg = np.zeros((self.PETImage_shape[0], self.PETImage_shape[1]), dtype='<f') # averaged image
            x_var = np.zeros((self.PETImage_shape[0], self.PETImage_shape[1]), dtype='<f') # uncertainty image

            list_samples = [] # list to store averaged and uncertainty images

            # Loading DIP input (we do not have CT-map, so random image created in block 1)
            self.image_net_input_torch = torch.load(self.subroot + 'Data/initialization/image_' + self.net + '_input_torch.pt') # DO NOT CREATE RANDOM INPUT IN BLOCK 2 !!! ONLY AT THE BEGINNING, IN BLOCK 1. JUST LOAD IT FROM BLOCK 1

            for i in range(n_posterior_samples):
                # Generate one descaled NN output
                out_descale = generate_nn_output(self.net, config, self.image_net_input_torch, self.PETImage_shape, self.finetuning, self.max_iter, self.test, self.suffix, self.subroot)
                list_samples.append(np.squeeze(out_descale))
                
            for i in range(n_posterior_samples):
                x_avg += list_samples[i] / n_posterior_samples
                x_var = (list_samples[i] - x_avg)**2 / n_posterior_samples
            
            # Computing metrics to compare averaging vae outputs with single output
            compute_metrics(x_avg,self.image_gt,i,self.PSNR_recon,self.PSNR_norm_recon,self.MSE_recon,self.MA_cold_recon,self.CRC_hot_recon,self.CRC_bkg_recon,self.IR_bkg_recon,self.phantom,write_tensorboard=False)

        # Display images in tensorboard
        #write_image_tensorboard(self.writer,f_init,"initialization of DIP output (f_0) (FULL CONTRAST)",self.suffix,self.image_gt,0,full_contrast=True) # initialization of DIP output in tensorboard
        #write_image_tensorboard(self.writer,image_init,"initialization of CASToR MAP reconstruction (x_0)",self.suffix,,self.image_gt) # initialization of CASToR MAP reconstruction in tensorboard
        write_image_tensorboard(self.writer,self.image_net_input,"DIP input (FULL CONTRAST)",self.suffix,self.image_gt,0,full_contrast=True) # DIP input in tensorboard
        write_image_tensorboard(self.writer,self.image_gt,"Ground Truth",self.suffix,self.image_gt) # Ground truth image in tensorboard

        if (self.net == 'DIP_VAE'):
            write_image_tensorboard(self.writer,x_avg,"Final averaged image (average over DIP outputs)",self.suffix,self.image_gt) # Final averaged image in tensorboard
            write_image_tensorboard(self.writer,x_var,"Uncertainty image (VARIANCE over DIP outputs)",self.suffix,self.image_gt) # Uncertainty image in tensorboard
