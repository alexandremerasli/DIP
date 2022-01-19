## Python libraries

# Pytorch
from torch.utils.tensorboard import SummaryWriter

# Math
import numpy as np
import matplotlib.pyplot as plt

# Useful
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio

# Local files to import
#from vGeneral import vGeneral
from vDenoising import vDenoising

class iResults(vDenoising):
    def __init__(self,fixed_config,hyperparameters_config,root):
    #def __init__(self,fixed_config,hyperparameters_config,max_iter,PETImage_shape,phantom,subroot,suffix,net,experiment):
        print("__init__")

    def initializeSpecific(self,fixed_config,hyperparameters_config,root):
        # Initialize general variables
        self.initializeGeneralVariables(fixed_config,hyperparameters_config,root)
        vDenoising.initializeSpecific(self,fixed_config,hyperparameters_config,root)

        # Create summary writer from tensorboard
        self.writer = SummaryWriter()
        
        #Loading Ground Truth image to compute metrics
        self.image_gt = self.fijii_np(self.subroot+'Data/database_v2/' + self.phantom + '/' + self.phantom + '.raw',shape=(self.PETImage_shape))
        
        # Metrics arrays
        self.PSNR_recon = np.zeros(self.max_iter)
        self.PSNR_norm_recon = np.zeros(self.max_iter)
        self.MSE_recon = np.zeros(self.max_iter)
        self.MA_cold_recon = np.zeros(self.max_iter)
        self.CRC_hot_recon = np.zeros(self.max_iter)
        self.CRC_bkg_recon = np.zeros(self.max_iter)
        self.IR_bkg_recon = np.zeros(self.max_iter)

    def writeBeginningImages(self,image_net_input,suffix):
        self.write_image_tensorboard(self.writer,image_net_input,"DIP input (FULL CONTRAST)",suffix,self.image_gt,0,full_contrast=True) # DIP input in tensorboard

    def writeCorruptedImage(self,i,max_iter,x_label,suffix,pet_algo,iteration_name='iterations'):
        if (((max_iter>=10) and (i%(max_iter // 10) == 0)) or (max_iter<10)):
            self.write_image_tensorboard(self.writer,x_label,"Corrupted image (x_label) over " + pet_algo + " " + iteration_name,suffix,self.image_gt,i) # Showing all corrupted images with same contrast to compare them together
            self.write_image_tensorboard(self.writer,x_label,"Corrupted image (x_label) over " + pet_algo + " " + iteration_name + " (FULL CONTRAST)",suffix,self.image_gt,i,full_contrast=True) # Showing each corrupted image with contrast = 1

    def writeEndImages(self,subroot,i,max_iter,PETImage_shape,f,suffix,phantom,net,pet_algo,iteration_name='iterations'):
        # Metrics for NN output
        self.compute_metrics(subroot,PETImage_shape,f,self.image_gt,i,self.PSNR_recon,self.PSNR_norm_recon,self.MSE_recon,self.MA_cold_recon,self.CRC_hot_recon,self.CRC_bkg_recon,self.IR_bkg_recon,phantom,writer=self.writer,write_tensorboard=True)

        # Write image over ADMM iterations
        if (((max_iter>=10) and (i%(max_iter // 10) == 0)) or (max_iter<10)):
            self.write_image_tensorboard(self.writer,f,"Image over " + pet_algo + " " + iteration_name + "(" + net + "output)",suffix,self.image_gt,i) # Showing all images with same contrast to compare them together
            self.write_image_tensorboard(self.writer,f,"Image over " + pet_algo + " " + iteration_name + "(" + net + "output, FULL CONTRAST)",suffix,self.image_gt,i,full_contrast=True) # Showing each image with contrast = 1

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

    def runComputation(self,config,fixed_config,hyperparameters_config,root):

        self.writeBeginningImages(self.image_net_input,self.suffix)
        #self.writeCorruptedImage(0,self.max_iter,self.image_corrupt,self.suffix,pet_algo="to fit",iteration_name="(post reconstruction)")


        for i in range(self.max_iter):
            f = self.fijii_np(self.subroot+'Block2/out_cnn/'+ format(self.experiment)+'/out_' + self.net + '' + format(i) + self.suffix + '.img',shape=(self.PETImage_shape)) # loading DIP output

            # Write images over epochs
            self.writeEndImages(self.subroot,i,self.max_iter,self.PETImage_shape,f,self.suffix,self.phantom,self.net,pet_algo="to fit",iteration_name="(post reconstruction)")

    def compute_metrics(self, subroot, PETImage_shape, image_recon,image_gt,i,PSNR_recon,PSNR_norm_recon,MSE_recon,MA_cold_recon,CRC_hot_recon,CRC_bkg_recon,IR_bkg_recon,image,writer=None,write_tensorboard=False):
        # radius - 1 is to remove partial volume effect in metrics computation / radius + 1 must be done on cold and hot ROI when computing backround ROI, because we want to exclude those regions from big cylinder
        
        # Select only phantom ROI, not whole reconstructed image
        print(subroot)
        print(image)
        print(str(image[-1]))
        path_phantom_ROI = subroot+'Data/database_v2/' + image + '/' + "phantom_mask" + str(image[-1]) + '.raw'
        my_file = Path(path_phantom_ROI)
        if (my_file.is_file()):
            phantom_ROI = self.fijii_np(path_phantom_ROI, shape=(PETImage_shape))
        else:
            phantom_ROI = self.fijii_np(subroot+'Data/database_v2/' + image + '/' + "background_mask" + image[-1] + '.raw', shape=(PETImage_shape))
        image_gt_norm = self.norm_imag(image_gt*phantom_ROI)[0]

        # Print metrics
        print('Metrics for iteration',i)

        image_recon_norm = self.norm_imag(image_recon*phantom_ROI)[0] # normalizing DIP output
        print('Dif for PSNR calculation',np.amax(image_recon*phantom_ROI) - np.amin(image_recon*phantom_ROI),' , must be as small as possible')

        # PSNR calculation
        PSNR_recon[i] = peak_signal_noise_ratio(image_gt*phantom_ROI, image_recon*phantom_ROI, data_range=np.amax(image_recon*phantom_ROI) - np.amin(image_recon*phantom_ROI)) # PSNR with true values
        PSNR_norm_recon[i] = peak_signal_noise_ratio(image_gt_norm,image_recon_norm) # PSNR with scaled values [0-1]
        print('PSNR calculation', PSNR_norm_recon[i],' , must be as high as possible')

        # MSE calculation
        MSE_recon[i] = np.mean((image_gt - image_recon)**2)
        print('MSE gt', MSE_recon[i],' , must be as small as possible')
        MSE_recon[i] = np.mean((image_gt*phantom_ROI - image_recon*phantom_ROI)**2)
        print('MSE phantom gt', MSE_recon[i],' , must be as small as possible')

        # Contrast Recovery Coefficient calculation    
        # Mean activity in cold cylinder calculation (-c -40. -40. 0. 40. 4. 0.)
        cold_ROI = self.fijii_np(subroot+'Data/database_v2/' + image + '/' + "cold_mask" + image[-1] + '.raw', shape=(PETImage_shape))
        cold_ROI_act = image_recon[cold_ROI==1]
        MA_cold_recon[i] = np.mean(cold_ROI_act)
        #IR_cold_recon[i] = np.std(cold_ROI_act) / MA_cold_recon[i]
        print('Mean activity in cold cylinder', MA_cold_recon[i],' , must be close to 0')
        #print('Image roughness in the cold cylinder', IR_cold_recon[i])

        # Mean Concentration Recovery coefficient (CRCmean) in hot cylinder calculation (-c 50. 10. 0. 20. 4. 400)
        hot_ROI = self.fijii_np(subroot+'Data/database_v2/' + image + '/' + "tumor_mask" + image[-1] + '.raw', shape=(PETImage_shape))
        hot_ROI_act = image_recon[hot_ROI==1]
        CRC_hot_recon[i] = np.mean(hot_ROI_act) / 400.
        #IR_hot_recon[i] = np.std(hot_ROI_act) / np.mean(hot_ROI_act)
        print('Mean Concentration Recovery coefficient in hot cylinder', CRC_hot_recon[i],' , must be close to 1')
        #print('Image roughness in the hot cylinder', IR_hot_recon[i])

        # Mean Concentration Recovery coefficient (CRCmean) in background calculation (-c 0. 0. 0. 150. 4. 100)
        #m0_bkg = (np.sum(coord_to_value_array(bkg_ROI,image_recon*phantom_ROI)) - np.sum([coord_to_value_array(cold_ROI,image_recon*phantom_ROI),coord_to_value_array(hot_ROI,image_recon*phantom_ROI)])) / (len(bkg_ROI) - (len(cold_ROI) + len(hot_ROI)))
        #CRC_bkg_recon[i] = m0_bkg / 100.
        #         
        bkg_ROI = self.fijii_np(subroot+'Data/database_v2/' + image + '/' + "background_mask" + image[-1] + '.raw', shape=(PETImage_shape))
        bkg_ROI_act = image_recon[bkg_ROI==1]
        CRC_bkg_recon[i] = np.mean(bkg_ROI_act) / 100.
        IR_bkg_recon[i] = np.std(bkg_ROI_act) / np.mean(bkg_ROI_act)
        print('Mean Concentration Recovery coefficient in background', CRC_bkg_recon[i],' , must be close to 1')
        print('Image roughness in the background', IR_bkg_recon[i],' , must be as small as possible')

        if (write_tensorboard):
            print("Metrics saved in tensorboard")
            writer.add_scalars('MSE gt (best : 0)', {'MSE':  MSE_recon[i], 'best': 0,}, i)
            writer.add_scalars('Mean activity in cold cylinder (best : 0)', {'mean_cold':  MA_cold_recon[i], 'best': 0,}, i)
            writer.add_scalars('Mean Concentration Recovery coefficient in hot cylinder (best : 1)', {'CRC_hot':  CRC_hot_recon[i], 'best': 1,}, i)
            writer.add_scalars('Mean Concentration Recovery coefficient in background (best : 1)', {'CRC_bkg':  CRC_bkg_recon[i], 'best': 1,}, i)
            writer.add_scalars('Image roughness in the background (best : 0)', {'IR':  IR_bkg_recon[i], 'best': 0,}, i)
