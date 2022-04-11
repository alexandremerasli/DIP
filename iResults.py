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
    def __init__(self,config):
        print("__init__")

    def initializeSpecific(self,fixed_config,hyperparameters_config,root):
        # Initialize general variables
        self.initializeGeneralVariables(fixed_config,hyperparameters_config,root)
        vDenoising.initializeSpecific(self,fixed_config,hyperparameters_config,root)
        
        if (fixed_config["method"] == 'ADMMLim'):
            self.total_nb_iter = hyperparameters_config["nb_iter_second_admm"]
            self.beta = hyperparameters_config["alpha"]
        else:
            self.total_nb_iter = self.max_iter

            if (fixed_config["method"] == 'AML'):
                self.beta = hyperparameters_config["A_AML"]
            else:                
                self.beta = self.rho
        # Create summary writer from tensorboard
        self.writer = SummaryWriter()
        
        #Loading Ground Truth image to compute metrics
        self.image_gt = self.fijii_np(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.raw',shape=(self.PETImage_shape),type='<f')
        
        # Metrics arrays
        self.PSNR_recon = np.zeros(self.total_nb_iter)
        self.PSNR_norm_recon = np.zeros(self.total_nb_iter)
        self.MSE_recon = np.zeros(self.total_nb_iter)
        self.MA_cold_recon = np.zeros(self.total_nb_iter)
        self.AR_hot_recon = np.zeros(self.total_nb_iter)
        self.AR_bkg_recon = np.zeros(self.total_nb_iter)
        self.IR_bkg_recon = np.zeros(self.total_nb_iter)
        
    def writeBeginningImages(self,image_net_input,suffix):
        self.write_image_tensorboard(self.writer,image_net_input,"DIP input (FULL CONTRAST)",suffix,self.image_gt,0,full_contrast=True) # DIP input in tensorboard

    def writeCorruptedImage(self,i,max_iter,x_label,suffix,pet_algo,iteration_name='iterations'):
        if (((max_iter>=10) and (i%(max_iter // 10) == 0)) or (max_iter<10)):
            self.write_image_tensorboard(self.writer,x_label,"Corrupted image (x_label) over " + pet_algo + " " + iteration_name,suffix,self.image_gt,i) # Showing all corrupted images with same contrast to compare them together
            self.write_image_tensorboard(self.writer,x_label,"Corrupted image (x_label) over " + pet_algo + " " + iteration_name + " (FULL CONTRAST)",suffix,self.image_gt,i,full_contrast=True) # Showing each corrupted image with contrast = 1

    def writeEndImagesAndMetrics(self,i,max_iter,PETImage_shape,f,suffix,phantom,net,pet_algo,iteration_name='iterations'):
        # Metrics for NN output
        self.compute_metrics(PETImage_shape,f,self.image_gt,i,self.PSNR_recon,self.PSNR_norm_recon,self.MSE_recon,self.MA_cold_recon,self.AR_hot_recon,self.AR_bkg_recon,self.IR_bkg_recon,phantom,writer=self.writer,write_tensorboard=True)

        # Write image over ADMM iterations
        if (((max_iter>=10) and (i%(max_iter // 10) == 0)) or (max_iter<10)):
            self.write_image_tensorboard(self.writer,f,"Image over " + pet_algo + " " + iteration_name + "(" + net + "output)",suffix,self.image_gt,i) # Showing all images with same contrast to compare them together
            self.write_image_tensorboard(self.writer,f,"Image over " + pet_algo + " " + iteration_name + "(" + net + "output, FULL CONTRAST)",suffix,self.image_gt,i,full_contrast=True) # Showing each image with contrast = 1

        # Display AR (hot) /MA (cold) vs STD curve in tensorboard
        if (i == self.total_nb_iter):
            # Creating matplotlib figure
            plt.plot(self.IR_bkg_recon,self.AR_hot_recon,linestyle='None',marker='x')
            plt.xlabel('IR')
            plt.ylabel('AR')
            # Saving this figure locally
            Path(self.subroot + 'Images/tmp/' + suffix).mkdir(parents=True, exist_ok=True)
            #os.system('rm -rf' + self.subroot + 'Images/tmp/' + suffix + '/*')
            print(self.subroot + 'Images/tmp/' + suffix + '/' + 'AR in hot region vs IR in background' + '_' + str(i) + '.png')
            plt.savefig(self.subroot + 'Images/tmp/' + suffix + '/' + 'AR in hot region vs IR in background' + '_' + str(i) + '.png')
            from textwrap import wrap
            wrapped_title = "\n".join(wrap(suffix, 50))
            plt.title(wrapped_title,fontsize=12)

            # Adding this figure to tensorboard
            self.writer.flush()
            self.writer.add_figure('AR in hot region vs IR in background', plt.gcf(),global_step=i,close=True)
            self.writer.close()


            # Creating matplotlib figure
            plt.plot(self.IR_bkg_recon,self.MA_cold_recon,linestyle='None',marker='x')
            plt.xlabel('IR')
            plt.ylabel('MA')
            # Saving this figure locally
            Path(self.subroot + 'Images/tmp/' + suffix).mkdir(parents=True, exist_ok=True)
            #os.system('rm -rf' + self.subroot + 'Images/tmp/' + suffix + '/*')
            print(self.subroot + 'Images/tmp/' + suffix + '/' + 'MA in cold region vs IR in background' + '_' + str(i) + '.png')
            plt.savefig(self.subroot + 'Images/tmp/' + suffix + '/' + 'MA in cold region vs IR in background' + '_' + str(i) + '.png')
            from textwrap import wrap
            wrapped_title = "\n".join(wrap(suffix, 50))
            plt.title(wrapped_title,fontsize=12)

            # Adding this figure to tensorboard
            self.writer.flush()
            self.writer.add_figure('MA in cold region vs IR in background', plt.gcf(),global_step=i,close=True)
            self.writer.close()



    def runComputation(self,config,fixed_config,hyperparameters_config,root):
        beta_string = ', beta = ' + str(self.beta)

        if (fixed_config["method"] == "nested" or fixed_config["method"] == "Gong"):
            self.writeBeginningImages(self.image_net_input,self.suffix)
            #self.writeCorruptedImage(0,self.total_nb_iter,self.image_corrupt,self.suffix,pet_algo="to fit",iteration_name="(post reconstruction)")

        for i in range(1,self.total_nb_iter+1):
            print(i)

            f = np.zeros(self.PETImage_shape,dtype='<f')
            for p in range(1,self.nb_replicates+1):
                self.subroot_p = self.subroot_data + 'debug/'*self.debug + 'replicate_' + str(p) + '/'

                # Take NNEPPS images if NNEPPS is asked for this run
                if (hyperparameters_config["NNEPPS"]):
                    NNEPPS_string = "_NNEPPS"
                else:
                    NNEPPS_string = ""
                if (config["method"] == 'Gong' or config["method"] == 'nested'):
                    pet_algo=config["method"]+"to fit"
                    iteration_name="(post reconstruction)"
                    f_p = self.fijii_np(self.subroot_p+'Block2/out_cnn/'+ format(self.experiment)+'/out_' + self.net + '' + format(i) + self.suffix + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading DIP output
                elif (config["method"] == 'ADMMLim' or config["method"] == 'MLEM' or config["method"] == 'BSREM' or config["method"] == 'AML'):
                    pet_algo=config["method"]
                    iteration_name="iterations"+beta_string
                    if (config["method"] == 'ADMMLim'):
                        subdir = 'ADMM' + '_' + str(fixed_config["nb_threads"])
                        f_p = self.fijii_np(self.subroot_p+'Comparison/' + config["method"] + '/' + self.suffix + '/' + subdir + '/0_' + format(i) + '_it' + str(hyperparameters_config["sub_iter_MAP"]) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                    #elif (config["method"] == 'BSREM'):
                    #    f_p = self.fijii_np(self.subroot_p+'Comparison/' + config["method"] + '/' + self.suffix + '/' +  config["method"] + '_beta_' + str(self.beta) + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                    else:
                        f_p = self.fijii_np(self.subroot_p+'Comparison/' + config["method"] + '/' + self.suffix + '/' +  config["method"] + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                print(self.subroot_p+'Comparison/' + config["method"] + '_beta_' + str(self.beta) + '/' +  config["method"] + '_beta_' + str(self.beta) + '_it' + format(i) + NNEPPS_string + '.img')
                f += f_p
                # Metrics for NN output 
                self.compute_IR_bkg(self.PETImage_shape,f_p,self.image_gt,i,self.PSNR_recon,self.PSNR_norm_recon,self.MSE_recon,self.MA_cold_recon,self.AR_hot_recon,self.AR_bkg_recon,self.IR_bkg_recon,self.phantom,writer=self.writer,write_tensorboard=True)
    
            print("Metrics saved in tensorboard")
            self.writer.add_scalar('Image roughness in the background (best : 0)', self.IR_bkg_recon[i-1], i)

            # Compute metrics after averaging images across replicates
            f = f / self.nb_replicates
            self.writeEndImagesAndMetrics(i,self.total_nb_iter,self.PETImage_shape,f,self.suffix,self.phantom,self.net,pet_algo,iteration_name)


    def compute_IR_bkg(self, PETImage_shape, image_recon,image_gt,i,PSNR_recon,PSNR_norm_recon,MSE_recon,MA_cold_recon,AR_hot_recon,AR_bkg_recon,IR_bkg_recon,image,writer=None,write_tensorboard=False):
        # radius - 1 is to remove partial volume effect in metrics computation / radius + 1 must be done on cold and hot ROI when computing background ROI, because we want to exclude those regions from big cylinder
        
        # Select only phantom ROI, not whole reconstructed image
        path_phantom_ROI = self.subroot_data+'Data/database_v2/' + image + '/' + "phantom_mask" + str(image[-1]) + '.raw'
        my_file = Path(path_phantom_ROI)
        print(path_phantom_ROI)
        if (my_file.is_file()):
            phantom_ROI = self.fijii_np(path_phantom_ROI, shape=(PETImage_shape),type='<f')
        else:
            phantom_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + image + '/' + "background_mask" + image[-1] + '.raw', shape=(PETImage_shape),type='<f')

              
        bkg_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + image + '/' + "background_mask" + image[-1] + '.raw', shape=(PETImage_shape),type='<f')
        bkg_ROI_act = image_recon[bkg_ROI==1]
        IR_bkg_recon[i-1] += (np.std(bkg_ROI_act) / np.mean(bkg_ROI_act)) / self.nb_replicates
        print("IR_bkg_recon",IR_bkg_recon)
        print('Image roughness in the background', IR_bkg_recon[i-1],' , must be as small as possible')

    def compute_metrics(self, PETImage_shape, image_recon,image_gt,i,PSNR_recon,PSNR_norm_recon,MSE_recon,MA_cold_recon,AR_hot_recon,AR_bkg_recon,IR_bkg_recon,image,writer=None,write_tensorboard=False):
        # radius - 1 is to remove partial volume effect in metrics computation / radius + 1 must be done on cold and hot ROI when computing background ROI, because we want to exclude those regions from big cylinder
        
        # Select only phantom ROI, not whole reconstructed image
        path_phantom_ROI = self.subroot_data+'Data/database_v2/' + image + '/' + "phantom_mask" + str(image[-1]) + '.raw'
        my_file = Path(path_phantom_ROI)
        if (my_file.is_file()):
            phantom_ROI = self.fijii_np(path_phantom_ROI, shape=(PETImage_shape),type='<f')
        else:
            phantom_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + image + '/' + "background_mask" + image[-1] + '.raw', shape=(PETImage_shape),type='<f')
        image_gt_norm = self.norm_imag(image_gt*phantom_ROI)[0]

        # Print metrics
        print('Metrics for iteration',i)

        image_recon_norm = self.norm_imag(image_recon*phantom_ROI)[0] # normalizing DIP output
        print('Dif for PSNR calculation',np.amax(image_recon*phantom_ROI) - np.amin(image_recon*phantom_ROI),' , must be as small as possible')

        # PSNR calculation
        PSNR_recon[i-1] = peak_signal_noise_ratio(image_gt*phantom_ROI, image_recon*phantom_ROI, data_range=np.amax(image_recon*phantom_ROI) - np.amin(image_recon*phantom_ROI)) # PSNR with true values
        PSNR_norm_recon[i-1] = peak_signal_noise_ratio(image_gt_norm,image_recon_norm) # PSNR with scaled values [0-1]
        print('PSNR calculation', PSNR_norm_recon[i-1],' , must be as high as possible')

        # MSE calculation
        MSE_recon[i-1] = np.mean((image_gt - image_recon)**2)
        print('MSE gt', MSE_recon[i-1],' , must be as small as possible')
        MSE_recon[i-1] = np.mean((image_gt*phantom_ROI - image_recon*phantom_ROI)**2)
        print('MSE phantom gt', MSE_recon[i-1],' , must be as small as possible')

        # Contrast Recovery Coefficient calculation    
        # Mean activity in cold cylinder calculation (-c -40. -40. 0. 40. 4. 0.)
        cold_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + image + '/' + "cold_mask" + image[-1] + '.raw', shape=(PETImage_shape),type='<f')
        cold_ROI_act = image_recon[cold_ROI==1]
        MA_cold_recon[i-1] = np.mean(cold_ROI_act)
        #IR_cold_recon[i-1] = np.std(cold_ROI_act) / MA_cold_recon[i-1]
        print('Mean activity in cold cylinder', MA_cold_recon[i-1],' , must be close to 0')
        #print('Image roughness in the cold cylinder', IR_cold_recon[i-1])

        # Mean Activity Recovery (ARmean) in hot cylinder calculation (-c 50. 10. 0. 20. 4. 400)
        print(self.subroot_data+'Data/database_v2/' + image + '/' + "tumor_mask" + image[-1] + '.raw')
        hot_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + image + '/' + "tumor_mask" + image[-1] + '.raw', shape=(PETImage_shape),type='<f')
        hot_ROI_act = image_recon[hot_ROI==1]
        AR_hot_recon[i-1] = np.mean(hot_ROI_act) / 400.
        #IR_hot_recon[i-1] = np.std(hot_ROI_act) / np.mean(hot_ROI_act)
        print('Mean Activity Recovery in hot cylinder', AR_hot_recon[i-1],' , must be close to 1')
        #print('Image roughness in the hot cylinder', IR_hot_recon[i-1])

        # Mean Activity Recovery (ARmean) in background calculation (-c 0. 0. 0. 150. 4. 100)
        #m0_bkg = (np.sum(coord_to_value_array(bkg_ROI,image_recon*phantom_ROI)) - np.sum([coord_to_value_array(cold_ROI,image_recon*phantom_ROI),coord_to_value_array(hot_ROI,image_recon*phantom_ROI)])) / (len(bkg_ROI) - (len(cold_ROI) + len(hot_ROI)))
        #AR_bkg_recon[i-1] = m0_bkg / 100.
        #         
        bkg_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + image + '/' + "background_mask" + image[-1] + '.raw', shape=(PETImage_shape),type='<f')
        bkg_ROI_act = image_recon[bkg_ROI==1]
        AR_bkg_recon[i-1] = np.mean(bkg_ROI_act) / 100.
        #IR_bkg_recon[i-1] = np.std(bkg_ROI_act) / np.mean(bkg_ROI_act)
        print('Mean Activity Recovery in background', AR_bkg_recon[i-1],' , must be close to 1')
        #print('Image roughness in the background', IR_bkg_recon[i-1],' , must be as small as possible')

        # Save metrics in csv
        from csv import writer as writer_csv
        with open(self.subroot_data + 'metrics/' + self.method + '/' + self.suffix_metrics + '/metrics.csv', 'w', newline='') as myfile:
            wr = writer_csv(myfile,delimiter=';')
            wr.writerow(PSNR_recon)
            wr.writerow(PSNR_norm_recon)
            wr.writerow(MSE_recon)
            wr.writerow(MA_cold_recon)
            wr.writerow(AR_hot_recon)
            wr.writerow(AR_bkg_recon)
            wr.writerow(IR_bkg_recon)

        print(PSNR_recon)
        print(PSNR_norm_recon)
        print(MSE_recon)
        print(MA_cold_recon)
        print(AR_hot_recon)
        print(AR_bkg_recon)
        print(IR_bkg_recon)

        if (write_tensorboard):
            print("Metrics saved in tensorboard")
            '''
            writer.add_scalars('MSE gt (best : 0)', {'MSE':  MSE_recon[i-1], 'best': 0,}, i)
            writer.add_scalars('Mean activity in cold cylinder (best : 0)', {'mean_cold':  MA_cold_recon[i-1], 'best': 0,}, i)
            writer.add_scalars('Mean Concentration Recovery coefficient in hot cylinder (best : 1)', {'AR_hot':  AR_hot_recon[i-1], 'best': 1,}, i)
            writer.add_scalars('Mean Concentration Recovery coefficient in background (best : 1)', {'MA_bkg':  AR_bkg_recon[i-1], 'best': 1,}, i)
            #writer.add_scalars('Image roughness in the background (best : 0)', {'IR':  IR_bkg_recon[i-1], 'best': 0,}, i)
            '''
            writer.add_scalar('MSE gt (best : 0)', MSE_recon[i-1], i)
            writer.add_scalar('Mean activity in cold cylinder (best : 0)', MA_cold_recon[i-1], i)
            writer.add_scalar('Mean Concentration Recovery coefficient in hot cylinder (best : 1)', AR_hot_recon[i-1], i)
            writer.add_scalar('Mean Concentration Recovery coefficient in background (best : 1)', AR_bkg_recon[i-1], i)
            writer.add_scalar('Image roughness in the background (best : 0)', IR_bkg_recon[i-1], i)