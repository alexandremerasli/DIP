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

class iResultsADMMLim_VS_APGMAP(vDenoising):
    def __init__(self,config, *args, **kwargs):
        print("__init__")

    def initializeSpecific(self,config,root, *args, **kwargs):
        # Initialize general variables
        self.initializeGeneralVariables(config,root)
        #'''
        vDenoising.initializeSpecific(self,config,root)

        if ('ADMMLim' in config["method"]):
            try:
                self.path_stopping_criterion = self.subroot + self.suffix + '/' + format(0) + '_adaptive_stopping_criteria.log'
                with open(self.path_stopping_criterion) as f:
                    first_line = f.readline() # Read first line to get second one
                    self.total_nb_iter = min(int(f.readline().rstrip()) - 2, config["nb_outer_iteration"] - 1)
            except:
                self.total_nb_iter = config["nb_outer_iteration"] - 1
            self.beta = config["alpha"]
        elif (config["method"] == 'nested' or config["method"] == 'Gong'):
            if ('post_reco' in config["task"]):
                self.total_nb_iter = config["sub_iter_DIP"]
            else:
                self.total_nb_iter = config["max_iter"]
        else:
            self.total_nb_iter = self.max_iter

            if (config["method"] == 'AML'):
                self.beta = config["A_AML"]
            if (config["method"] == 'BSREM' or config["method"] == 'nested' or config["method"] == 'Gong' or config["method"] == 'APGMAP'):
                self.rho = config["rho"]
                self.beta = self.rho
        # Create summary writer from tensorboard
        self.tensorboard = config["tensorboard"]
        self.writer = SummaryWriter()
        
        #Loading Ground Truth image to compute metrics
        self.image_gt = self.fijii_np(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.raw',shape=(self.PETImage_shape),type='<f')
        if config["FLTNB"] == "double":
            self.image_gt = self.image_gt.astype(np.float64)

        # Defining ROIs
        self.bkg_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "background_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type='<f')
        self.hot_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "tumor_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type='<f')
        self.cold_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "cold_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type='<f')
        self.phantom_ROI = self.get_phantom_ROI(self.phantom)

        # Metrics arrays
        self.PSNR_recon = np.zeros(self.total_nb_iter)
        self.PSNR_norm_recon = np.zeros(self.total_nb_iter)
        self.MSE_recon = np.zeros(self.total_nb_iter)
        self.SSIM_recon = np.zeros(self.total_nb_iter)
        self.MA_cold_recon = np.zeros(self.total_nb_iter)
        self.AR_hot_recon = np.zeros(self.total_nb_iter)
        self.AR_bkg_recon = np.zeros(self.total_nb_iter)
        self.IR_bkg_recon = np.zeros(self.total_nb_iter)
        #'''

    def runComputation(self,config,root):
        
        if (hasattr(self,'beta')):
            beta_string = ', beta = ' + str(self.beta)

        '''
        if (config["method"] == "nested" or config["method"] == "Gong"):
            self.writeBeginningImages(self.suffix,self.image_net_input) # Write GT and DIP input
            #self.writeCorruptedImage(0,self.total_nb_iter,self.image_corrupt,self.suffix,pet_algo="to fit",iteration_name="(post reconstruction)")
        else:
            self.writeBeginningImages(self.suffix) # Write GT
        '''

        if (self.FLTNB == 'float'):
            type = '<f'
        elif (self.FLTNB == 'double'):
            type = '<d'

        f = np.zeros(self.PETImage_shape,dtype=type)
        f_p = np.zeros(self.PETImage_shape,dtype=type)
        f_var = np.zeros(self.PETImage_shape,dtype=type)

        if ('ADMMLim' in config["method"]):
            i_init = 20
        else:
            i_init = 1

        #for i in range(i_init,self.total_nb_iter+1):
        for i in range(self.total_nb_iter,self.total_nb_iter+1):
            IR = 0
            for p in range(self.nb_replicates,0,-1):
                if (config["average_replicates"] or (config["average_replicates"] == False and p == self.replicate)):
                    self.subroot_p = self.subroot_data + 'debug/'*self.debug + '/' + self.phantom + '/' + 'replicate_' + str(p) + '/' + self.method + '/' # Directory root

                    # Take NNEPPS images if NNEPPS is asked for this run
                    if (config["NNEPPS"]):
                        NNEPPS_string = "_NNEPPS"
                    else:
                        NNEPPS_string = ""
                    if (config["method"] == 'Gong' or config["method"] == 'nested'):
                        if ('post_reco' in config["task"]):
                            pet_algo=config["method"]+"to fit"
                            iteration_name="(post reconstruction)"
                        else:
                            pet_algo=config["method"]
                            iteration_name="iterations"
                        if ('post_reco' in config["task"]):
                            try:
                                f_p = self.fijii_np(self.subroot_p+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.net + '' + format(0) + '_epoch=' + format(i-i_init) + NNEPPS_string + '.img',shape=(self.PETImage_shape),type='<f') # loading DIP output
                            except: # ES point is reached
                                break
                        else:
                            f_p = self.fijii_np(self.subroot_p+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.net + '' + format(i-i_init) + "_FINAL" + NNEPPS_string + '.img',shape=(self.PETImage_shape),type='<f') # loading DIP output
                        if config["FLTNB"] == "double":
                            f_p.astype(np.float64)
                    elif ('ADMMLim' in config["method"] or config["method"] == 'MLEM' or config["method"] == 'OPTITR' or config["method"] == 'OSEM' or config["method"] == 'BSREM' or config["method"] == 'AML' or config["method"] == 'APGMAP'):
                        pet_algo=config["method"]
                        iteration_name = "iterations"
                        if (hasattr(self,'beta')):
                            iteration_name += beta_string
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
                            f_p = self.fijii_np(self.subroot_p + self.suffix + '/' +  config["method"] + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output

                    '''
                    # Compute IR metric (different from others with several replicates)
                    self.compute_IR_bkg(self.PETImage_shape,f_p,i-i_init,self.IR_bkg_recon,self.phantom)

                    # Specific average for IR
                    if (config["average_replicates"] == False and p == self.replicate):
                        IR = self.IR_bkg_recon[i-i_init]
                    elif (config["average_replicates"]):
                        IR += self.IR_bkg_recon[i-i_init] / self.nb_replicates
                    '''    
                    if (config["average_replicates"]): # Average images across replicates (for metrics except IR)
                        f += f_p / self.nb_replicates
                    elif (config["average_replicates"] == False and p == self.replicate):
                        f = f_p

                    f_var += (f - f_p)**2 / self.nb_replicates

        # Save images 
        self.write_image_tensorboard(self.writer,f_p,self.method + " at convergence, for replicate 1",self.suffix,self.image_gt,0) # image at convergence in tensorboard
        self.write_image_tensorboard(self.writer,f_p,self.method + " at convergence, for replicate 1 (FULL CONTRAST)",self.suffix,self.image_gt,0,full_contrast=True) # image at convergence in tensorboard
        self.write_image_tensorboard(self.writer,f,self.method + " at convergence, averaged on " + str(self.nb_replicates) + " replicates",self.suffix,self.image_gt,0) # mean of images at convergence across replicates in tensorboard
        self.write_image_tensorboard(self.writer,f,self.method + " at convergence, averaged on " + str(self.nb_replicates) + " replicates (FULL CONTRAST)",self.suffix,self.image_gt,0,full_contrast=True) # mean of images at convergence across replicates in tensorboard
        self.write_image_tensorboard(self.writer,np.sqrt(f_var),self.method + " at convergence, std over " + str(self.nb_replicates) + " replicates",self.suffix,self.image_gt,0) # std of images at convergence across replicates in tensorboard
        self.write_image_tensorboard(self.writer,np.sqrt(f_var),self.method + " at convergence, std over " + str(self.nb_replicates) + " replicates (FULL CONTRAST)",self.suffix,self.image_gt,0,full_contrast=True) # std of images at convergence across replicates in tensorboard
        
    def compareImages(self,suffix):
        if (self.tensorboard):
            self.write_image_tensorboard(self.writer,self.image_method,self.method + " at convergence",suffix,self.image_gt,0,full_contrast=True) # ADMMLim at convergence in tensorboard
            #self.write_image_tensorboard(self.writer,self.image_APGMAP,"APGMAP at convergence",suffix,self.image_gt,0,full_contrast=True) # APGMAP at convergence in tensorboard
       