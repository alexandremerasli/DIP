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
        #vDenoising.initializeSpecific(self,config,root)

        if ('ADMMLim' in config["method"]):
            self.i_init = 30 # Remove first iterations
            self.i_init = 1 # Remove first iterations
        else:
            self.i_init = 1

        self.defineTotalNbIter_beta_rho(config["method"], config, config["task"])

        # Create summary writer from tensorboard
        self.tensorboard = config["tensorboard"]
        self.writer = SummaryWriter()
        
        #Loading Ground Truth image to compute metrics
        self.image_gt = self.fijii_np(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.raw',shape=(self.PETImage_shape),type_im='<f')
        if config["FLTNB"] == "double":
            self.image_gt = self.image_gt.astype(np.float64)

        # Defining ROIs
        self.bkg_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "background_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
        self.phantom_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "phantom_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
        self.phantom_ROI = (self.phantom_ROI).astype(np.int)
        # Metrics arrays
        '''
        self.PSNR_recon = np.zeros(self.total_nb_iter)
        self.PSNR_norm_recon = np.zeros(self.total_nb_iter)
        self.MSE_recon = np.zeros(self.total_nb_iter)
        self.SSIM_recon = np.zeros(self.total_nb_iter)
        self.MA_cold_recon = np.zeros(self.total_nb_iter)
        self.AR_hot_recon = np.zeros(self.total_nb_iter)
        self.AR_bkg_recon = np.zeros(self.total_nb_iter)
        '''
        self.IR_bkg_recon = np.zeros(self.total_nb_iter)


    def compute_IR_bkg(self, PETImage_shape, image_recon,i,IR_bkg_recon,image):
        # radius - 1 is to remove partial volume effect in metrics computation / radius + 1 must be done on cold and hot ROI when computing background ROI, because we want to exclude those regions from big cylinder
        bkg_ROI_act = image_recon[self.bkg_ROI==1]
        IR_bkg_recon[i] = (np.std(bkg_ROI_act) / np.mean(bkg_ROI_act))
        #print("IR_bkg_recon",IR_bkg_recon)
        #print('Image roughness in the background', IR_bkg_recon[i],' , must be as small as possible')


    def runComputation(self,config,root):
        
        config["average_replicates"] = True




        

        if (self.replicate != 1):
            print(self.replicate)
            return 0
            raise ValueError("Only computing for 1st replicate")

        if (hasattr(self,'beta')):
            beta_string = ', beta = ' + str(self.beta)
        else:
            beta_string = ""

        '''
        if ( 'nested' in config["method"] or  'Gong' in config["method"]):
            self.writeBeginningImages(self.suffix,self.image_net_input) # Write GT and DIP input
            #self.writeCorruptedImage(0,self.total_nb_iter,self.image_corrupt,self.suffix,pet_algo="to fit",iteration_name="(post reconstruction)")
        else:
            self.writeBeginningImages(self.suffix) # Write GT
        '''

        if (self.FLTNB == 'float'):
            type_im = '<f'
        elif (self.FLTNB == 'double'):
            type_im = '<d'

        f = np.zeros(self.PETImage_shape,dtype=type_im)
        self.f_p = np.zeros(self.PETImage_shape,dtype=type_im)
        f_var = np.zeros(self.PETImage_shape,dtype=type_im)

        f_init_p = np.zeros(self.PETImage_shape,dtype=type_im)
        f_init_avg = np.zeros(self.PETImage_shape,dtype=type_im)

        f_list = self.nb_replicates * [0]

        if ('ADMMLim' in config["method"]):
            i_init = 20
        else:
            i_init = 1

        #for i in range(i_init,self.total_nb_iter+1):
        for i in range(self.total_nb_iter,self.total_nb_iter+1):
            IR = 0
            nan_replicates = []
            for p in range(self.nb_replicates,0,-1):
                if (config["average_replicates"] or (config["average_replicates"] == False and p == self.replicate)):
                    self.subroot_p = self.subroot_data + 'debug/'*self.debug + '/' + self.phantom + '/' + 'replicate_' + str(p) + '/' + self.method + '/' # Directory root

                    # Take NNEPPS images if NNEPPS is asked for this run
                    if (config["NNEPPS"]):
                        NNEPPS_string = "_NNEPPS"
                    else:
                        NNEPPS_string = ""
                    if ( 'Gong' in config["method"] or  'nested' in config["method"]):
                        if ('post_reco' in config["task"]):
                            pet_algo=config["method"]+"to fit"
                            iteration_name="(post reconstruction)"
                        else:
                            pet_algo=config["method"]
                            iteration_name="iterations"
                        if ('post_reco' in config["task"]):
                            try:
                                self.f_p = self.fijii_np(self.subroot_p+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.net + '' + format(0) + '_epoch=' + format(i-i_init) + NNEPPS_string + '.img',shape=(self.PETImage_shape),type_im='<f') # loading DIP output
                            except: # ES point is reached
                                break
                        else:
                            self.f_p = self.fijii_np(self.subroot_p+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.net + '' + format(i-i_init) + "_FINAL" + NNEPPS_string + '.img',shape=(self.PETImage_shape),type_im='<f') # loading DIP output
                            f_init_p = self.fijii_np(self.subroot_p+'Block1/' + self.suffix + '/before_eq22/' + '0_f_mu.img',shape=(self.PETImage_shape),type_im='<f') # loading DIP output
                        if config["FLTNB"] == "double":
                            self.f_p.astype(np.float64)
                    elif ('ADMMLim' in config["method"] or config["method"] == 'MLEM' or config["method"] == 'OPTITR' or config["method"] == 'OSEM' or config["method"] == 'BSREM' or config["method"] == 'AML' or config["method"] == 'APGMAP'):
                        pet_algo=config["method"]
                        iteration_name = "iterations"
                        if (hasattr(self,'beta')):
                            iteration_name += beta_string
                        if ('ADMMLim' in config["method"]):
                            subdir = 'ADMM' + '_' + str(config["nb_threads"])
                            subdir = ''
                            #self.f_p = self.fijii_np(self.subroot_p + self.suffix + '/' + subdir + '/0_' + format(i) + '_it' + str(config["nb_inner_iteration"]) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                            #self.f_p = self.fijii_np(self.subroot_p + self.suffix + '/' + subdir + '/0_' + format(i) + '_it1' + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                            #self.f_p = self.fijii_np(self.subroot_p + self.suffix + '/' + subdir + '/0_1'  + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                            self.f_p = self.fijii_np(self.subroot_p + self.suffix + '/' + subdir + '/0'  + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                        #elif (config["method"] == 'BSREM'):
                        #    self.f_p = self.fijii_np(self.subroot_p + self.suffix + '/' +  config["method"] + '_beta_' + str(self.beta) + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                        else:
                            self.f_p = self.fijii_np(self.subroot_p + self.suffix + '/' +  config["method"] + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output

                    if (np.isnan(np.sum(self.f_p.astype(float)))):
                        nan_replicates.append(p)
                        continue

                    if (config["average_replicates"]): # Average images across replicates (for metrics except IR)
                        f += self.f_p
                        f_init_avg += f_init_p
                    elif (config["average_replicates"] == False and p == self.replicate):
                        f = self.f_p

                    f_list[p-1] = self.f_p

            # if len(nan_replicates) > 0:
            #     raise ValueError("naaaaaaaaaaaaaaaaaaaaaaaaaaaaaaan",nan_replicates)

            nb_usable_replicates = self.nb_replicates - len(nan_replicates)
            f /= self.nb_usable_replicates
            f_init_avg /= self.nb_usable_replicates

            Path(self.subroot + 'Images/tmp/' + self.suffix + '/' + 'binary/').mkdir(parents=True, exist_ok=True)

            for p in set(range(self.nb_replicates,0,-1)) - set(nan_replicates):
                f_var[self.phantom_ROI==1] += (f[self.phantom_ROI==1] - f_list[p-1][self.phantom_ROI==1])**2 / self.nb_usable_replicates
            
            self.write_image_tensorboard(self.writer,np.sqrt(f_var),self.method + " at convergence, std (not normalised) over " + str(self.nb_usable_replicates) + " replicates (FULL CONTRAST)",self.suffix,self.image_gt,0,full_contrast=True) # std of images at convergence across replicates in tensorboard
            path_img = self.subroot + 'Images/tmp/' + self.suffix + '/' + 'binary/'
            self.save_img(np.sqrt(f_var),path_img + self.method + " at convergence, std (not normalised) over " + str(self.nb_usable_replicates) + " replicates (FULL CONTRAST)" + ".img")

            f_var_gt = np.array(f_var)
            f_var[self.phantom_ROI==1] /= np.abs(f[self.phantom_ROI==1])
            f_var_gt[self.phantom_ROI==1] /= np.abs(self.image_gt[self.phantom_ROI==1])
            # print("f_var[0,:]",f_var[0,:].T)
            # f_var[self.phantom_ROI==1] += (f[self.phantom_ROI==1] - f_list[p-1][self.phantom_ROI==1])**2 / self.nb_usable_replicates
            # print("self.phantom_ROI[0,:]",self.phantom_ROI[0,:].T)
            # print("f[0,:]",f[0,:].T)
            # print("self.f_p[0,:]",f_list[p-1][0,:].T)
            # print("f_var[0,:]",f_var[0,:].T)
            # f_var[self.phantom_ROI==1] /= np.abs(self.f_p[self.phantom_ROI==1])
            # print("self.phantom_ROI[0,:]",self.phantom_ROI[0,:].T)
            # print("f[0,:]",f[0,:].T)
            # print("self.f_p[0,:]",f_list[p-1][0,:].T)
            # print("f_var[0,:]",f_var[0,:].T)
            # print("np.min(f_var)",np.min(np.sqrt(f_var)))
            # print("np.max(f_var)",np.min(np.sqrt(f_var)))

        # Save images 
        self.write_image_tensorboard(self.writer,self.f_p,self.method + " at convergence, for replicate 1",self.suffix,self.image_gt,0) # image at convergence in tensorboard
        self.write_image_tensorboard(self.writer,self.f_p,self.method + " at convergence, for replicate 1 (FULL CONTRAST)",self.suffix,self.image_gt,0,full_contrast=True) # image at convergence in tensorboard
        self.write_image_tensorboard(self.writer,f,self.method + " at convergence, averaged on " + str(self.nb_usable_replicates) + " replicates",self.suffix,self.image_gt,0) # mean of images at convergence across replicates in tensorboard
        self.write_image_tensorboard(self.writer,f,self.method + " at convergence, averaged on " + str(self.nb_usable_replicates) + " replicates (FULL CONTRAST)",self.suffix,self.image_gt,0,full_contrast=True) # mean of images at convergence across replicates in tensorboard
        # self.write_image_tensorboard(self.writer,np.sqrt(f_var),self.method + " at convergence, std over " + str(self.nb_usable_replicates) + " replicates",self.suffix,self.image_gt,0) # std of images at convergence across replicates in tensorboard
        # self.write_image_tensorboard(self.writer,self.phantom_ROI,"self.phantom_ROI",self.suffix,self.image_gt,0,full_contrast=True) # std of images at convergence across replicates in tensorboard
        self.write_image_tensorboard(self.writer,np.sqrt(f_var),self.method + " at convergence, std over " + str(self.nb_usable_replicates) + " replicates (FULL CONTRAST)",self.suffix,self.image_gt,0,full_contrast=True) # std of images at convergence across replicates in tensorboard
        self.write_image_tensorboard(self.writer,np.sqrt(f_var_gt),self.method + " at convergence, std (normalised by GT) over " + str(self.nb_usable_replicates) + " replicates (FULL CONTRAST)",self.suffix,self.image_gt,0,full_contrast=True) # std of images at convergence across replicates in tensorboard
        self.write_image_tensorboard(self.writer,f_init_p,self.method + " denoised initialization, for replicate 1 " + str(self.nb_usable_replicates) + " replicates",self.suffix,self.image_gt,0) # denoised initialization for one replicate in tensorboard
        self.write_image_tensorboard(self.writer,f_init_p,self.method + " denoised initialization, for replicate 1 (FULL CONTRAST) " + str(self.nb_usable_replicates) + " replicates (FULL CONTRAST)",self.suffix,self.image_gt,0,full_contrast=True) # denoised initialization for one replicate in tensorboard
        self.write_image_tensorboard(self.writer,f_init_avg,self.method + " denoised initialization over " + str(self.nb_usable_replicates) + " replicates",self.suffix,self.image_gt,0) # denoised initialization across replicates in tensorboard
        self.write_image_tensorboard(self.writer,f_init_avg,self.method + " denoised initialization over " + str(self.nb_usable_replicates) + " replicates (FULL CONTRAST)",self.suffix,self.image_gt,0,full_contrast=True) # denoised initialization across replicates in tensorboard
        
        # Save images as .img
        path_img = self.subroot + 'Images/tmp/' + self.suffix + '/' + 'binary/'
        self.save_img(self.f_p,path_img + self.method + " at convergence, for replicate 1" + ".img")
        self.save_img(f,path_img + self.method + " at convergence, averaged on " + str(self.nb_usable_replicates) + " replicates" + ".img")
        self.save_img(np.sqrt(f_var),path_img + self.method + " at convergence, std over " + str(self.nb_usable_replicates) + " replicates (FULL CONTRAST)" + ".img")
        self.save_img(np.sqrt(f_var_gt),path_img + self.method + " at convergence, std (normalised by GT) over " + str(self.nb_usable_replicates) + " replicates (FULL CONTRAST)" + ".img")
        self.save_img(f_init_p,path_img + self.method + " denoised initialization, for replicate 1 " + str(self.nb_usable_replicates) + " replicates" + ".img")
        self.save_img(f_init_avg,path_img + self.method + " denoised initialization over " + str(self.nb_usable_replicates) + " replicates" + ".img")

        IR_common = 23 # in %
        IR_common = 11 # in %

        for IR_common in [11,13,23,30]:
            IR_min = np.inf
            i_min = self.total_nb_iter
            
            # Show image with IR = 30%
            for i in range(self.total_nb_iter,i_init-1,-1):
                IR = 0
                p = 1
                if (config["average_replicates"] or (config["average_replicates"] == False and p == self.replicate)):
                    # Read image into array according to method
                    if(self.read_image_method(config,beta_string,i_init,p,i)): # ES found
                        break
                    # Compute IR metric for first replicate
                    self.compute_IR_bkg(self.PETImage_shape,self.f_p,i-i_init,self.IR_bkg_recon,self.phantom)
                    IR = self.IR_bkg_recon[i-i_init]

                    print("i",i)
                    print("IR = ",IR)
                    '''
                    print(IR < IR_common/100)
                    print(IR_common)
                    if (IR_common == 23):
                        raise ValueError("IR_commons")
                    '''
                    if (IR < IR_min):
                        IR_min = IR
                        i_min = i

                    
                    if (IR < IR_common/100):
                        print("IR = ",IR)
                        break

                    if (i == i_init): # No image was at IR_common level of noise, so save first and l
                        self.read_image_method(config,beta_string,i_init,p,i_min)
                        # Compute IR metric for first replicate
                        self.compute_IR_bkg(self.PETImage_shape,self.f_p,i-i_init,self.IR_bkg_recon,self.phantom)
                        IR = self.IR_bkg_recon[i-i_init]
                        i = i_min
                        break

            # Save images 
            self.write_image_tensorboard(self.writer,self.f_p,self.method + " at IR=" + str(int(round(IR,2)*100)) + "%, for replicate 1, it=" + str(i),self.suffix,self.image_gt,0) # image at IR=30% in tensorboard
            self.write_image_tensorboard(self.writer,self.f_p,self.method + " at IR=" + str(int(round(IR,2)*100)) + ", for replicate 1, it=" + str(i) + " (FULL CONTRAST)",self.suffix,self.image_gt,0,full_contrast=True) # image at IR=30% in tensorboard
 
    def compareImages(self,suffix):
        if (self.tensorboard):
            self.write_image_tensorboard(self.writer,self.image_method,self.method + " at convergence",suffix,self.image_gt,0,full_contrast=True) # ADMMLim at convergence in tensorboard
            #self.write_image_tensorboard(self.writer,self.image_APGMAP,"APGMAP at convergence",suffix,self.image_gt,0,full_contrast=True) # APGMAP at convergence in tensorboard
       
    def read_image_method(self,config,beta_string,i_init,p,i):
        self.subroot_p = self.subroot_data + 'debug/'*self.debug + '/' + self.phantom + '/' + 'replicate_' + str(p) + '/' + self.method + '/' # Directory root

        # Take NNEPPS images if NNEPPS is asked for this run
        if (config["NNEPPS"]):
            NNEPPS_string = "_NNEPPS"
        else:
            NNEPPS_string = ""
        if ( 'Gong' in config["method"] or  'nested' in config["method"]):
            if ('post_reco' in config["task"]):
                pet_algo=config["method"]+"to fit"
                iteration_name="(post reconstruction)"
            else:
                pet_algo=config["method"]
                iteration_name="iterations"
            if ('post_reco' in config["task"]):
                try:
                    self.f_p = self.fijii_np(self.subroot_p+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.net + '' + format(0) + '_epoch=' + format(i-i_init) + NNEPPS_string + '.img',shape=(self.PETImage_shape),type_im='<f') # loading DIP output
                except: # ES point is reached
                    return 1
            else:
                self.f_p = self.fijii_np(self.subroot_p+'Block2/' + self.suffix + '/out_cnn/'+ format(self.experiment)+'/out_' + self.net + '' + format(i-i_init) + "_FINAL" + NNEPPS_string + '.img',shape=(self.PETImage_shape),type_im='<f') # loading DIP output
                f_init_p = self.fijii_np(self.subroot_p+'Block1/' + self.suffix + '/before_eq22/' + '0_f_mu.img',shape=(self.PETImage_shape),type_im='<f') # loading DIP output
            if config["FLTNB"] == "double":
                self.f_p.astype(np.float64)
        elif ('ADMMLim' in config["method"] or config["method"] == 'MLEM' or config["method"] == 'OPTITR' or config["method"] == 'OSEM' or config["method"] == 'BSREM' or config["method"] == 'AML' or config["method"] == 'APGMAP'):
            pet_algo=config["method"]
            iteration_name = "iterations"
            if (hasattr(self,'beta')):
                iteration_name += beta_string
            if ('ADMMLim' in config["method"]):
                subdir = 'ADMM' + '_' + str(config["nb_threads"])
                subdir = ''
                #self.f_p = self.fijii_np(self.subroot_p + self.suffix + '/' + subdir + '/0_' + format(i) + '_it' + str(config["nb_inner_iteration"]) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                #self.f_p = self.fijii_np(self.subroot_p + self.suffix + '/' + subdir + '/0_' + format(i) + '_it1' + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                #self.f_p = self.fijii_np(self.subroot_p + self.suffix + '/' + subdir + '/0_1'  + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
                self.f_p = self.fijii_np(self.subroot_p + self.suffix + '/' + subdir + '/0'  + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
            #elif (config["method"] == 'BSREM'):
            #    self.f_p = self.fijii_np(self.subroot_p + self.suffix + '/' +  config["method"] + '_beta_' + str(self.beta) + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output
            else:
                self.f_p = self.fijii_np(self.subroot_p + self.suffix + '/' +  config["method"] + '_it' + format(i) + NNEPPS_string + '.img',shape=(self.PETImage_shape)) # loading optimizer output

        return 0