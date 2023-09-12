from torch import max, abs, optim, load, mean
from torch.nn import ReplicationPad2d, Conv2d, BatchNorm2d, LeakyReLU, Conv2d, BatchNorm2d, LeakyReLU, Sequential, Upsample, ReLU, MSELoss
from pytorch_lightning import LightningModule, seed_everything
from numpy import min as min_np
from numpy import max as max_np
from numpy import mean as mean_np
from numpy import std as std_np
from numpy import ones_like, dtype, fromfile, sign, newaxis, copy

from pathlib import Path
from os.path import isfile

# Local files to import
from iWMV import iWMV

class DIP_2D(LightningModule):

    def __init__(self, param1_scale_im_corrupt, param2_scale_im_corrupt, scaling_input, config, root, subroot, method, all_images_DIP, global_it, fixed_hyperparameters_list, hyperparameters_list, debug, suffix, last_iter, override_input, scanner):
        super().__init__()

        #'''
        # Set random seed if asked (for NN weights here)
        if (isfile(root + "/seed.txt")): # Put root for path because raytune path !!!
            with open(root + "/seed.txt", 'r') as file:
                random_seed = file.read().rstrip()
            if (eval(random_seed)):
                seed_everything(1)
                # import torch
                # torch.manual_seed(1)
                # torch.cuda.seed()
                # torch.use_deterministic_algorithms(True)

        #'''
        
        # Defining variables from config        
        self.lr = config['lr']
        self.opti_DIP = config['opti_DIP']
        self.sub_iter_DIP = config['sub_iter_DIP']
        self.skip = config['skip_connections']
        self.method = method
        self.all_images_DIP = all_images_DIP
        self.global_it = global_it
        self.param1_scale_im_corrupt = param1_scale_im_corrupt
        self.param2_scale_im_corrupt = param2_scale_im_corrupt

        self.fixed_hyperparameters_list = fixed_hyperparameters_list
        self.hyperparameters_list = hyperparameters_list
        self.scaling_input = scaling_input
        self.debug = debug
        self.subroot = subroot
        self.root = root
        self.config = config
        self.experiment = config["experiment"]
        '''
        ## Variables for WMV ##
        self.queueQ = []
        self.VAR_min = inf
        self.SUCCESS = False
        self.stagnate = 0
        '''
        self.DIP_early_stopping = config["DIP_early_stopping"]
        self.override_input = override_input
        self.scanner = scanner
        
        # Initialize early stopping method if asked for
        if(self.DIP_early_stopping):
            self.initialize_WMV(config,fixed_hyperparameters_list,hyperparameters_list,debug,param1_scale_im_corrupt,param2_scale_im_corrupt,scaling_input,suffix,global_it,root,scanner)

        self.write_current_img_mode = True
        #self.suffix = self.suffix_func(config,hyperparameters_list)
        #if (config["task"] == "post_reco"):
        #    self.suffix = config["task"] + ' ' + self.suffix
        self.suffix = suffix
        
        self.last_iter = last_iter + 1

        # Monitor lr
        self.mean_inside_list = []
        self.ema_lr = [0, 0]

        '''
        if (config['mlem_sequence'] is None):
            self.write_current_img_mode = True
            self.suffix = self.suffix_func(config)
        else:
            self.write_current_img_mode = False
        '''
        # Defining CNN variables
        L_relu = 0.2
        num_channel = [16, 32, 64, 128]
        pad = [0, 0]

        # Layers in CNN architecture
        self.deep1 = Sequential(ReplicationPad2d(1),
                                   Conv2d(1, num_channel[0], (3, 3), stride=1, padding=pad[1]))#,
                                #    BatchNorm2d(num_channel[0]),
                                #    LeakyReLU(L_relu),
                                #    ReplicationPad2d(1),
                                #    Conv2d(num_channel[0], num_channel[0], (3, 3), stride=1, padding=pad[1]),
                                #    BatchNorm2d(num_channel[0]),
                                #    LeakyReLU(L_relu))

        self.bn1 = BatchNorm2d(num_channel[0])
        self.leaky1 = LeakyReLU(L_relu)
        # self.leaky1 = LeakyReLU(L_relu)
        # self.leaky1 = LeakyReLU(L_relu)


        self.down1 = Sequential(ReplicationPad2d(1),
                                   Conv2d(num_channel[0], num_channel[0], 3, stride=(2, 2), padding=pad[1]),
                                   BatchNorm2d(num_channel[0]),
                                   LeakyReLU(L_relu))

        self.deep2 = Sequential(ReplicationPad2d(1),
                                   Conv2d(num_channel[0], num_channel[1], (3, 3), stride=1, padding=pad[1]),
                                   BatchNorm2d(num_channel[1]),
                                   LeakyReLU(L_relu),
                                   ReplicationPad2d(1),
                                   Conv2d(num_channel[1], num_channel[1], (3, 3), stride=1, padding=pad[1]),
                                   BatchNorm2d(num_channel[1]),
                                   LeakyReLU(L_relu))

        self.down2 = Sequential(ReplicationPad2d(1),
                                   Conv2d(num_channel[1], num_channel[1], 3, stride=(2, 2), padding=pad[1]),
                                   BatchNorm2d(num_channel[1]),
                                   LeakyReLU(L_relu))

        self.deep3 = Sequential(ReplicationPad2d(1),
                                   Conv2d(num_channel[1], num_channel[2], (3, 3), stride=1, padding=pad[1]),
                                   BatchNorm2d(num_channel[2]),
                                   LeakyReLU(L_relu),
                                   ReplicationPad2d(1),
                                   Conv2d(num_channel[2], num_channel[2], (3, 3), stride=1, padding=pad[1]),
                                   BatchNorm2d(num_channel[2]),
                                   LeakyReLU(L_relu))

        self.down3 = Sequential(ReplicationPad2d(1),
                                   Conv2d(num_channel[2], num_channel[2], 3, stride=(2, 2), padding=pad[1]),
                                   BatchNorm2d(num_channel[2]),
                                   LeakyReLU(L_relu))

        self.deep4 = Sequential(ReplicationPad2d(1),
                                   Conv2d(num_channel[2], num_channel[3], (3, 3), stride=1, padding=pad[1]),
                                   BatchNorm2d(num_channel[3]),
                                   LeakyReLU(L_relu),
                                   ReplicationPad2d(1),
                                   Conv2d(num_channel[3], num_channel[3], (3, 3), stride=1, padding=pad[1]),
                                   BatchNorm2d(num_channel[3]),
                                   LeakyReLU(L_relu))

        self.up1 = Sequential(Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False),
                                 ReplicationPad2d(1),
                                 Conv2d(num_channel[3], num_channel[2], 3, stride=(1, 1), padding=pad[0]),
                                 BatchNorm2d(num_channel[2]),
                                 LeakyReLU(L_relu))

        self.deep5 = Sequential(ReplicationPad2d(1),
                                   Conv2d(num_channel[2], num_channel[2], (3, 3), stride=1, padding=pad[1]),
                                   BatchNorm2d(num_channel[2]),
                                   LeakyReLU(L_relu),
                                   ReplicationPad2d(1),
                                   Conv2d(num_channel[2], num_channel[2], (3, 3), stride=1, padding=pad[1]),
                                   BatchNorm2d(num_channel[2]),
                                   LeakyReLU(L_relu))

        self.up2 = Sequential(Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False),
                                 ReplicationPad2d(1),
                                 Conv2d(num_channel[2], num_channel[1], 3, stride=(1, 1), padding=pad[0]),
                                 BatchNorm2d(num_channel[1]),
                                 LeakyReLU(L_relu))

        self.deep6 = Sequential(ReplicationPad2d(1),
                                   Conv2d(num_channel[1], num_channel[1], (3, 3), stride=1, padding=pad[0]),
                                   BatchNorm2d(num_channel[1]),
                                   LeakyReLU(L_relu),
                                   ReplicationPad2d(1),
                                   Conv2d(num_channel[1], num_channel[1], (3, 3), stride=1, padding=pad[1]),
                                   BatchNorm2d(num_channel[1]),
                                   LeakyReLU(L_relu))

        self.up3 = Sequential(Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False),
                                 ReplicationPad2d(1),
                                 Conv2d(num_channel[1], num_channel[0], 3, stride=(1, 1), padding=pad[0]),
                                 BatchNorm2d(num_channel[0]),
                                 LeakyReLU(L_relu))

        self.deep7 = Sequential(ReplicationPad2d(1),
                                   Conv2d(num_channel[0], num_channel[0], (3, 3), stride=1, padding=pad[0]),
                                   BatchNorm2d(num_channel[0]),
                                   LeakyReLU(L_relu),
                                   ReplicationPad2d(1),
                                   Conv2d(num_channel[0], 1, (3, 3), stride=1, padding=pad[1]),
                                   BatchNorm2d(1))

        self.positivity = ReLU() # Final ReLU to enforce positivity of ouput image
        # self.positivity = SiLU() # Final SiLU, smoother than ReLU but not positive
        # self.positivity = Softplus() # Final SiLU to enforce positivity of ouput image, smoother than ReLU

    def forward(self, x):
        # Encoder
        out1 = self.deep1(x)
        self.write_current_img_task(out1,inside=True,name="beforebn1")
        out1 = self.bn1(out1)
        self.write_current_img_task(out1,inside=True,name="beforeleaky1")
        out1 = self.leaky1(out1)
        self.write_current_img_task(out1,inside=True,name="nextleaky1")
        out = self.down1(out1)
        out2 = self.deep2(out)
        out = self.down2(out2)
        out3 = self.deep3(out)
        out = self.down3(out3)
        out = self.deep4(out)

        # Decoder
        out = self.up1(out)
        if (self.skip >= 1): # or self.override_input):
            out_skip1 = out3 + out
            out = self.deep5(out_skip1)
        else:
            out = self.deep5(out)
        out = self.up2(out)
        if (self.skip >= 2): # or self.override_input):
            out_skip2 = out2 + out
            out = self.deep6(out_skip2)
        else:
            out = self.deep6(out)
        out = self.up3(out)
        self.write_current_img_task(out1,inside=True,name="from encoder before adding SC3")
        self.write_current_img_task(out,inside=True,name="from decoder before adding SC3")
        if (self.skip >= 3): # or self.override_input):
            out_skip3 = out1 + out
            out_skip3 = out1
            self.write_current_img_task(out_skip3,inside=True,name="after adding SC3")
            out = self.deep7(out_skip3)
        else:
            out = self.deep7(out)

        if (self.method == 'Gong'):
            # self.write_current_img_task(out,inside=True)
            out = self.positivity(out)

        return out

    def DIP_loss(self, out, image_corrupt_torch):
        return MSELoss()(out, image_corrupt_torch) # for DIP and DD

    def training_step(self, train_batch, batch_idx):
        image_net_input_torch, image_corrupt_torch = train_batch
        out = self.forward(image_net_input_torch)
        # logging using tensorboard logger
        loss = self.DIP_loss(out, image_corrupt_torch)
        self.logger.experiment.add_scalar('loss', loss,self.current_epoch)        
        
        # Save image over epochs
        if (self.write_current_img_mode):
            self.write_current_img(out)

        # Monitor learning rate across iterations
        self.monitor_lr(out,image_corrupt_torch)

        # WMV
        self.run_WMV(out,self.config,self.fixed_hyperparameters_list,self.hyperparameters_list,self.debug,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input,self.suffix,self.global_it,self.root,self.scanner)
        
        return loss

    def configure_optimizers(self):
        # Optimization algorithm according to command line

        """
        Optimization of the DNN with SGLD
        """

        if (self.opti_DIP == 'Adam'):
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=5E-8) # Optimizing using Adam
            #optimizer = optim.Adam(self.parameters(), lr=self.lr) # Optimizing using Adam
        elif (self.opti_DIP == 'LBFGS' or self.opti_DIP is None): # None means no argument was given in command line
            optimizer = optim.LBFGS(self.parameters(), lr=self.lr, history_size=10, max_iter=10,line_search_fn=None) # Optimizing using L-BFGS
            # optimizer = optim.LBFGS(self.parameters(), lr=self.lr, history_size=10, max_iter=4,line_search_fn="strong_wolfe") # Optimizing using L-BFGS 1
            #optimizer = optim.LBFGS(self.parameters(), lr=self.lr, history_size=10, max_iter=40,line_search_fn="strong_wolfe") # Optimizing using L-BFGS 3
        elif (self.opti_DIP == 'SGD'):
            optimizer = optim.SGD(self.parameters(), lr=self.lr) # Optimizing using SGD
        elif (self.opti_DIP == 'Adadelta'):
            optimizer = optim.Adadelta(self.parameters()) # Optimizing using Adadelta
        return optimizer

    def write_current_img(self,out):
        if (self.all_images_DIP == "False"):
            if ((self.current_epoch%(self.sub_iter_DIP // 10) == (self.sub_iter_DIP // 10) -1)):
                self.write_current_img_task(out)
        elif (self.all_images_DIP == "True"):
            self.write_current_img_task(out)
        elif (self.all_images_DIP == "Last"):
            if (self.current_epoch == self.sub_iter_DIP - 1):
                self.write_current_img_task(out)

    def write_current_img_task(self,out,inside=False,name=""):
        try:
            out_np = out.detach().numpy()[0,:,:,:]
        except:
            out_np = out.cpu().detach().numpy()[0,:,:,:]

        print("self.current_epoch",self.current_epoch)
        if (inside):
            print("save before ReLU here")
            if (name == ""):
                name = "beforeReLU"
            self.save_img(out_np, self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/' + name + '_' + 'DIP' + format(self.global_it) + '_epoch=' + format(self.current_epoch + self.last_iter) + '.img') # The saved images are not destandardized !!!!!! Do it when showing images in tensorboard
        else:
            self.save_img(out_np, self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + 'DIP' + format(self.global_it) + '_epoch=' + format(self.current_epoch + self.last_iter) + '.img') # The saved images are not destandardized !!!!!! Do it when showing images in tensorboard
                            
    def suffix_func(self,config,hyperparameters_list,NNEPPS=False):
        config_copy = dict(config)
        if (NNEPPS==False):
            config_copy.pop('NNEPPS',None)
        #config_copy.pop('nb_outer_iteration',None)
        suffix = "config"
        for key, value in config_copy.items():
            if key in hyperparameters_list:
                suffix +=  "_" + key[:min(len(key),5)] + "=" + str(value)
        return suffix

    def save_img(self,img,name):
        fp=open(name,'wb')
        img.tofile(fp)
        print('Succesfully save in:', name)

    def monitor_lr(self,out,image_corrupt_torch):
        if (not hasattr(self.config,"monitor_lr")):
            self.config["monitor_lr"] = False
        if (self.config["monitor_lr"]):
            out_descale_np = self.descale_imag(copy(out),self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)
            # try:
            #     # out_descale_np = out_descale.detach().numpy()[0,0,:,:]
            #     out_descale_np = out_descale
            #     image_corrupt_np = image_corrupt_torch.detach().numpy()[0,0,:,:]
            # except:
            #     # out_descale_np = out_descale.cpu().detach().numpy()[0,0,:,:]
            #     image_corrupt_np = image_corrupt_torch.cpu().detach().numpy()[0,0,:,:]
            
            image_corrupt_np = self.descale_imag(image_corrupt_torch,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)

            self.subroot_data = self.root + '/data/Algo/' # Directory root
            self.phantom = self.config["image"]

            self.PETImage_shape_str = self.read_input_dim(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.hdr')
            self.PETImage_shape = self.input_dim_str_to_list(self.PETImage_shape_str)

            self.phantom_ROI = self.get_phantom_ROI(self.phantom)


            mean_inside = mean_np(out_descale_np * self.phantom_ROI) / mean_np(image_corrupt_np * self.phantom_ROI)
            self.mean_inside_list.append(mean_inside)

            if (self.current_epoch >= 2):
                alpha_ema_lr = 0.1
                self.ema_lr.append((1-alpha_ema_lr) * self.ema_lr[self.current_epoch-1] + alpha_ema_lr * self.mean_inside_list[self.current_epoch])

                print(self.ema_lr[self.current_epoch])

                if (sign(self.ema_lr[self.current_epoch] - self.ema_lr[self.current_epoch - 1]) != sign(self.ema_lr[self.current_epoch - 1] - self.ema_lr[self.current_epoch - 2])):
                    # if (self.lr > 1e-5): # Minimum lr value to 1e-5, does not need to better stability
                    self.lr /= 2
                    print(self.lr)
                    print("chaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaange lrrrrrrrrrrrrrrrrrrrrrrrrrrr")
    
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

    def run_WMV(self,out,config,fixed_hyperparameters_list,hyperparameters_list,debug,param1_scale_im_corrupt,param2_scale_im_corrupt,scaling_input,suffix,global_it,root,scanner):
        if (self.DIP_early_stopping):
            self.SUCCESS = self.classWMV.SUCCESS
            self.log("SUCCESS", int(self.classWMV.SUCCESS))

            try:
                out_np = out.detach().numpy()[0,0,:,:]
            except:
                out_np = out.cpu().detach().numpy()[0,0,:,:]

            if (len(out_np.shape) == 2): # 2D
                out_np = out_np[:,:,newaxis]

            self.classWMV.SUCCESS,self.classWMV.VAR_min,self.classWMV.stagnate = self.classWMV.WMV(copy(out_np),self.current_epoch,self.classWMV.queueQ,self.classWMV.SUCCESS,self.classWMV.VAR_min,self.classWMV.stagnate)
            self.VAR_recon = self.classWMV.VAR_recon
            self.MSE_WMV = self.classWMV.MSE_WMV
            self.PSNR_WMV = self.classWMV.PSNR_WMV
            self.SSIM_WMV = self.classWMV.SSIM_WMV
            self.epochStar = self.classWMV.epochStar
            '''
            if self.EMV_or_WMV == "EMV":
                self.alpha_EMV = self.classWMV.alpha_EMV
            else:
                self.windowSize = self.classWMV.windowSize
            '''
            self.patienceNumber = self.classWMV.patienceNumber

            if self.SUCCESS:
                print("SUCCESS WMVVVVVVVVVVVVVVVVVV")
                self.initialize_WMV(config,fixed_hyperparameters_list,hyperparameters_list,debug,param1_scale_im_corrupt,param2_scale_im_corrupt,scaling_input,suffix,global_it,root,scanner)
        
        else:
            self.log("SUCCESS", int(False))


    def norm_imag(self,img):
        print("nooooooooorm")
        """ Normalization of input - output [0..1] and the normalization value for each slide"""
        if (max_np(img) - min_np(img)) != 0:
            return (img - min_np(img)) / (max_np(img) - min_np(img)), min_np(img), max_np(img)
        else:
            return img, min_np(img), max_np(img)

    def denorm_imag(self,image, mini, maxi):
        """ Denormalization of input - output [0..1] and the normalization value for each slide"""
        image_np = image.detach().numpy()
        return self.denorm_numpy_imag(image_np, mini, maxi)

    def denorm_numpy_imag(self,img, mini, maxi):
        if (maxi - mini) != 0:
            return img * (maxi - mini) + mini
        else:
            return img


    def norm_positive_imag(self,img):
        """ Positive normalization of input - output [0..1] and the normalization value for each slide"""
        if (max_np(img) - min_np(img)) != 0:
            print(max_np(img))
            print(min_np(img))
            return img / max_np(img), 0, max_np(img)
        else:
            return img, 0, max_np(img)

    def denorm_positive_imag(self,image, mini, maxi):
        """ Positive normalization of input - output [0..1] and the normalization value for each slide"""
        image_np = image.detach().numpy()
        return self.denorm_numpy_imag(image_np, mini, maxi)

    def denorm_numpy_positive_imag(self, img, mini, maxi):
        if (maxi - mini) != 0:
            return img * maxi 
        else:
            return img

    def stand_imag(self,image_corrupt):
        print("staaaaaaaaaaand")
        """ Standardization of input - output with mean 0 and std 1 for each slide"""
        mean_im=mean_np(image_corrupt)
        std_im=std_np(image_corrupt)
        image_center = image_corrupt - mean_im
        if (std_im == 0.):
            raise ValueError("std 0")
        image_corrupt_std = image_center / std_im
        return image_corrupt_std,mean_im,std_im

    def destand_numpy_imag(self,image, mean_im, std_im):
        """ Destandardization of input - output with mean 0 and std 1 for each slide"""
        return image * std_im + mean_im

    def destand_imag(self,image, mean_im, std_im):
        image_np = image.detach().numpy()
        return self.destand_numpy_imag(image_np, mean_im, std_im)

    def rescale_imag(self,image_corrupt, scaling):
        """ Scaling of input """
        if (scaling == 'standardization'):
            return self.stand_imag(image_corrupt)
        elif (scaling == 'normalization'):
            return self.norm_imag(image_corrupt)
        elif (scaling == 'positive_normalization'):
            return self.norm_positive_imag(image_corrupt)
        else: # No scaling required
            return image_corrupt, 0, 0

    def descale_imag(self,image, param_scale1, param_scale2, scaling='standardization'):
        """ Descaling of input """
        try:
            image_np = image.detach().numpy()
        except:
            image_np = image.cpu().detach().numpy()
        if (scaling == 'standardization'):
            return self.destand_numpy_imag(image_np, param_scale1, param_scale2)
        elif (scaling == 'normalization'):
            return self.denorm_numpy_imag(image_np, param_scale1, param_scale2)
        elif (scaling == 'positive_normalization'):
            return self.denorm_numpy_positive_imag(image_np, param_scale1, param_scale2)
        else: # No scaling required
            return image_np

    def get_phantom_ROI(self,image='image0'):
        # Select only phantom ROI, not whole reconstructed image
        path_phantom_ROI = self.subroot_data+'Data/database_v2/' + image + '/' + "phantom_mask" + str(image[5:]) + '.raw'
        my_file = Path(path_phantom_ROI)
        if (my_file.is_file()):
            phantom_ROI = self.fijii_np(path_phantom_ROI, shape=(self.PETImage_shape),type_im='<f')
        else:
            print("No phantom file for this phantom")
            # Loading Ground Truth image to compute metrics
            try:
                image_gt = self.fijii_np(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.raw',shape=(self.PETImage_shape),type_im='<f')
            except:
                raise ValueError("Please put the header file from CASToR with name of phantom")
            phantom_ROI = ones_like(image_gt)
            #raise ValueError("No phantom file for this phantom")
            #phantom_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + image + '/' + "background_mask" + image[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
            
        return phantom_ROI
    
    def fijii_np(self,path,shape,type_im=None):
        """"Transforming raw data to numpy array"""
        if (type_im is None):
            if (self.FLTNB == 'float'):
                type_im = '<f'
            elif (self.FLTNB == 'double'):
                type_im = '<d'

        attempts = 0

        while attempts < 1000:
            attempts += 1
            try:
                type_im = ('<f')*(type_im=='<f') + ('<d')*(type_im=='<d')
                file_path=(path)
                dtype_np = dtype(type_im)
                with open(file_path, 'rb') as fid:
                    data = fromfile(fid,dtype_np)
                    if (1 in shape): # 2D
                        #shape = (shape[0],shape[1])
                        image = data.reshape(shape)
                    else: # 3D
                        image = data.reshape(shape[::-1])
                attempts = 1000
                break
            except:
                # fid.close()
                type_im = ('<f')*(type_im=='<d') + ('<d')*(type_im=='<f')
                file_path=(path)
                dtype_np = dtype(type_im)
                with open(file_path, 'rb') as fid:
                    data = fromfile(fid,dtype_np)
                    if (1 in shape): # 2D
                        #shape = (shape[0],shape[1])
                        try:
                            image = data.reshape(shape)
                        except Exception as e:
                            # print(data.shape)
                            # print(type_im)
                            # print(dtype_np)
                            # print(fid)
                            # '''
                            # import numpy as np
                            # data = fromfile(fid,dtype('<f'))
                            # np.save('data' + str(self.replicate) + '_' + str(attempts) + '_f.npy', data)
                            # '''
                            # print('Failed: '+ str(e) + '_' + str(attempts))
                            pass
                    else: # 3D
                        image = data.reshape(shape[::-1])
                
                fid.close()
            '''
            image = data.reshape(shape)
            #image = transpose(image,axes=(1,2,0)) # imshow ok
            #image = transpose(image,axes=(1,0,2)) # imshow ok
            #image = transpose(image,axes=(0,1,2)) # imshow ok
            #image = transpose(image,axes=(0,2,1)) # imshow ok
            #image = transpose(image,axes=(2,0,1)) # imshow ok
            #image = transpose(image,axes=(2,1,0)) # imshow ok
            '''
            
        #'''
        #image = data.reshape(shape)
        '''
        try:
            print(image[0,0])
        except Exception as e:
            print('exception image: '+ str(e))
        '''
        # print("read from ", path)
        return image

    def read_input_dim(self,file_path):
        # Read CASToR header file to retrieve image dimension """
        try:
            with open(file_path) as f:
                for line in f:
                    if 'matrix size [1]' in line.strip():
                        dim1 = [int(s) for s in line.split() if s.isdigit()][-1]
                    if 'matrix size [2]' in line.strip():
                        dim2 = [int(s) for s in line.split() if s.isdigit()][-1]
                    if 'matrix size [3]' in line.strip():
                        dim3 = [int(s) for s in line.split() if s.isdigit()][-1]
        except:
            raise ValueError("Please put the header file from CASToR with name of phantom")
        # Create variables to store dimensions
        PETImage_shape = (dim1,dim2,dim3)
        # if (self.scanner == "mMR_3D"):
        #     PETImage_shape = (int(dim1/2),int(dim2/2),dim3)
        PETImage_shape_str = str(dim1) + ','+ str(dim2) + ',' + str(dim3)
        print('image shape :', PETImage_shape)
        return PETImage_shape_str

    def input_dim_str_to_list(self,PETImage_shape_str):
        return [int(e.strip()) for e in PETImage_shape_str.split(',')]#[:-1]