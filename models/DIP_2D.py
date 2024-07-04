from torch import optim, clone, matmul, Tensor, reshape, cat
from torch.nn import ReplicationPad2d, Conv2d, BatchNorm2d, LeakyReLU, Conv2d, BatchNorm2d, LeakyReLU, Sequential, Upsample, ReLU, MSELoss
from pytorch_lightning import LightningModule, seed_everything
from numpy import min as min_np
from numpy import max as max_np
from numpy import mean as mean_np
from numpy import std as std_np
from numpy import ravel as ravel_np
from numpy import ones_like, dtype, fromfile, sign, newaxis, copy, zeros, float32, squeeze, where
from numpy.random import seed, uniform


from pathlib import Path
from os.path import isfile

# Local files to import
from iMovingVariance import iMovingVariance

class DIP_2D(LightningModule):

    def __init__(self, param1_scale_im_corrupt, param2_scale_im_corrupt, scaling_input, config, root, subroot, subroot_phantom, method, all_images_DIP, global_it, fixed_hyperparameters_list, hyperparameters_list, debug, suffix, override_input, scanner, simulation, sub_iter_DIP_already_done, override_SC_init, DIP_early_stopping, image_net_input_torch):
        super().__init__()

        # Save all the arguments passed to your model in the checkpoint, especially to save learning rate
        self.save_hyperparameters()

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
        
        # from torch import load
        # if (isfile(self.checkpoint_simple_path_exp + '/optimizer.pth')):
        #     ckpt = load(self.checkpoint_simple_path_exp + '/optimizer.pth')
        #     self.current_epoch = ckpt['epoch']

        # Defining variables from config        
        self.lr = config['lr']
        self.opti_DIP = config['opti_DIP']
        if (global_it == -1):
            self.sub_iter_DIP = config["sub_iter_DIP_initial_and_final"] # User defined maximum number of initial DIP iterations
            # self.sub_iter_DIP = config["DIP_it_if_no_ES_found"] + config["patienceNumber"] # Maximum number of initial DIP iterations is set to DIP_it_if_no_ES_found + patienceNumber
        else:
            self.sub_iter_DIP = config['sub_iter_DIP']
        self.skip = config['skip_connections']
        self.method = method
        self.all_images_DIP = all_images_DIP
        self.global_it = global_it
        self.param1_scale_im_corrupt = param1_scale_im_corrupt
        self.param2_scale_im_corrupt = param2_scale_im_corrupt
        
        self.sub_iter_DIP_already_done_before_training = sub_iter_DIP_already_done
        self.sub_iter_DIP_already_done = sub_iter_DIP_already_done
        self.sub_iter_DIP_this_global_it = 0
        self.fixed_hyperparameters_list = fixed_hyperparameters_list
        self.hyperparameters_list = hyperparameters_list
        self.scaling_input = scaling_input
        self.debug = debug
        self.root = root
        self.subroot = root + subroot
        self.subroot_phantom = subroot_phantom
        self.config = config
        self.experiment = config["experiment"]
        self.image_net_input_torch = image_net_input_torch

        # MIC study

        if (self.global_it < 0):
            if ("initDNA" in self.config):
                if (self.config["initDNA"]):
                    self.initDNA = True
                else:
                    self.initDNA = False
            else:
                self.initDNA = False
        else:
            self.initDNA = False

        if (self.global_it < 0):
            if ("initDIPRecon" in self.config):
                if (self.config["initDIPRecon"]):
                    self.initDIPRecon = True
                else:
                    self.initDIPRecon = False
            else:
                self.initDIPRecon = False
        else:
            self.initDIPRecon = False

        self.override_SC_init = override_SC_init
        if ("dropout" in config):
            self.dropout = config['dropout']
        else:
            self.dropout = 0
        if ("several_DIP_inputs" in config): # Put several times the input
            self.several_DIP_inputs = config["several_DIP_inputs"]
        else:
            self.several_DIP_inputs = 1

        self.num_total_batch = -1
        self.end_epoch = False

        self.DIP_early_stopping = DIP_early_stopping
        self.override_input = override_input
        self.scanner = scanner
        self.simulation = simulation

        # Initialize early stopping method if asked for
        if(self.DIP_early_stopping):
            self.classMV = iMovingVariance(config)
            # self.classMV.model_class = DIP_2D
            self.classMV.model_class = type(self)
            self.classMV.image_net_input_torch = self.image_net_input_torch
            self.classMV.initialize_MV(config,fixed_hyperparameters_list,hyperparameters_list,debug,param1_scale_im_corrupt,param2_scale_im_corrupt,scaling_input,suffix,global_it,self.sub_iter_DIP,root,subroot,scanner, simulation, image_net_input_torch)

        self.write_current_img_mode = True
        #self.suffix = self.suffix_func(config,hyperparameters_list)
        #if (config["task"] == "post_reco"):
        #    self.suffix = config["task"] + ' ' + self.suffix
        self.suffix = suffix
        
        # Monitor lr
        self.mean_inside_list = []
        self.ema_lr = [0, 0]

        # End to end reconstruction : load system matrix A
        if ("end_to_end" in config): # Check if run DNA with end to end mode
            if (config["end_to_end"]):
                self.end_to_end = True
            else:
                self.end_to_end = False
        else:
            self.end_to_end = False
        if (self.end_to_end):
            # Define shapes
            self.subroot = self.root + '/data/Algo/' # Directory root
            self.phantom = self.config["image"]
            self.PETImage_shape_str = self.read_input_dim(self.subroot + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.hdr')
            self.PETImage_shape = self.input_dim_str_to_list(self.PETImage_shape_str)
            if (self.simulation and self.scanner == "mMR_2D"):
                self.sinogram_shape = (344,252,1)
                self.sinogram_shape_transpose = (252,344,1)
            elif (self.simulation and self.scanner == "mCT_2D"):
                self.sinogram_shape = (336,336,1)
                self.sinogram_shape_transpose = (336,336,1)
            # Load stored system matrix A
            if ("4" in self.phantom or "10" in self.phantom): # cylindrical phantom are reconstructed with voxels of 4mm
                A = squeeze(self.fijii_np(self.subroot + "/final_syst_mat_vox_4mm.img",(self.sinogram_shape[0]*self.sinogram_shape[1],self.PETImage_shape[0]*self.PETImage_shape[1],1),type_im='<f'))
            elif ("5" in self.phantom): # brain phantom are reconstructed with voxels of 2mm
                A = squeeze(self.fijii_np(self.subroot + "/final_syst_mat_vox_2mm.img",(self.sinogram_shape[0]*self.sinogram_shape[1],self.PETImage_shape[0]*self.PETImage_shape[1],1),type_im='<f'))
            else: # Assuming that other cases are reconstructed with voxels of 2mm
                A = squeeze(self.fijii_np(self.subroot + "/final_syst_mat_vox_2mm.img",(self.sinogram_shape[0]*self.sinogram_shape[1],self.PETImage_shape[0]*self.PETImage_shape[1],1),type_im='<f'))
            # Load randoms, scatters, norm and attenuation sinogram
            self.randoms_sinogram = Tensor(ravel_np(self.fijii_np(self.subroot + "/Data/database_v2/" + self.phantom + "/simu0_1/simu0_1_rd.s",self.sinogram_shape_transpose,type_im='<f')))
            self.scatters_sinogram = Tensor(ravel_np(self.fijii_np(self.subroot + "/Data/database_v2/" + self.phantom + "/simu0_1/simu0_1_sc.s",self.sinogram_shape_transpose,type_im='<f')))

            atn = ravel_np(self.fijii_np(self.subroot + "Data/database_v2/" + self.phantom + "/simu0_1/simu0_1_at.s",self.sinogram_shape_transpose,type_im='<f'))
            self.attenuation_sinogram = where(atn==0,0,1/atn)
            norm = ravel_np(self.fijii_np(self.subroot + "/Data/database_v2/" + self.phantom + "/simu0_1/simu0_1_nm.s",self.sinogram_shape_transpose,type_im='<f'))
            self.normalization_sinogram = where(norm==0,0,1/norm)
            # Define mask (sinogram bins == 0 should not be taken into account in loss computation)
            self.norm_mask = Tensor(self.normalization_sinogram)

            # Retrieve calibration factor
            datafile_castor_path = self.subroot+'Data/database_v2/' + self.phantom + '/' + "data" + self.phantom[5:] + '_' + str(config["replicates"]) + '/' + "data" + self.phantom[5:] + '_' + str(config["replicates"] ) + '.cdh'
            # Open the file
            with open(datafile_castor_path, 'r') as file:
                lines = file.readlines()
                # Get the 12th line
                line_12 = lines[11]
                # Split the line into words
                words = line_12.split()
                # Find the index of the word 'factor:'
                index = words.index('factor:')
                # The value of the calibration factor is the next word
                self.calibration_factor = float(words[index + 1])
            
            # Apply all corrections and convert A to torch tensor
            for j in range(self.PETImage_shape[0]*self.PETImage_shape[1]):
                print(j)
                A[:,j] = 1 / self.calibration_factor * A[:,j] * self.normalization_sinogram * self.attenuation_sinogram
            self.A_torch = Tensor(A)

            # Put variables on GPU if asked
            if (config["processing_unit"] == "GPU"):
                self.randoms_sinogram = self.randoms_sinogram.to("cuda")
                self.scatters_sinogram = self.scatters_sinogram.to("cuda")
                self.norm_mask = self.norm_mask.to("cuda")
                self.A_torch = self.A_torch.to("cuda")


            # For LBFGS
            self.counter_inside_epoch = 0


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
                                   Conv2d(1, num_channel[0], (3, 3), stride=1, padding=pad[1]),
                                   BatchNorm2d(num_channel[0]),
                                   LeakyReLU(L_relu),
                                   ReplicationPad2d(1),
                                   Conv2d(num_channel[0], num_channel[0], (3, 3), stride=1, padding=pad[1]),
                                   BatchNorm2d(num_channel[0]),
                                   LeakyReLU(L_relu))

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

        # Dropout, changing numpy seed at each global iteration
        if (self.dropout > 0):
            # seed((self.global_it+100) * 10000 + self.current_epoch)
            drop_sample = uniform(0,1,3)
            # drop_sample = rand(3)
            # seed(1)
        else:
            drop_sample = (1,1,1)

        # Encoder
        out1 = self.deep1(x)
        out = self.down1(out1)
        out2 = self.deep2(out)
        out = self.down2(out2)
        out3 = self.deep3(out)
        out = self.down3(out3)
        out = self.deep4(out)

        # Decoder
        out = self.up1(out)
        if ((self.skip >= 1 or self.override_SC_init) and drop_sample[0] > 0.2*self.dropout and drop_sample[1] > 1*self.dropout and drop_sample[2] > 5*self.dropout): # or self.override_input):
            out_skip1 = out3 + out
            out = self.deep5(out_skip1)
        else:
            out = self.deep5(out)
        out = self.up2(out)
        if ((self.skip >= 2 or self.override_SC_init) and drop_sample[1] > 1*self.dropout and drop_sample[2] > 5*self.dropout): # or self.override_input):
            out_skip2 = out2 + out
            out = self.deep6(out_skip2)
        else:
            out = self.deep6(out)
        out = self.up3(out)
        if ((self.skip >= 3 or self.override_SC_init) and drop_sample[2] > 5*self.dropout): # or self.override_input):
            out_skip3 = out1 + out
            out = self.deep7(out_skip3)
        else:
            out = self.deep7(out)

        if ((self.method == "DIPRecon" and not self.initDNA and self.config["mu_DIP"] != 1851221) or (self.method == 'DNA' and self.config["mu_DIP"] == 1851221) or (self.method == 'DNA' and self.initDIPRecon)): # 1851221 means ReLU ablation study
            # self.write_current_img_task(out,inside=True) # Write image before ReLU
            out = self.positivity(out)

        return out

    # Define the MSE loss function
    # def mse_loss(self,x,sinogram_corrupt_torch):
    #     return np.mean((np.dot(self.A_torch.detach().numpy(), x)*ravel_np(self.norm_mask.detach().numpy()) - ravel_np(sinogram_corrupt_torch.detach().numpy())*ravel_np(self.norm_mask.detach().numpy()))**2)
    #     return np.mean((np.dot(self.A_torch.detach().numpy(), x) - ravel_np(sinogram_corrupt_torch.detach().numpy()))**2)


    def DIP_loss(self, out, image_corrupt_torch):
        return MSELoss()(out, image_corrupt_torch) # for DIP and DD

    def DIP_loss_end_to_end(self, A, out, sinogram_corrupt_torch):
        Ax = matmul(A,out.ravel())
        # Concatenate the two halves of vector Ax
        Ax = cat([Ax[len(Ax)//2:],Ax[:len(Ax)//2]])

        forward_model = Ax + self.randoms_sinogram + self.scatters_sinogram
        mse_torch = MSELoss()(forward_model.ravel()*self.norm_mask.ravel(), sinogram_corrupt_torch.ravel()*self.norm_mask.ravel()) # Take mask into account
        return mse_torch
        
    def training_step(self, train_batch, batch_idx):
        self.num_total_batch += 1
        if (self.num_total_batch == 0):
            if (self.sub_iter_DIP_already_done_before_training - self.current_epoch == 0):
                self.out_np_all_inputs = zeros((self.several_DIP_inputs,train_batch[0].shape[3],train_batch[0].shape[4]),dtype=float32)
            else:
                self.out_np_all_inputs[:,:,:] = 0 # Do not instantiate a new array for memory efficiency
        loss = 0
        for self.idx_inside_this_batch in range(train_batch[0].shape[0]):
            image_net_input_torch, image_corrupt_torch = train_batch[0][self.idx_inside_this_batch,:,:,:,:],train_batch[1][self.idx_inside_this_batch,:,:,:,:]
            out = self.forward(image_net_input_torch)
            # logging using tensorboard logger
            if (self.end_to_end):
                loss += self.DIP_loss_end_to_end(self.A_torch, out, image_corrupt_torch)
            else:
                loss += self.DIP_loss(out, image_corrupt_torch)
            self.logger.experiment.add_scalar('loss', loss,self.current_epoch)

            try:
                # self.out_np[self.idx_inside_this_batch,:,:] = out.detach().numpy()[0,0,:,:]
                self.out_np = out.detach().numpy()[0,0,:,:]
            except:
                # self.out_np[self.idx_inside_this_batch,:,:] = out.cpu().detach().numpy()[0,0,:,:]
                self.out_np = out.cpu().detach().numpy()[0,0,:,:]

            if (self.num_total_batch != self.several_DIP_inputs - 1):
                if (self.write_current_img_mode):
                    self.write_current_img(out,batch_idx)
            else:
                # if (self.write_current_img_mode):
                #     self.write_current_img(out,batch_idx)
                self.end_epoch = True

        try:
            self.out_np_all_inputs[self.num_total_batch,:,:] = out.detach().numpy()[0,0,:,:]
        except:
            self.out_np_all_inputs[self.num_total_batch,:,:] = out.cpu().detach().numpy()[0,0,:,:]        

        # For L-BFGS
        end_epoch_LBFGS = True
        if (self.opti_DIP == "LBFGS"):
            end_epoch_LBFGS = False
            self.SUCCESS = False
            self.counter_inside_epoch += 1
            self.log("SUCCESS", int(self.classMV.SUCCESS))
            if (self.counter_inside_epoch == 10): # number of max iter in lbfgs opti
                end_epoch_LBFGS = True
                self.counter_inside_epoch = 0


        # Save image over epochs
        if (end_epoch_LBFGS):
            if (self.write_current_img_mode):
                self.write_current_img(out)
        # Monitor learning rate across iterations
        self.monitor_lr_func(out,image_corrupt_torch)

        # MV
        if (self.DIP_early_stopping):
            if (end_epoch_LBFGS):
                if (self.num_total_batch == self.several_DIP_inputs - 1):
                    self.SUCCESS = self.classMV.SUCCESS
                    self.log("SUCCESS", int(self.SUCCESS))
                    self.SUCCESS, self.VAR_recon, self.MSE_MV, self.PSNR_MV, self.SSIM_MV, self.epochStar, self.patienceNumber = self.classMV.run_MV(out.detach().numpy(),self.config, self.current_epoch)
                    self.epochStar = self.classMV.epochStar
            
        # Increment number of iterations since beginnning of DNA
        if (self.end_epoch): # We looped over all images of the batch
            self.sub_iter_DIP_already_done += 1
            self.sub_iter_DIP_this_global_it += 1
        if (self.several_DIP_inputs > 1): # If several inputs, save MR forward
            if (self.num_total_batch == self.several_DIP_inputs - 1):
                if ((self.current_epoch == self.sub_iter_DIP + self.sub_iter_DIP_already_done_before_training - 1)):
                    batch_idx = "MR_forward"
                    self.save_img(self.out_np_all_inputs[0,:,:], self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + 'DIP' + format(self.global_it) + '_epoch=' + format(self.current_epoch) + ('_batchidx=' + format(batch_idx))*(batch_idx!=-1) + '.img') # The saved images are not destandardized !!!!!! Do it when showing images in tensorboard
                if (self.DIP_early_stopping):
                    if (self.SUCCESS):
                        batch_idx = "MR_forward"
                        self.save_img(self.out_np_all_inputs[0,:,:], self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + 'DIP' + format(self.global_it) + '_epoch=' + format(self.current_epoch) + ('_batchidx=' + format(batch_idx))*(batch_idx!=-1) + '.img') # The saved images are not destandardized !!!!!! Do it when showing images in tensorboard
        if (self.end_epoch):
            self.num_total_batch = -1
            self.end_epoch = False
        
        return loss

    def configure_optimizers(self):
        # Optimization algorithm according to command line

        """
        Optimization of the DNN with SGLD
        """

        if (self.opti_DIP == 'Adam'):
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=5E-8) # Optimizing using Adam
            # optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=(0,0)) # Optimizing using Adam
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

    def write_current_img(self,out,batch_idx=-1):
        if (self.all_images_DIP == "False"):
            if ((self.current_epoch%(self.sub_iter_DIP // 10) == (self.sub_iter_DIP // 10) -1)):
                self.write_current_img_task(out,batch_idx=batch_idx)
        elif (self.all_images_DIP == "True"):
            self.write_current_img_task(out,batch_idx=batch_idx)
        elif (self.all_images_DIP == "Unique" and not self.DIP_early_stopping): # Write last computed image
            if (self.current_epoch == self.sub_iter_DIP + self.sub_iter_DIP_already_done_before_training - 1):
                self.write_current_img_task(out,batch_idx=batch_idx)

    def write_current_img_task(self,out,inside=False,batch_idx=-1):
        print("self.current_epoch",self.current_epoch)
        if (inside):
            print("save before ReLU here")
            # self.save_img(out_np, self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/beforeReLU_' + 'DIP' + format(self.global_it) + '_epoch=' + format(self.current_epoch + self.last_iter) + '.img') # The saved images are not destandardized !!!!!! Do it when showing images in tensorboard
        else:
            self.save_img(self.out_np, self.subroot_phantom +'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + 'DIP' + format(self.global_it) + '_epoch=' + format(self.current_epoch) + ('_batchidx=' + format(batch_idx))*(batch_idx!=-1) + '.img') # The saved images are not destandardized !!!!!! Do it when showing images in tensorboard
                            
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

    def monitor_lr_func(self,out,image_corrupt_torch):
        if ("monitor_lr" not in self.config):
            self.monitor_lr = False
        else:
            self.monitor_lr = self.config["monitor_lr"]
        if (self.monitor_lr):
            out_descale_np = self.descale_imag(clone(out),self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)

            image_corrupt_np = self.descale_imag(image_corrupt_torch,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)

            # self.subroot = self.root + '/data/Algo/' # Directory root
            self.phantom = self.config["image"]

            self.PETImage_shape_str = self.read_input_dim(self.subroot + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.hdr')
            self.PETImage_shape = self.input_dim_str_to_list(self.PETImage_shape_str)

            self.phantom_ROI = self.get_phantom_ROI(self.phantom)


            mean_inside = mean_np(out_descale_np * self.phantom_ROI) / mean_np(image_corrupt_np * self.phantom_ROI)
            self.mean_inside_list.append(mean_inside)

            if (self.current_epoch >= 2):
                alpha_ema_lr = 0.1
                self.ema_lr.append((1-alpha_ema_lr) * self.ema_lr[self.current_epoch-1] + alpha_ema_lr * self.mean_inside_list[self.current_epoch])

                # print(self.ema_lr[self.current_epoch])

                if (sign(self.ema_lr[self.current_epoch] - self.ema_lr[self.current_epoch - 1]) != sign(self.ema_lr[self.current_epoch - 1] - self.ema_lr[self.current_epoch - 2])):
                    # if (self.lr > 1e-5): # Minimum lr value to 1e-5, does not need to better stability
                    self.lr /= 2
                    print(self.lr)
                    print("chaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaange lrrrrrrrrrrrrrrrrrrrrrrrrrrr")