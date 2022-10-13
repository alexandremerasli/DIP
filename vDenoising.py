## Python libraries

# Pytorch
import torch
import pytorch_lightning as pl

# Useful
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# Set random seed if asked (for random input here)
if (os.path.isfile(os.getcwd() + "/seed.txt")):
    with open(os.getcwd() + "/seed.txt", 'r') as file:
        random_seed = file.read().rstrip()
    if (eval(random_seed)):
        np.random.seed(1)

# Local files to import
from vGeneral import vGeneral

from models.DIP_2D import DIP_2D # DIP
from models.DIP_3D import DIP_3D # DIP
from models.VAE_DIP_2D import VAE_DIP_2D # DIP vae
from models.DD_2D import DD_2D # DD
from models.DD_AE_2D import DD_AE_2D # DD adding encoder part

import abc
class vDenoising(vGeneral):
    @abc.abstractmethod
    def __init__(self,config, *args, **kwargs):
        print('__init__')

    def initializeSpecific(self,config,root, *args, **kwargs):
        self.createDirectoryAndConfigFile(config)
        # Specific hyperparameters for reconstruction module (Do it here to have raytune config hyperparameters selection)
        if (config["net"] == "DD" or config["net"] == "DD_AE"):
            self.d_DD = config["d_DD"]
            self.k_DD = config["k_DD"]
        if (config["method"] == "nested" or config["method"] == "Gong"):
            self.input = config["input"]
            self.scaling_input = config["scaling"]
            # Loading DIP input
            # Creating random image input for DIP while we do not have CT, but need to be removed after
            self.create_input(self.net,self.PETImage_shape,config,self.subroot_data) # to be removed when CT will be used instead of random input. DO NOT PUT IT IN BLOCK 2 !!!
            # Loading DIP input (we do not have CT-map, so random image created in block 1)
            self.image_net_input = self.load_input(self.net,self.PETImage_shape,self.subroot_data) # Scaling of network input. DO NOT CREATE RANDOM INPUT IN BLOCK 2 !!! ONLY AT THE BEGINNING, IN BLOCK 1    
            #image_atn = fijii_np(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '_atn.raw',shape=(self.PETImage_shape),type='<f')
            #write_image_tensorboard(writer,image_atn,"Attenuation map (FULL CONTRAST)",self.suffix,image_gt,0,full_contrast=True) # Attenuation map in tensorboard
            image_net_input_scale = self.rescale_imag(self.image_net_input,self.scaling_input)[0] # Rescale of network input
            # DIP input image, numpy --> torch
            self.image_net_input_torch = torch.Tensor(image_net_input_scale)
            # Adding dimensions to fit network architecture
            if (self.net == 'DIP' or self.net == 'DIP_VAE' or self.net == 'DD_AE'): # For autoencoders structure
                self.image_net_input_torch = self.image_net_input_torch.view(1,1,self.PETImage_shape[0],self.PETImage_shape[1],self.PETImage_shape[2])
                if (self.PETImage_shape[2] == 1): # if 3D but with dim3 = 1 -> 2D
                        self.image_net_input_torch = self.image_net_input_torch[:,:,:,:,0]
            elif (self.net == 'DD'):
                    input_size_DD = int(self.PETImage_shape[0] / (2**config["d_DD"])) # if original Deep Decoder (i.e. only with decoder part)
                    self.image_net_input_torch = self.image_net_input_torch.view(1,config["k_DD"],input_size_DD,input_size_DD) # For Deep Decoder, if original Deep Decoder (i.e. only with decoder part)
            torch.save(self.image_net_input_torch,self.subroot_data + 'Data/initialization/image_' + self.net + '_input_torch.pt')

    def train_process(self, param1_scale_im_corrupt, param2_scale_im_corrupt, scaling_input, suffix, config, finetuning, processing_unit, sub_iter_DIP, method, global_it, image_net_input_torch, image_corrupt_torch, net, PETImage_shape, experiment, checkpoint_simple_path, name_run, subroot, all_images_DIP):
        # Implements Dataset
        train_dataset = torch.utils.data.TensorDataset(image_net_input_torch, image_corrupt_torch)
        # train_dataset = Dataset(image_net_input_torch, image_corrupt_torch)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1) # num_workers is 0 by default, which means the training process will work sequentially inside the main process

        # Choose network architecture as model
        model, model_class = self.choose_net(net, param1_scale_im_corrupt, param2_scale_im_corrupt, scaling_input, config, method, all_images_DIP, global_it, PETImage_shape)
        
        #checkpoint_simple_path = 'runs/' # To log loss in tensorboard thanks to Logger
        checkpoint_simple_path_exp = subroot+'Block2/' + self.suffix + '/checkpoint/'+format(experiment)  + '/' + suffix + '/'

        model = self.load_model(param1_scale_im_corrupt, param2_scale_im_corrupt, scaling_input, image_net_input_torch, config, finetuning, global_it, model, model_class, method, all_images_DIP, checkpoint_simple_path_exp, training=True)
        
        #from torchsummary import summary
        #summary(model, input_size=(1,112,112,59))

        # Start training
        print('Starting optimization, iteration',global_it)
        trainer = self.create_pl_trainer(finetuning, processing_unit, sub_iter_DIP, global_it, net, checkpoint_simple_path, experiment, checkpoint_simple_path_exp,name=name_run)

        trainer.fit(model, train_dataloader)

        return model

    def create_pl_trainer(self,finetuning, processing_unit, sub_iter_DIP, global_it, net, checkpoint_simple_path, experiment, checkpoint_simple_path_exp, name=''):
        from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
        TuneReportCheckpointCallback

        tuning_callback = TuneReportCallback({"loss": "val_loss"}, on="validation_end")
        accelerator = None
        if (processing_unit == 'CPU'): # use cpus and no gpu
            gpus = 0
        elif (processing_unit == 'GPU' or processing_unit == 'both'): # use all available gpus, no cpu (pytorch lightning does not handle cpus and gpus at the same time)
            gpus = -1
            #if (torch.cuda.device_count() > 1):
            #    accelerator = 'dp'


        # Early stopping callback
        from pytorch_lightning.callbacks.early_stopping import EarlyStopping
        early_stopping_callback = EarlyStopping(monitor="SUCCESS", mode="max",stopping_threshold=0.9,patience=np.inf) # SUCCESS will be 1 when ES if found, which is greater than stopping_threshold = 0.9

        print("global_it",global_it)
        if (global_it == -1): # Beginning nested or Gong in block2. For first epoch, change number of epochs to sub_iter_DIP_initial for Gong
            print(str(self.sub_iter_DIP_initial) + " initial iterations for Gong")
            self.sub_iter_DIP = self.sub_iter_DIP_initial
            sub_iter_DIP = self.sub_iter_DIP
        if (finetuning == 'False'): # Do not save and use checkpoints (still save hparams and event files for now ...)
            logger = pl.loggers.TensorBoardLogger(save_dir=checkpoint_simple_path, version=format(experiment), name=name) # Store checkpoints in checkpoint_simple_path path
            #checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_simple_path_exp, save_top_k=0, save_weights_only=True) # Do not save any checkpoint (save_top_k = 0)
            checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_simple_path_exp, save_last=True, save_top_k=0) # Only save last checkpoint as last.ckpt (save_last = True), do not save checkpoint at each epoch (save_top_k = 0). We do not use it a priori, except in post reconstruction to initialize
            trainer = pl.Trainer(max_epochs=sub_iter_DIP,log_every_n_steps=1, callbacks=[checkpoint_callback, tuning_callback, early_stopping_callback], logger=logger,gpus=gpus, accelerator=accelerator, profiler="simple")
        else:
            if (finetuning == 'last'): # last model saved in checkpoint
                # Checkpoints pl variables
                logger = pl.loggers.TensorBoardLogger(save_dir=checkpoint_simple_path, version=format(experiment), name=name) # Store checkpoints in checkpoint_simple_path path
                checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_simple_path_exp, save_last=True, save_top_k=0) # Only save last checkpoint as last.ckpt (save_last = True), do not save checkpoint at each epoch (save_top_k = 0)
                trainer = pl.Trainer(max_epochs=sub_iter_DIP,log_every_n_steps=1, logger=logger, callbacks=[checkpoint_callback, tuning_callback, early_stopping_callback],gpus=gpus, accelerator=accelerator,log_gpu_memory="all") # Prepare trainer model with callback to save checkpoint        
            if (finetuning == 'best'): # best model saved in checkpoint
                # Checkpoints pl variables
                logger = pl.loggers.TensorBoardLogger(save_dir=checkpoint_simple_path, version=format(experiment), name=name) # Store checkpoints in checkpoint_simple_path path
                checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_simple_path_exp, filename = 'best_loss', monitor='loss_monitor', save_top_k=1) # Save best checkpoint (save_top_k = 1) (according to minimum loss (monitor)) as best_loss.ckpt
                trainer = pl.Trainer(max_epochs=sub_iter_DIP,log_every_n_steps=1, logger=logger, callbacks=[checkpoint_callback, tuning_callback, early_stopping_callback],gpus=gpus, accelerator=accelerator, profiler="simple") # Prepare trainer model with callback to save checkpoint

        return trainer

    def create_input(self,net,PETImage_shape,config,subroot): #CT map for high-count data, but not CT yet...
        constant_uniform = 1
        if (self.FLTNB == 'float'):
            type = 'float32'
        elif (self.FLTNB == 'double'):
            type = 'float64'
        if (net == 'DIP' or net == 'DIP_VAE'):
            if config["input"] == "random":
                im_input = np.random.normal(0,1,PETImage_shape[0]*PETImage_shape[1]*PETImage_shape[2]).astype(type) # initializing input image with random image (for DIP)
            elif config["input"] == "uniform":
                im_input = constant_uniform*np.ones((PETImage_shape[0]*PETImage_shape[1]*PETImage_shape[2])).astype(type) # initializing input image with random image (for DIP)
            else:
                return "CT input, do not need to create input"
            im_input = im_input.reshape(PETImage_shape) # reshaping (for DIP)
        else:
            if (net == 'DD'):
                input_size_DD = int(PETImage_shape[0] / (2**config["d_DD"])) # if original Deep Decoder (i.e. only with decoder part)
                if config["input"] == "random":
                    im_input = np.random.normal(0,1,config["k_DD"]*input_size_DD*input_size_DD).astype(type) # initializing input image with random image (for Deep Decoder) # if original Deep Decoder (i.e. only with decoder part)
                elif config["input"] == "uniform":
                    im_input = constant_uniform*np.ones((config["k_DD"],input_size_DD,input_size_DD)).astype(type) # initializing input image with random image (for Deep Decoder) # if original Deep Decoder (i.e. only with decoder part)
                else:
                    return "CT input, do not need to create input"
                im_input = im_input.reshape(config["k_DD"],input_size_DD,input_size_DD) # reshaping (for Deep Decoder) # if original Deep Decoder (i.e. only with decoder part)
                
            elif (net == 'DD_AE'):
                if config["input"] == "random":
                    im_input = np.random.normal(0,1,PETImage_shape[0]*PETImage_shape[1]*PETImage_shape[2]).astype(type) # initializing input image with random image (for Deep Decoder) # if auto encoder based on Deep Decoder
                elif config["input"] == "uniform":
                    im_input = constant_uniform*np.ones((PETImage_shape[0]*PETImage_shape[1]*PETImage_shape[2])).astype(type) # initializing input image with random image (for Deep Decoder) # if auto encoder based on Deep Decoder
                else:
                    return "CT input, do not need to create input"
                im_input = im_input.reshape(PETImage_shape[0],PETImage_shape[1],PETImage_shape[2]) # reshaping (for Deep Decoder) # if auto encoder based on Deep Decoder
        if config["input"] == "random":
            file_path = (subroot+'Data/initialization/random_input_' + net + '.img')
        elif config["input"] == "uniform":
            file_path = (subroot+'Data/initialization/uniform_input_' + net + '.img')
        self.save_img(im_input,file_path)

    def load_input(self,net,PETImage_shape,subroot):
        if self.input == "random":
            file_path = (subroot+'Data/initialization/random_input_' + net + '.img')
        elif self.input == "CT":
            file_path = (subroot+'Data/database_v2/' + self.phantom + '/' + self.phantom + '_atn.raw') #CT map, but not CT yet, attenuation for now...
        elif self.input == "BSREM":
            file_path = (subroot+'Data/initialization/BSREM_it30_REF_cropped.img') #
        elif self.input == "uniform":
            file_path = (subroot+'Data/initialization/uniform_input_' + net + '.img')
        if (net == 'DD'):
            input_size_DD = int(PETImage_shape[0] / (2**self.d_DD)) # if original Deep Decoder (i.e. only with decoder part)
            PETImage_shape = (self.k_DD,input_size_DD,input_size_DD) # if original Deep Decoder (i.e. only with decoder part)
        #elif (net == 'DD_AE'):   
        #    PETImage_shape = (PETImage_shape[0],PETImage_shape[1],PETImage_shape[2]) # if auto encoder based on Deep Decoder

        if (self.input == 'CT' and self.net != 'DD'):
            type = '<f'
        else:
            type = None

        im_input = self.fijii_np(file_path, shape=(PETImage_shape),type=type) # Load input of the DNN (CT image)
        return im_input


    def load_model(self,param1_scale_im_corrupt, param2_scale_im_corrupt, scaling_input, image_net_input_torch, config, finetuning, global_it, model, model_class, method, all_images_DIP, checkpoint_simple_path_exp, training):
        if (finetuning == 'last'): # last model saved in checkpoint
            if (global_it > 0): # if model has already been trained
                model = model_class.load_from_checkpoint(os.path.join(checkpoint_simple_path_exp,'last.ckpt'), config=config, method=method, all_images_DIP = all_images_DIP, global_it = global_it, param1_scale_im_corrupt=param1_scale_im_corrupt, param2_scale_im_corrupt=param2_scale_im_corrupt, scaling_input=scaling_input,root=self.root,subroot=self.subroot, fixed_hyperparameters_list=self.fixed_hyperparameters_list, hyperparameters_list=self.hyperparameters_list, debug=self.debug) # Load previous model in checkpoint        
        # if (global_it == 0):
            # DD finetuning, k=32, d=6
            #model = model_class.load_from_checkpoint(os.path.join(subroot,'high_statistics.ckpt'), config=config) # Load model coming from high statistics computation (normally coming from finetuning with supervised learning)
            #from torch.utils.tensorboard import SummaryWriter
            #writer = SummaryWriter()
            #out = model(image_net_input_torch)
            #write_image_tensorboard(writer,out.detach().numpy(),"high statistics output)",suffix,image_gt) # Showing all corrupted images with same contrast to compare them together
            #write_image_tensorboard(writer,out.detach().numpy(),"high statistics (" + "output, suffix,image_gt,FULL CONTRAST)",0,full_contrast=True) # Showing each corrupted image with contrast = 1
        
            # Set first network iterations to have convergence, as if we do post processing
            # model = model_class.load_from_checkpoint(os.path.join(subroot,'post_reco'+net+'.ckpt'), config=config) # Load model coming from high statistics computation (normally coming from finetuning with supervised learning)

        if (finetuning == 'best'): # best model saved in checkpoint
            if (global_it > 0): # if model has already been trained
                model = model_class.load_from_checkpoint(os.path.join(checkpoint_simple_path_exp,'best_loss.ckpt'), config=config,method=method, all_images_DIP = all_images_DIP) # Load best model in checkpoint
            #if (global_it == 0):
            # DD finetuning, k=32, d=6
                #model = model_class.load_from_checkpoint(os.path.join(subroot,'high_statistics.ckpt'), config=config) # Load model coming from high statistics computation (normally coming from finetuning with supervised learning)
            if (training):
                os.system('rm -rf ' + checkpoint_simple_path_exp + '/best_loss.ckpt') # Otherwise, pl will store checkpoint with version in filename
        
        return model

    def runComputation(self,config,root):
        # Scaling of x_label image
        image_corrupt_input_scale,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt = self.rescale_imag(self.image_corrupt,self.scaling_input) # Scaling of x_label image

        # Corrupted image x_label, numpy --> torch float32
        self.image_corrupt_torch = torch.Tensor(image_corrupt_input_scale)
        # Adding dimensions to fit network architecture
        self.image_corrupt_torch = self.image_corrupt_torch.view(1,1,self.PETImage_shape[0],self.PETImage_shape[1],self.PETImage_shape[2])
        if (self.PETImage_shape[2] == 1): # if 3D but with dim3 = 1 -> 2D
            self.image_corrupt_torch = self.image_corrupt_torch[:,:,:,:,0]
        # Training model with sub_iter_DIP iterations
        model = self.train_process(self.param1_scale_im_corrupt, self.param2_scale_im_corrupt, self.scaling_input, self.suffix, config, self.finetuning, self.processing_unit, self.sub_iter_DIP, self.method, self.global_it, self.image_net_input_torch, self.image_corrupt_torch, self.net, self.PETImage_shape, self.experiment, self.checkpoint_simple_path, self.name_run, self.subroot, self.all_images_DIP) # Not useful to make iterations, we just want to initialize writer. global_it must be set to -1, otherwise seeking for a checkpoint file...
        if (self.net == 'DIP_VAE'):
            out, mu, logvar, z = model(self.image_net_input_torch)
        else:
            out = model(self.image_net_input_torch)

        self.DIP_early_stopping = model.DIP_early_stopping
        if self.DIP_early_stopping:
            self.epochStar = model.epochStar
            self.windowSize = model.windowSize
            self.patienceNumber = model.patienceNumber
            self.VAR_recon = model.VAR_recon
            self.MSE_WMV = model.MSE_WMV
            self.PSNR_WMV = model.PSNR_WMV
            self.SSIM_WMV = model.SSIM_WMV
            self.SUCCESS = model.SUCCESS
            if (self.SUCCESS and self.epochStar!= self.sub_iter_DIP - self.patienceNumber): # ES point is reached
                self.sub_iter_DIP = self.epochStar + self.patienceNumber + 1

        '''
        # Descaling like at the beginning
        out_descale = self.descale_imag(out,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)
        # Saving image output
        self.save_img(out_descale, self.net_outputs_path)
        '''

        # Write descaled images in files
        if (self.all_images_DIP == "True"):
            epoch_values = np.arange(0,self.sub_iter_DIP)
        elif (self.all_images_DIP == "False"):
            epoch_values = np.arange(0,self.sub_iter_DIP,max(self.sub_iter_DIP//10,1))
        elif (self.all_images_DIP == "Last"):
            epoch_values = np.array([self.sub_iter_DIP-1])

        for epoch in epoch_values:
            net_outputs_path = self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + format(self.global_it) + '_epoch=' + format(epoch) + '.img'
            out = self.fijii_np(net_outputs_path,shape=(self.PETImage_shape),type='<f')
            out = torch.from_numpy(out)
            # Descale like at the beginning
            out_descale = self.descale_imag(out,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)
            #'''
            # Saving image output
            net_outputs_path = self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + format(self.global_it) + '_epoch=' + format(epoch) + '.img'
            self.save_img(out_descale, net_outputs_path)
            # Squeeze image by loading it
            out_descale = self.fijii_np(net_outputs_path,shape=(self.PETImage_shape),type='<f') # loading DIP output
            # Saving (now DESCALED) image output
            self.save_img(out_descale, net_outputs_path)

            '''
            # Compute IR metric (different from others with several replicates)
            classResults.compute_IR_bkg(self.PETImage_shape,out_descale,epoch,classResults.IR_bkg_recon,self.phantom)
            classResults.writer.add_scalar('Image roughness in the background (best : 0)', classResults.IR_bkg_recon[epoch], epoch+1)
            # Write images over epochs
            classResults.writeEndImagesAndMetrics(epoch,self.sub_iter_DIP,self.PETImage_shape,out_descale,self.suffix,self.phantom,self.net,pet_algo="to fit",iteration_name="(post reconstruction)",all_images_DIP=all_images_DIP)
            '''


    def choose_net(self, net, param1_scale_im_corrupt, param2_scale_im_corrupt, scaling_input, config, method, all_images_DIP, global_it, PETImage_shape):
        if (net == 'DIP'): # Loading DIP architecture
            if(PETImage_shape[2] == 1): # 2D
                model = DIP_2D(param1_scale_im_corrupt, param2_scale_im_corrupt, scaling_input, self.config,self.root,self.subroot,method,all_images_DIP,global_it, self.fixed_hyperparameters_list, self.hyperparameters_list, self.debug)
                model_class = DIP_2D
            else: # 3D
                model = DIP_3D(param1_scale_im_corrupt, param2_scale_im_corrupt, scaling_input, self.config,self.root,self.subroot,method,all_images_DIP,global_it, self.fixed_hyperparameters_list, self.hyperparameters_list, self.debug)
                model_class = DIP_3D
        elif (net == 'DIP_VAE'): # Loading DIP VAE architecture
            model = VAE_DIP_2D(config)
            model_class = VAE_DIP_2D
        elif (net == 'DD'): # Loading Deep Decoder architecture
                model = DD_2D(config,method)
                model_class = DD_2D
        elif (net == 'DD_AE'): # Loading Deep Decoder based autoencoder architecture
            model = DD_AE_2D(config) 
            model_class = DD_AE_2D
        return model, model_class
    
    def generate_nn_output(self, net, param1_scale_im_corrupt, param2_scale_im_corrupt, scaling_input, config, method, image_net_input_torch, PETImage_shape, finetuning, global_it, experiment, suffix, subroot, all_images_DIP):
        # Loading using previous model
        model, model_class = self.choose_net(net, config, method, all_images_DIP, global_it, PETImage_shape)
        checkpoint_simple_path_exp = subroot+'Block2/' + self.suffix + '/checkpoint/'+format(experiment) + '/' + suffix + '/'
        model = self.load_model(param1_scale_im_corrupt, param2_scale_im_corrupt, scaling_input, image_net_input_torch, config, finetuning, global_it, model, model_class, method, all_images_DIP, checkpoint_simple_path_exp, training=False)

        # Compute output image
        out, mu, logvar, z = model(image_net_input_torch)

        # Loading X_label from block1 to destandardize NN output
        image_corrupt = self.fijii_np(subroot+'Block2/' + self.suffix + '/x_label/' + format(experiment)+'/'+ format(global_it - 1) +'_x_label' + suffix + '.img',shape=(PETImage_shape))
        image_corrupt_scale,param1_scale_im_corrupt,param2_scale_im_corrupt = self.rescale_imag(image_corrupt)

        # Reverse scaling like at the beginning and add it to list of samples
        out_descale = self.descale_imag(out,param1_scale_im_corrupt,param2_scale_im_corrupt,config["scaling"])
        return out_descale