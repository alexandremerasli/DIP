## Python libraries

# Pytorch
import torch

import pytorch_lightning as pl

# Useful
import numpy as np
import os
if (os.path.isfile(os.getcwd() + "/seed.txt")):
    with open(os.getcwd() + "/seed.txt", 'r') as file:
        random_seed = file.read().rstrip()
    if (eval(random_seed)):
        np.random.seed(2)

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
    def __init__(self,config,root):
        print('__init__')

    def initializeSpecific(self,fixed_config,hyperparameters_config,root):
        self.createDirectoryAndConfigFile(hyperparameters_config)
        # Specific hyperparameters for reconstruction module (Do it here to have raytune hyperparameters_config hyperparameters selection)
        if (fixed_config["net"] == "DD" or fixed_config["net"] == "DD_AE"):
            self.d_DD = hyperparameters_config["d_DD"]
            self.k_DD = hyperparameters_config["k_DD"]
        if (fixed_config["method"] == "nested" or fixed_config["method"] == "Gong"):
            self.input = hyperparameters_config["input"]
            self.scaling_input = hyperparameters_config["scaling"]
            # Loading DIP input
            # Creating random image input for DIP while we do not have CT, but need to be removed after
            self.create_input(self.net,self.PETImage_shape,hyperparameters_config,self.subroot_data) # to be removed when CT will be used instead of random input. DO NOT PUT IT IN BLOCK 2 !!!
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
                if (len(self.image_net_input_torch.shape) == 5): # if 3D but with dim3 = 1 -> 2D
                    self.image_net_input_torch = self.image_net_input_torch[:,:,:,:,0]
            elif (self.net == 'DD'):
                    input_size_DD = int(self.PETImage_shape[0] / (2**hyperparameters_config["d_DD"])) # if original Deep Decoder (i.e. only with decoder part)
                    self.image_net_input_torch = self.image_net_input_torch.view(1,hyperparameters_config["k_DD"],input_size_DD,input_size_DD) # For Deep Decoder, if original Deep Decoder (i.e. only with decoder part)
            torch.save(self.image_net_input_torch,self.subroot_data + 'Data/initialization/image_' + self.net + '_input_torch.pt')

    def train_process(self, suffix, hyperparameters_config, finetuning, processing_unit, sub_iter_DIP, method, admm_it, image_net_input_torch, image_corrupt_torch, net, PETImage_shape, experiment, checkpoint_simple_path, name_run, subroot, all_images = True):
        # Implements Dataset
        train_dataset = torch.utils.data.TensorDataset(image_net_input_torch, image_corrupt_torch)
        # train_dataset = Dataset(image_net_input_torch, image_corrupt_torch)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1) # num_workers is 0 by default, which means the training process will work sequentially inside the main process

        # Choose network architecture as model
        model, model_class = self.choose_net(net, hyperparameters_config, method, all_images, admm_it, PETImage_shape)

        #checkpoint_simple_path = 'runs/' # To log loss in tensorboard thanks to Logger
        checkpoint_simple_path_exp = subroot+'Block2/checkpoint/'+format(experiment)  + '/' + suffix + '/'

        model = self.load_model(image_net_input_torch, hyperparameters_config, finetuning, admm_it, model, model_class, method, all_images, checkpoint_simple_path_exp, training=True)

        # Start training
        print('Starting optimization, iteration',admm_it)
        trainer = self.create_pl_trainer(finetuning, processing_unit, sub_iter_DIP, admm_it, net, checkpoint_simple_path, experiment, checkpoint_simple_path_exp,name=name_run)

        trainer.fit(model, train_dataloader)

        return model

    def create_pl_trainer(self,finetuning, processing_unit, sub_iter_DIP, admm_it, net, checkpoint_simple_path, experiment, checkpoint_simple_path_exp, name=''):
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

        print("admm_iiiiiiiiiiiiit",admm_it)

        if (admm_it == 0): # First ADMM iteration in block 1
            #sub_iter_DIP = 1000 if net.startswith('DD') else 200
            #sub_iter_DIP = 100 if net.startswith('DD') else 100
            print(admm_it)
        elif (admm_it == -1): # First ADMM iteration in block2 post reconstruction
            print("admmm_it = -1 must be remooooooooooooooooooooooooooooooooooooooooooooooooooooooooved")
            if (self.method != 'Gong'):
                sub_iter_DIP = 1 if net.startswith('DD') else 1
                import sys
                sys.exit()
            else:
                #print('whaaaaaaaaaaaaaaaaaaat ????????????')
                print("300 initial iterations for Gong")
                #import sys
                #sys.exit()
                sub_iter_DIP = 100
        if (finetuning == 'False'): # Do not save and use checkpoints (still save hparams and event files for now ...)
            logger = pl.loggers.TensorBoardLogger(save_dir=checkpoint_simple_path, version=format(experiment), name=name) # Store checkpoints in checkpoint_simple_path path
            #checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_simple_path_exp, save_top_k=0, save_weights_only=True) # Do not save any checkpoint (save_top_k = 0)
            checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_simple_path_exp, save_last=True, save_top_k=0) # Only save last checkpoint as last.ckpt (save_last = True), do not save checkpoint at each epoch (save_top_k = 0). We do not use it a priori, except in post reconstruction to initialize
            trainer = pl.Trainer(max_epochs=sub_iter_DIP,log_every_n_steps=1, callbacks=[checkpoint_callback, tuning_callback], logger=logger,gpus=gpus, accelerator=accelerator, profiler="simple")
        else:
            if (finetuning == 'last'): # last model saved in checkpoint
                # Checkpoints pl variables
                logger = pl.loggers.TensorBoardLogger(save_dir=checkpoint_simple_path, version=format(experiment), name=name) # Store checkpoints in checkpoint_simple_path path
                checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_simple_path_exp, save_last=True, save_top_k=0) # Only save last checkpoint as last.ckpt (save_last = True), do not save checkpoint at each epoch (save_top_k = 0)
                trainer = pl.Trainer(max_epochs=sub_iter_DIP,log_every_n_steps=1, logger=logger, callbacks=[checkpoint_callback, tuning_callback],gpus=gpus, accelerator=accelerator,log_gpu_memory="all") # Prepare trainer model with callback to save checkpoint        
            if (finetuning == 'best'): # best model saved in checkpoint
                # Checkpoints pl variables
                logger = pl.loggers.TensorBoardLogger(save_dir=checkpoint_simple_path, version=format(experiment), name=name) # Store checkpoints in checkpoint_simple_path path
                checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_simple_path_exp, filename = 'best_loss', monitor='loss_monitor', save_top_k=1) # Save best checkpoint (save_top_k = 1) (according to minimum loss (monitor)) as best_loss.ckpt
                trainer = pl.Trainer(max_epochs=sub_iter_DIP,log_every_n_steps=1, logger=logger, callbacks=[checkpoint_callback, tuning_callback],gpus=gpus, accelerator=accelerator, profiler="simple") # Prepare trainer model with callback to save checkpoint

        return trainer

    def create_input(self,net,PETImage_shape,hyperparameters_config,subroot): #CT map for high-count data, but not CT yet...
        constant_uniform = 1
        if (self.FLTNB == 'float'):
            type = 'float32'
        elif (self.FLTNB == 'double'):
            type = 'float64'
        if (net == 'DIP' or net == 'DIP_VAE'):
            if hyperparameters_config["input"] == "random":
                im_input = np.random.normal(0,1,PETImage_shape[0]*PETImage_shape[1]*PETImage_shape[2]).astype(type) # initializing input image with random image (for DIP)
            elif hyperparameters_config["input"] == "uniform":
                im_input = constant_uniform*np.ones((PETImage_shape[0]*PETImage_shape[1]*PETImage_shape[2])).astype(type) # initializing input image with random image (for DIP)
            else:
                return "CT input, do not need to create input"
            im_input = im_input.reshape(PETImage_shape) # reshaping (for DIP)
        else:
            if (net == 'DD'):
                input_size_DD = int(PETImage_shape[0] / (2**hyperparameters_config["d_DD"])) # if original Deep Decoder (i.e. only with decoder part)
                if hyperparameters_config["input"] == "random":
                    im_input = np.random.normal(0,1,hyperparameters_config["k_DD"]*input_size_DD*input_size_DD).astype(type) # initializing input image with random image (for Deep Decoder) # if original Deep Decoder (i.e. only with decoder part)
                elif hyperparameters_config["input"] == "uniform":
                    im_input = constant_uniform*np.ones((hyperparameters_config["k_DD"],input_size_DD,input_size_DD)).astype(type) # initializing input image with random image (for Deep Decoder) # if original Deep Decoder (i.e. only with decoder part)
                else:
                    return "CT input, do not need to create input"
                im_input = im_input.reshape(hyperparameters_config["k_DD"],input_size_DD,input_size_DD) # reshaping (for Deep Decoder) # if original Deep Decoder (i.e. only with decoder part)
                
            elif (net == 'DD_AE'):
                if hyperparameters_config["input"] == "random":
                    im_input = np.random.normal(0,1,PETImage_shape[0]*PETImage_shape[1]*PETImage_shape[2]).astype(type) # initializing input image with random image (for Deep Decoder) # if auto encoder based on Deep Decoder
                elif hyperparameters_config["input"] == "uniform":
                    im_input = constant_uniform*np.ones((PETImage_shape[0]*PETImage_shape[1]*PETImage_shape[2])).astype(type) # initializing input image with random image (for Deep Decoder) # if auto encoder based on Deep Decoder
                else:
                    return "CT input, do not need to create input"
                im_input = im_input.reshape(PETImage_shape[0],PETImage_shape[1],PETImage_shape[2]) # reshaping (for Deep Decoder) # if auto encoder based on Deep Decoder
        if hyperparameters_config["input"] == "random":
            file_path = (subroot+'Data/initialization/random_input_' + net + '.img')
        elif hyperparameters_config["input"] == "uniform":
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


    def load_model(self,image_net_input_torch, hyperparameters_config, finetuning, admm_it, model, model_class, method, all_images, checkpoint_simple_path_exp, training):
        if (finetuning == 'last'): # last model saved in checkpoint
            if (admm_it > 0): # if model has already been trained
                model = model_class.load_from_checkpoint(os.path.join(checkpoint_simple_path_exp,'last.ckpt'), hyperparameters_config=hyperparameters_config, method=method, all_images = all_images) # Load previous model in checkpoint        
        # if (admm_it == 0):
            # DD finetuning, k=32, d=6
            #model = model_class.load_from_checkpoint(os.path.join(subroot,'high_statistics.ckpt'), hyperparameters_config=hyperparameters_config) # Load model coming from high statistics computation (normally coming from finetuning with supervised learning)
            #from torch.utils.tensorboard import SummaryWriter
            #writer = SummaryWriter()
            #out = model(image_net_input_torch)
            #write_image_tensorboard(writer,out.detach().numpy(),"high statistics output)",suffix,image_gt) # Showing all corrupted images with same contrast to compare them together
            #write_image_tensorboard(writer,out.detach().numpy(),"high statistics (" + "output, suffix,image_gt,FULL CONTRAST)",0,full_contrast=True) # Showing each corrupted image with contrast = 1
        
            # Set first network iterations to have convergence, as if we do post processing
            # model = model_class.load_from_checkpoint(os.path.join(subroot,'post_reco'+net+'.ckpt'), hyperparameters_config=hyperparameters_config) # Load model coming from high statistics computation (normally coming from finetuning with supervised learning)

        if (finetuning == 'best'): # best model saved in checkpoint
            if (admm_it > 0): # if model has already been trained
                model = model_class.load_from_checkpoint(os.path.join(checkpoint_simple_path_exp,'best_loss.ckpt'), hyperparameters_config=hyperparameters_config,method=method, all_images = all_images) # Load best model in checkpoint
            #if (admm_it == 0):
            # DD finetuning, k=32, d=6
                #model = model_class.load_from_checkpoint(os.path.join(subroot,'high_statistics.ckpt'), hyperparameters_config=hyperparameters_config) # Load model coming from high statistics computation (normally coming from finetuning with supervised learning)
            if (training):
                os.system('rm -rf ' + checkpoint_simple_path_exp + '/best_loss.ckpt') # Otherwise, pl will store checkpoint with version in filename
        
        return model

    def runComputation(self,config,fixed_config,hyperparameters_config,root):
        # Scaling of x_label image
        image_corrupt_input_scale,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt = self.rescale_imag(self.image_corrupt,self.scaling_input) # Scaling of x_label image

        # Corrupted image x_label, numpy --> torch
        self.image_corrupt_torch = torch.Tensor(image_corrupt_input_scale)
        # Adding dimensions to fit network architecture
        self.image_corrupt_torch = self.image_corrupt_torch.view(1,1,self.PETImage_shape[0],self.PETImage_shape[1],self.PETImage_shape[2])
        if (len(self.image_corrupt_torch.shape) == 5): # if 3D but with dim3 = 1 -> 2D
            self.image_corrupt_torch = self.image_corrupt_torch[:,:,:,:,0]

        # Training model with sub_iter_DIP iterations
        model = self.train_process(self.suffix, hyperparameters_config, self.finetuning, self.processing_unit, self.sub_iter_DIP, self.method, self.admm_it, self.image_net_input_torch, self.image_corrupt_torch, self.net, self.PETImage_shape, self.experiment, self.checkpoint_simple_path, self.name_run, self.subroot) # Not useful to make iterations, we just want to initialize writer. admm_it must be set to -1, otherwise seeking for a checkpoint file...
        if (self.net == 'DIP_VAE'):
            out, mu, logvar, z = model(self.image_net_input_torch)
        else:
            out = model(self.image_net_input_torch)

        # Descaling like at the beginning
        out_descale = self.descale_imag(out,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)
        # Saving image output
        self.save_img(out_descale, self.net_outputs_path)

    def choose_net(self,net, hyperparameters_config, method, all_images, admm_it, PETImage_shape):
        if (net == 'DIP'): # Loading DIP architecture
            if(PETImage_shape[2] == 1): # 2D
                model = DIP_2D(hyperparameters_config,method,all_images,admm_it) 
                model_class = DIP_2D
            else: # 3D
                model = DIP_3D(hyperparameters_config,method)
                model_class = DIP_3D
        elif (net == 'DIP_VAE'): # Loading DIP VAE architecture
            model = VAE_DIP_2D(hyperparameters_config)
            model_class = VAE_DIP_2D
        elif (net == 'DD'): # Loading Deep Decoder architecture
                model = DD_2D(hyperparameters_config,method)
                model_class = DD_2D
        elif (net == 'DD_AE'): # Loading Deep Decoder based autoencoder architecture
            model = DD_AE_2D(hyperparameters_config) 
            model_class = DD_AE_2D
        return model, model_class
    
    def generate_nn_output(self,net, hyperparameters_config, method, image_net_input_torch, PETImage_shape, finetuning, admm_it, experiment, suffix, subroot, all_images = True):
        # Loading using previous model
        model, model_class = self.choose_net(net, hyperparameters_config, method, all_images, admm_it, PETImage_shape)
        checkpoint_simple_path_exp = subroot+'Block2/checkpoint/'+format(experiment) + '/' + suffix + '/'
        model = self.load_model(image_net_input_torch, hyperparameters_config, finetuning, admm_it, model, model_class, method, all_images, checkpoint_simple_path_exp, training=False)

        # Compute output image
        out, mu, logvar, z = model(image_net_input_torch)

        # Loading X_label from block1 to destandardize NN output
        image_corrupt = self.fijii_np(subroot+'Block2/x_label/' + format(experiment)+'/'+ format(admm_it - 1) +'_x_label' + suffix + '.img',shape=(PETImage_shape))
        image_corrupt_scale,param1_scale_im_corrupt,param2_scale_im_corrupt = self.rescale_imag(image_corrupt)

        # Reverse scaling like at the beginning and add it to list of samples
        out_descale = self.descale_imag(out,param1_scale_im_corrupt,param2_scale_im_corrupt,hyperparameters_config["scaling"])
        return out_descale
