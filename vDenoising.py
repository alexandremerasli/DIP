## Python libraries

# Pytorch
from torch import Tensor, save
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Useful
from numpy import inf, array, arange, ones, copy, zeros, linspace, float32
from numpy.random import seed, uniform, normal
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# Local files to import
from vGeneral import vGeneral

from models.DIP_2D import DIP_2D # DIP
from models.DIP_3D import DIP_3D # DIP
from models.VAE_DIP_2D import VAE_DIP_2D # DIP vae
from models.DD_2D import DD_2D # DD
from models.DD_AE_2D import DD_AE_2D # DD adding encoder part

import abc
class vDenoising(vGeneral):
    #@abc.abstractmethod
    def __init__(self,config, *args, **kwargs):
        print('__init__')

    def points_in_circle_edge(self,center_y,center_x,radius,PETImage_shape,inner_circle=True): # x and y are inverted in an array compared to coordinates
        liste = [] 

        center_x += int(PETImage_shape[0]/2)
        center_y += int(PETImage_shape[1]/2)
        for x in range(0,PETImage_shape[0]):
            for y in range(0,PETImage_shape[1]):
                if (x+0.5-center_x)**2 + (y+0.5-center_y)**2 <= radius**2 and (x+0.5-center_x)**2 + (y+0.5-center_y)**2 > (radius - 2)**2:
                    liste.append((x,y))

        return liste

    def initializeSpecific(self,config,root, *args, **kwargs):
        # Set random seed if asked (for random input here)
        if (os.path.isfile(os.getcwd() + "/seed.txt")):
            with open(os.getcwd() + "/seed.txt", 'r') as file:
                random_seed = file.read().rstrip()
            if (eval(random_seed)):
                seed(1)

        self.all_images_DIP = config["all_images_DIP"]


        self.createDirectoryAndConfigFile(config)
        # Specific hyperparameters for reconstruction module (Do it here to have raytune config hyperparameters selection)
        if (config["net"] == "DD" or config["net"] == "DD_AE"):
            self.d_DD = config["d_DD"]
            self.k_DD = config["k_DD"]
        if ( 'nested' in config["method"] or  'Gong' in config["method"]):
            self.input = config["input"]
            self.scaling_input = config["scaling"]
            # Loading DIP input
            # Creating random image input for DIP while we do not have CT, but need to be removed after
            self.create_input(self.net,self.PETImage_shape,config,self.subroot_data) # to be removed when CT will be used instead of random input. DO NOT PUT IT IN BLOCK 2 !!!

            # Create random image to fit by DIP (test for ML reading group)
            '''
            if (self.FLTNB == 'float'):
                type_im = 'float32'
            elif (self.FLTNB == 'double'):
                type_im = 'float64'
            file_path = (self.subroot_data+'Data/initialization/random_1.img')
            im_input = np.random.uniform(0,1,self.PETImage_shape[0]*self.PETImage_shape[1]*self.PETImage_shape[2]).astype(type_im) # initializing input image with random image (for DIP)
            im_input = im_input.reshape(self.PETImage_shape) # reshaping (for DIP)
            self.save_img(im_input,file_path)
            '''


            # Loading DIP input (we do not have CT-map, so random image created in block 1)
            self.image_net_input = self.load_input(self.net,self.PETImage_shape,self.subroot_data) # Scaling of network input. DO NOT CREATE RANDOM INPUT IN BLOCK 2 !!! ONLY AT THE BEGINNING, IN BLOCK 1    
            # modify input with line on the edge of the phantom, or to remove a region (DIP input tests)
            # self.modify_input_line_edge(config)     
            # Rescale network input
            self.image_net_input_scale = self.rescale_imag(self.image_net_input,self.scaling_input)[0]
            # Diffusion model like : add random noise to anatomical input or use several inputs for the same training
            # if (self.input == "CT"):
            # Generate random input
            # gaussian_distribution = normal(0, (self.global_it+1) * self.diffusion_model_like,self.PETImage_shape[0]*self.PETImage_shape[1]*self.PETImage_shape[2]).reshape(self.PETImage_shape) # reshaping (for DIP)
            # self.image_net_input_scale += gaussian_distribution
            if (self.diffusion_model_like != 0):
                if (self.several_DIP_inputs == 1):
                    self.image_net_input_scale = self.add_gaussian_noise(copy(self.image_net_input_scale), self.global_it + 1,self.diffusion_model_like)
                else:
                    raise ValueError("not implemented")
            else:
                if (self.diffusion_model_like_each_DIP != 0 and self.several_DIP_inputs != 1):
                    image_net_initial = copy(self.image_net_input_scale)
                    it_list = linspace(0,int(1/self.diffusion_model_like_each_DIP),self.several_DIP_inputs)
                    dim_image_net = list(self.PETImage_shape)
                    dim_image_net.insert(0,self.several_DIP_inputs)
                    self.image_net_input_scale = zeros(dim_image_net)
                    for i in range(len(it_list)):
                        self.image_net_input_scale[i,:,:,:] = self.add_gaussian_noise(copy(image_net_initial), it_list[i],self.diffusion_model_like_each_DIP)
                elif (self.diffusion_model_like_each_DIP == 0 and self.several_DIP_inputs != 1):
                    image_net_initial = copy(self.image_net_input_scale)
                    dim_image_net = list(self.PETImage_shape)
                    dim_image_net.insert(0,self.several_DIP_inputs)
                    self.image_net_input_scale = zeros(dim_image_net,dtype=float32) # Like if one DIP input
                    for i in range(self.several_DIP_inputs):
                        self.image_net_input_scale[i,:,:,:] = copy(image_net_initial)
                elif (self.diffusion_model_like_each_DIP != 0 and self.several_DIP_inputs == 1):
                    self.image_net_input_scale = self.add_gaussian_noise(copy(self.image_net_input_scale), 1,self.diffusion_model_like_each_DIP)


            # DIP input image, numpy --> torch
            self.image_net_input_torch = Tensor(self.image_net_input_scale)
            # Adding dimensions to fit network architecture
            if (self.net == 'DIP' or self.net == 'DIP_VAE' or self.net == 'DD_AE' or self.net == "DIP_Xin" or self.net == "Swin_Unetr"): # For autoencoders structure
                if (self.PETImage_shape[2] == 1): # if 3D but with dim3 = 1 -> 2D
                    self.image_net_input_torch = self.image_net_input_torch.view(self.several_DIP_inputs,1,self.PETImage_shape[0],self.PETImage_shape[1],self.PETImage_shape[2])
                    self.image_net_input_torch = self.image_net_input_torch[:,:,:,:,0]
                else: #3D
                    self.image_net_input_torch = self.image_net_input_torch.view(1,1,self.PETImage_shape[2],self.PETImage_shape[1],self.PETImage_shape[0])
            elif (self.net == 'DD'):
                    input_size_DD = int(self.PETImage_shape[0] / (2**config["d_DD"])) # if original Deep Decoder (i.e. only with decoder part)
                    self.image_net_input_torch = self.image_net_input_torch.view(1,config["k_DD"],input_size_DD,input_size_DD) # For Deep Decoder, if original Deep Decoder (i.e. only with decoder part)
            save(self.image_net_input_torch,self.subroot_data + 'Data/initialization/pytorch/replicate_' + str(self.replicate) + '/image_' + self.net + '_input_torch.pt')

    def add_gaussian_noise(self,img,it,diffusion_model_like_each_DIP):
        gaussian_distribution = normal(0, it * diffusion_model_like_each_DIP,self.PETImage_shape[0]*self.PETImage_shape[1]*self.PETImage_shape[2]).reshape(self.PETImage_shape) # reshaping (for DIP)
        return img + gaussian_distribution

    def train_process(self, param1_scale_im_corrupt, param2_scale_im_corrupt, scaling_input, suffix, config, finetuning, processing_unit, sub_iter_DIP, method, global_it, image_net_input_torch, image_corrupt_torch, net, PETImage_shape, experiment, checkpoint_simple_path, name_run, subroot, all_images_DIP):
        # Implements Dataset
        train_dataset = TensorDataset(image_net_input_torch, image_corrupt_torch) # Put several times the input
        # train_dataset = TensorDataset(*self.several_DIP_inputs*[image_net_input_torch], *self.several_DIP_inputs*[image_corrupt_torch])
        
        # Add different level of gaussian noise to input
        if (self.diffusion_model_like_each_DIP != 0):
            it_list = arange(0,self.several_DIP_inputs)
            # train_dataset = ImagePairDataset([(image_net_input_torch[i], image_corrupt_torch) for i in range(len(it_list))])
            image_net_input_torch = image_net_input_torch[:, :, None, :]
            image_corrupt_torch = image_corrupt_torch[:, :, None, :]
            train_dataset = ImagePairDataset([(image_net_input_torch[i], image_corrupt_torch) for i in range(len(it_list))])
            # train_dataset = ImagePairDataset([(image_net_input_torch[len(it_list)-i-1], image_corrupt_torch) for i in range(len(it_list))])

        else:
            train_dataset = ImagePairDataset([(image_net_input_torch,image_corrupt_torch) for i in range(self.several_DIP_inputs)])

        if (config["tau_DIP"] == 200):        
            train_dataloader = DataLoader(train_dataset, batch_size=1,num_workers=0,shuffle=False) # Mini batch training without shuffle
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=1,num_workers=0,shuffle=True) # Mini batch training
        # train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=1, persistent_workers=True) # num_workers is 0 by default, which means the training process will work sequentially inside the main process
        # Choose network architecture as model
        model, model_class = self.choose_net(net, param1_scale_im_corrupt, param2_scale_im_corrupt, scaling_input, config, method, all_images_DIP, global_it, PETImage_shape, suffix, self.override_input)
        # Define path for this global iteration
        self.checkpoint_simple_path_exp = subroot+'Block2/' + self.suffix + '/checkpoint/'+format(experiment) + '/' + str(self.global_it)
        Path(self.checkpoint_simple_path_exp+'/').mkdir(parents=True, exist_ok=True)
        # Define path for previous global iteration
        checkpoint_simple_path_previous_exp = subroot+'Block2/' + self.suffix + '/checkpoint/'+format(experiment)
        if (self.global_it != -100): # if not in post reco mode
            checkpoint_simple_path_previous_exp += '/' + str(self.global_it - 1)
        else:
            checkpoint_simple_path_previous_exp += '/' + str(self.global_it)

        # model = self.load_model(param1_scale_im_corrupt, param2_scale_im_corrupt, scaling_input, image_net_input_torch, config, finetuning, global_it, model, model_class, method, all_images_DIP, checkpoint_simple_path_previous_exp, training=True)
    
        # if (self.processing_unit == 'CPU'):
        #     summary_model = model
        # else:
        #     summary_model = model.cuda()
        # from torchsummary import summary
        # if (PETImage_shape[2] == 1): # 2D
        #     summary(model, input_size=(1,PETImage_shape[0],PETImage_shape[1])) # for DIP
        # else: # 3D
        #     summary(model, input_size=(1,PETImage_shape[0],PETImage_shape[1],PETImage_shape[2])) # for DIP

        # # Save the original standard output
        # import sys
        # original_stdout = sys.stdout 

        # with open('model_summary.txt', 'w') as f:
        #     sys.stdout = f # Change the standard output to the file we created.
        #     summary(model, input_size=(1,PETImage_shape[0],PETImage_shape[1])) # for DIP
        #     sys.stdout = original_stdout # Reset the standard output to its original value
        
        # Start training
        print('Starting optimization, iteration',global_it)
        trainer = self.create_pl_trainer(finetuning, processing_unit, sub_iter_DIP, global_it, net, checkpoint_simple_path, experiment, self.checkpoint_simple_path_exp, checkpoint_simple_path_previous_exp, config,name=name_run)

        trainer.fit(model, train_dataloader)
        
        if (finetuning == "last"):
            trainer.save_checkpoint(self.checkpoint_simple_path_exp + "/last.ckpt")

        # Copy last checkpoint to file "last.ckpt" or to ES checkpoint 
        import shutil
        for file in os.listdir(self.checkpoint_simple_path_exp):
            if (config["finetuning"] != "ES"):# or self.global_it >= 0):
                if ("epoch" in file):
                    shutil.copy(os.path.join(self.checkpoint_simple_path_exp,file),os.path.join(self.checkpoint_simple_path_exp,"last.ckpt"))
                    os.remove(os.path.join(self.checkpoint_simple_path_exp,file))
            if (config["finetuning"] == "ES"):
                if (config["DIP_early_stopping"]):
                    if (model.epochStar != -1): # if ES point found, save ES ckpt
                        if (file == "epoch=" + str(model.epochStar) + "-step=" + str((model.epochStar+1)*self.several_DIP_inputs-1) + ".ckpt"):
                            shutil.copy(os.path.join(self.checkpoint_simple_path_exp,"epoch=" + str(model.epochStar) + "-step=" + str((model.epochStar+1)*self.several_DIP_inputs-1) + ".ckpt"),os.path.join(self.checkpoint_simple_path_exp,"last.ckpt"))
                        # os.remove(os.path.join(self.checkpoint_simple_path_exp,"epoch=" + str(model.epochStar) + "-step=" + str(model.epochStar) + ".ckpt"))
                        else:
                            print(os.path.join(self.checkpoint_simple_path_exp,file))
                            # os.remove(os.path.join(self.checkpoint_simple_path_exp,file))
                    else: # if ES point not found, save last ckpt
                        if (file == "epoch=" + str(model.sub_iter_DIP_already_done-1) + "-step=" + str(model.sub_iter_DIP_already_done*self.several_DIP_inputs-1) + ".ckpt"):
                            shutil.copy(os.path.join(self.checkpoint_simple_path_exp,"epoch=" + str(model.sub_iter_DIP_already_done-1) + "-step=" + str(model.sub_iter_DIP_already_done*self.several_DIP_inputs-1) + ".ckpt"),os.path.join(self.checkpoint_simple_path_exp,"last.ckpt"))
                            # os.remove(os.path.join(self.checkpoint_simple_path_exp,"epoch=" + str(model.epochStar) + "-step=" + str(model.epochStar) + ".ckpt"))
                        else:
                            # os.remove(os.path.join(self.checkpoint_simple_path_exp,file))
                            print(os.path.join(self.checkpoint_simple_path_exp,file))
        if(self.global_it >= 0):
            if (os.path.isdir(os.path.join(checkpoint_simple_path_previous_exp))):
                shutil.rmtree(os.path.join(checkpoint_simple_path_previous_exp))

        return model

    def create_pl_trainer(self,finetuning, processing_unit, sub_iter_DIP, global_it, net, checkpoint_simple_path, experiment, checkpoint_simple_path_exp, checkpoint_simple_path_previous_exp, config, name=''):
        
        from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
        TuneReportCheckpointCallback

        tuning_callback = TuneReportCallback({"loss": "val_loss"}, on="validation_end")
        if (processing_unit == 'CPU'): # use cpus and no gpu
            gpus = 0
            accelerator = "cpu"
        elif (processing_unit == 'GPU' or processing_unit == 'both'): # use all available gpus, no cpu (pytorch lightning does not handle cpus and gpus at the same time)
            gpus = 1
            accelerator = "gpu"
            #if (torch.cuda.device_count() > 1):
            #    accelerator = 'dp'


        # Early stopping callback
        from pytorch_lightning.callbacks.early_stopping import EarlyStopping
        early_stopping_callback = EarlyStopping(monitor="SUCCESS", mode="max",stopping_threshold=0.9,patience=inf) # SUCCESS will be 1 when ES if found, which is greater than stopping_threshold = 0.9

        if (hasattr(config,"override_it_DIP_with_EMV_it")):
            if (config["override_it_DIP_with_EMV_it"]):
                self.sub_iter_DIP = self.sub_iter_DIP_initial_and_final
                sub_iter_DIP = self.sub_iter_DIP

        print("global_it",global_it)
        if (global_it == -1): # or global_it == self.max_iter - 1): # Number of initial and final iterations are overrided here
            print(str(self.sub_iter_DIP_initial_and_final) + " initial iterations for Gong")
            self.sub_iter_DIP = self.sub_iter_DIP_initial_and_final
            sub_iter_DIP = self.sub_iter_DIP
        if (finetuning == 'False'): # Do not save and use checkpoints (still save hparams and event files for now ...)
            logger = TensorBoardLogger(save_dir=checkpoint_simple_path, version=format(experiment), name=name) # Store checkpoints in checkpoint_simple_path path
            #checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_simple_path_exp, save_top_k=0, save_weights_only=True) # Do not save any checkpoint (save_top_k = 0)
            # checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_simple_path_exp, save_last=True, save_top_k=0) # Only save last checkpoint as last.ckpt (save_last = True), do not save checkpoint at each epoch (save_top_k = 0). We do not use it a priori, except in post reconstruction to initialize
            trainer = Trainer(max_epochs=sub_iter_DIP,log_every_n_steps=1, callbacks=[tuning_callback, early_stopping_callback], logger=logger,gpus=gpus, accelerator=accelerator, profiler="simple", progress_bar_refresh_rate=0, weights_summary=None)
        else:
            if (finetuning == 'last'): # last model saved in checkpoint
                # Checkpoints pl variables
                logger = TensorBoardLogger(save_dir=checkpoint_simple_path, version=format(experiment), name=name) # Store checkpoints in checkpoint_simple_path path
                checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_simple_path_exp, save_top_k=0) # Only save last checkpoint as last.ckpt (save_last = True), do not save checkpoint at each epoch (save_top_k = 0)
                # checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_simple_path_exp) # Only save last checkpoint as last.ckpt (save_last = True), do not save checkpoint at each epoch (save_top_k = 0)
                # trainer = Trainer(max_epochs=sub_iter_DIP,log_every_n_steps=1, logger=logger, callbacks=[checkpoint_callback, tuning_callback, early_stopping_callback],gpus=gpus, accelerator=accelerator,log_gpu_memory="all", progress_bar_refresh_rate=0, weights_summary=None, profiler="simple") # Prepare trainer model with callback to save checkpoint        
                

                
                from os.path import isfile
                if (isfile(checkpoint_simple_path_previous_exp + '/last.ckpt')):
                    trainer = Trainer(resume_from_checkpoint=checkpoint_simple_path_previous_exp + "/last.ckpt", max_epochs=sub_iter_DIP,log_every_n_steps=1,logger=logger, callbacks=[checkpoint_callback,early_stopping_callback],gpus=gpus)#, , early_stopping_callback])#, callbacks=[checkpoint_callback, tuning_callback, early_stopping_callback], logger=logger,gpus=gpus, accelerator=accelerator, profiler="simple")
                    # trainer = Trainer(max_epochs=sub_iter_DIP,log_every_n_steps=1,logger=logger, callbacks=[checkpoint_callback,early_stopping_callback],gpus=gpus)#, , early_stopping_callback])#, callbacks=[checkpoint_callback, tuning_callback, early_stopping_callback], logger=logger,gpus=gpus, accelerator=accelerator, profiler="simple")
                else:
                    trainer = Trainer(max_epochs=sub_iter_DIP,log_every_n_steps=1,logger=logger, callbacks=[checkpoint_callback,early_stopping_callback],gpus=gpus)#, , early_stopping_callback])#, callbacks=[checkpoint_callback, tuning_callback, early_stopping_callback], logger=logger,gpus=gpus, accelerator=accelerator, profiler="simple")
            if (finetuning == 'best'): # best model saved in checkpoint
                # Checkpoints pl variables
                logger = TensorBoardLogger(save_dir=checkpoint_simple_path, version=format(experiment), name=name) # Store checkpoints in checkpoint_simple_path path
                checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_simple_path_exp, filename = 'best_loss', monitor='loss_monitor', save_top_k=1) # Save best checkpoint (save_top_k = 1) (according to minimum loss (monitor)) as best_loss.ckpt
                trainer = Trainer(max_epochs=sub_iter_DIP,log_every_n_steps=1, logger=logger, callbacks=[checkpoint_callback, tuning_callback, early_stopping_callback],gpus=gpus, accelerator=accelerator, profiler="simple", progress_bar_refresh_rate=0, weights_summary=None) # Prepare trainer model with callback to save checkpoint
            if (finetuning == 'ES'): # best model saved in checkpoint
                # Delete previous checkpoints from previous runs
                # if (global_it > -1): # Beginning nested or Gong in block2. For first epoch, change number of epochs to sub_iter_DIP_initial_and_final for Gong
                #     os.system("rm -rf " + os.path.join(checkpoint_simple_path_previous_exp))
                    #for f in os.listdir(checkpoint_simple_path_previous_exp):
                        #if (int(re.search(r'\d+', f).group()) != self.epochStar):
                        
                # Checkpoints pl variables
                logger = TensorBoardLogger(save_dir=checkpoint_simple_path, version=format(experiment), name=name) # Store checkpoints in checkpoint_simple_path path
                checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_simple_path_exp, save_top_k=-1) # Save checkpoint at each epoch (save_top_k = -1) to use the one corresponding to ES point
                from os.path import isfile
                if (isfile(checkpoint_simple_path_previous_exp + '/last.ckpt')):
                    trainer = Trainer(resume_from_checkpoint=checkpoint_simple_path_previous_exp + "/last.ckpt", max_epochs=sub_iter_DIP,log_every_n_steps=1,logger=logger, callbacks=[checkpoint_callback,early_stopping_callback],gpus=gpus)#, , early_stopping_callback])#, callbacks=[checkpoint_callback, tuning_callback, early_stopping_callback], logger=logger,gpus=gpus, accelerator=accelerator, profiler="simple")
                    # trainer = Trainer(max_epochs=sub_iter_DIP,log_every_n_steps=1,logger=logger, callbacks=[checkpoint_callback,early_stopping_callback],gpus=gpus)#, , early_stopping_callback])#, callbacks=[checkpoint_callback, tuning_callback, early_stopping_callback], logger=logger,gpus=gpus, accelerator=accelerator, profiler="simple")
                else:
                    trainer = Trainer(max_epochs=sub_iter_DIP,log_every_n_steps=1,logger=logger, callbacks=[checkpoint_callback,early_stopping_callback],gpus=gpus)#, , early_stopping_callback])#, callbacks=[checkpoint_callback, tuning_callback, early_stopping_callback], logger=logger,gpus=gpus, accelerator=accelerator, profiler="simple")
        return trainer

    def create_input(self,net,PETImage_shape,config,subroot): #CT map for high-count data, but not CT yet...
        
        # Write image if it does not exist
        if config["input"] == "random":
            name = 'random/replicate_' + str(self.replicate) + '/random_input'
            if (not os.path.isdir(subroot+'Data/initialization/' + 'random/replicate_' + str(self.replicate) + '/')):
                Path(subroot+'Data/initialization/' + 'random/replicate_' + str(self.replicate) + '/').mkdir(parents=True, exist_ok=True)
        elif config["input"] == "uniform":
            name = 'uniform_input'
        else: # CT input, do not need to create one
            return 1
        
        if (PETImage_shape[2] == 1):
            name += '_2D_'
        else:
            name += '_3D_'

        file_path = (subroot+'Data/initialization/' + name + net + '_' + str(self.PETImage_shape[0]) + '.img')
        if ((os.path.isfile(file_path) and not config["random_seed"]) or (not os.path.isfile(file_path))):
            constant_uniform = 1
            if (self.FLTNB == 'float'):
                type_im = 'float32'
            elif (self.FLTNB == 'double'):
                type_im = 'float64'
            if (net == 'DIP' or net == 'DIP_VAE' or self.net == "DIP_Xin" or self.net == "Swin_Unetr"):
                if config["input"] == "random":
                    im_input = uniform(0,1,PETImage_shape[0]*PETImage_shape[1]*PETImage_shape[2]).astype(type_im) # initializing input image with random image (for DIP)
                elif config["input"] == "uniform":
                    im_input = constant_uniform*ones((PETImage_shape[0]*PETImage_shape[1]*PETImage_shape[2])).astype(type_im) # initializing input image with random image (for DIP)
                else:
                    return "CT input, do not need to create input"
                im_input = im_input.reshape(PETImage_shape) # reshaping (for DIP)
            else:
                if (net == 'DD'):
                    input_size_DD = int(PETImage_shape[0] / (2**config["d_DD"])) # if original Deep Decoder (i.e. only with decoder part)
                    if config["input"] == "random":
                        im_input = uniform(0,1,config["k_DD"]*input_size_DD*input_size_DD).astype(type_im) # initializing input image with random image (for Deep Decoder) # if original Deep Decoder (i.e. only with decoder part)
                    elif config["input"] == "uniform":
                        im_input = constant_uniform*ones((config["k_DD"],input_size_DD,input_size_DD)).astype(type_im) # initializing input image with random image (for Deep Decoder) # if original Deep Decoder (i.e. only with decoder part)
                    else:
                        return "CT input, do not need to create input"
                    im_input = im_input.reshape(config["k_DD"],input_size_DD,input_size_DD) # reshaping (for Deep Decoder) # if original Deep Decoder (i.e. only with decoder part)
                    
                elif (net == 'DD_AE'):
                    if config["input"] == "random":
                        im_input = uniform(0,1,PETImage_shape[0]*PETImage_shape[1]*PETImage_shape[2]).astype(type_im) # initializing input image with random image (for Deep Decoder) # if auto encoder based on Deep Decoder
                    elif config["input"] == "uniform":
                        im_input = constant_uniform*ones((PETImage_shape[0]*PETImage_shape[1]*PETImage_shape[2])).astype(type_im) # initializing input image with random image (for Deep Decoder) # if auto encoder based on Deep Decoder
                    else:
                        return "CT input, do not need to create input"
                    im_input = im_input.reshape(PETImage_shape[0],PETImage_shape[1],PETImage_shape[2]) # reshaping (for Deep Decoder) # if auto encoder based on Deep Decoder

            self.save_img(im_input,file_path)

    def load_input(self,net,PETImage_shape,subroot):
        if (self.override_input): # TESTCT_random
            self.input = "CT"
            #self.input = "random"

        if self.input == "random":        
            if (PETImage_shape[2] == 1):
                file_path = (subroot+'Data/initialization/random/replicate_' + str(self.replicate) + '/random_input_2D_' + net + '_' + str(self.PETImage_shape[0]) + '.img')
            else:
                file_path = (subroot+'Data/initialization/random_input_3D_' + net + '_' + str(self.PETImage_shape[0]) + '.img')
        elif self.input == "CT":
            if ("4_" in self.phantom or self.phantom == "image400_0" or self.phantom == "image40_0" or self.phantom == "image40_1" or self.phantom == "image50_0" or self.phantom == "image50_1" or self.phantom == "image50_2"):
                if (os.path.isfile(subroot+'Data/database_v2/' + self.phantom + '/' + self.phantom + '_mr.raw')): # If MR exists
                    file_path = subroot+'Data/database_v2/' + self.phantom + '/' + self.phantom + '_mr.raw'
            else:
                file_path = (subroot+'Data/database_v2/' + self.phantom + '/' + self.phantom + '_atn.raw') #CT map, but not CT yet, attenuation for now...
        elif self.input == "BSREM":
            file_path = (subroot+'Data/initialization/BSREM_it30_REF_cropped.img') #
        elif self.input == "uniform":
            file_path = (subroot+'Data/initialization/uniform_input_' + net + '.img')
        if (net == 'DD'):
            if (self.input != "random"):
                raise ValueError("input must be random with Deep Decoder")
            input_size_DD = int(PETImage_shape[0] / (2**self.d_DD)) # if original Deep Decoder (i.e. only with decoder part)
            PETImage_shape = (self.k_DD,input_size_DD,input_size_DD) # if original Deep Decoder (i.e. only with decoder part)
        #elif (net == 'DD_AE'):   
        #    PETImage_shape = (PETImage_shape[0],PETImage_shape[1],PETImage_shape[2]) # if auto encoder based on Deep Decoder

        if (self.input == 'CT' and self.net != 'DD'):
            type_im = '<f'
        else:
            type_im = '<d' # random images were generated in double
            #type_im = None

        im_input = self.fijii_np(file_path, shape=(PETImage_shape),type_im=type_im) # Load input of the DNN (CT image)
        return im_input


    def load_model(self,param1_scale_im_corrupt, param2_scale_im_corrupt, scaling_input, image_net_input_torch, config, finetuning, global_it, model, model_class, method, all_images_DIP, checkpoint_simple_path_exp, training):
        if (finetuning == 'last'): # last model saved in checkpoint
            if (global_it > 0): # if model has already been trained
                model = model_class.load_from_checkpoint(os.path.join(checkpoint_simple_path_exp,'last.ckpt'), config=config, method=method, all_images_DIP = all_images_DIP, global_it = global_it, param1_scale_im_corrupt=param1_scale_im_corrupt, param2_scale_im_corrupt=param2_scale_im_corrupt, scaling_input=scaling_input,root=self.root,subroot=self.subroot, fixed_hyperparameters_list=self.fixed_hyperparameters_list, hyperparameters_list=self.hyperparameters_list, debug=self.debug, suffix=self.suffix, override_input = self.override_input, scanner = self.scanner) # Load previous model in checkpoint
            elif (global_it == 0 and not config["unnested_1st_global_iter"]):
                model = model_class.load_from_checkpoint(os.path.join(checkpoint_simple_path_exp,'last.ckpt'), config=config, method=method, all_images_DIP = all_images_DIP, global_it = global_it, param1_scale_im_corrupt=param1_scale_im_corrupt, param2_scale_im_corrupt=param2_scale_im_corrupt, scaling_input=scaling_input,root=self.root,subroot=self.subroot, fixed_hyperparameters_list=self.fixed_hyperparameters_list, hyperparameters_list=self.hyperparameters_list, debug=self.debug, suffix=self.suffix, override_input = self.override_input, scanner = self.scanner) # Load previous model in checkpoint
                #model = model_class.load_from_checkpoint(self.subroot_data + 'Data/initialization/' + 'last.ckpt', config=config, method=method, all_images_DIP = all_images_DIP, global_it = global_it, param1_scale_im_corrupt=param1_scale_im_corrupt, param2_scale_im_corrupt=param2_scale_im_corrupt, scaling_input=scaling_input,root=self.root,subroot=self.subroot, fixed_hyperparameters_list=self.fixed_hyperparameters_list, hyperparameters_list=self.hyperparameters_list, debug=self.debug) # enable to avoid pre iteratio
            elif (global_it == 0 and config["unnested_1st_global_iter"]):
                model = model_class.load_from_checkpoint(os.path.join(checkpoint_simple_path_exp,'last.ckpt'), config=config, method=method, all_images_DIP = all_images_DIP, global_it = global_it, param1_scale_im_corrupt=param1_scale_im_corrupt, param2_scale_im_corrupt=param2_scale_im_corrupt, scaling_input=scaling_input,root=self.root,subroot=self.subroot, fixed_hyperparameters_list=self.fixed_hyperparameters_list, hyperparameters_list=self.hyperparameters_list, debug=self.debug, suffix=self.suffix, override_input = self.override_input, scanner = self.scanner) # Load previous model in checkpoint
                #model = model_class.load_from_checkpoint(self.subroot_data + 'Data/initialization/' + 'last.ckpt', config=config, method=method, all_images_DIP = all_images_DIP, global_it = global_it, param1_scale_im_corrupt=param1_scale_im_corrupt, param2_scale_im_corrupt=param2_scale_im_corrupt, scaling_input=scaling_input,root=self.root,subroot=self.subroot, fixed_hyperparameters_list=self.fixed_hyperparameters_list, hyperparameters_list=self.hyperparameters_list, debug=self.debug) # enable to avoid pre iteratio
            elif (global_it == -100): # post reco was not already launched once
                if (os.path.isfile(os.path.join(checkpoint_simple_path_exp,'last.ckpt'))):
                    model = model_class.load_from_checkpoint(os.path.join(checkpoint_simple_path_exp,'last.ckpt'), config=config, method=method, all_images_DIP = all_images_DIP, global_it = global_it, param1_scale_im_corrupt=param1_scale_im_corrupt, param2_scale_im_corrupt=param2_scale_im_corrupt, scaling_input=scaling_input,root=self.root,subroot=self.subroot, fixed_hyperparameters_list=self.fixed_hyperparameters_list, hyperparameters_list=self.hyperparameters_list, debug=self.debug, suffix=self.suffix, override_input = self.override_input, scanner = self.scanner) # Load previous model in checkpoint

        elif (finetuning == 'ES'): # ES model saved in checkpoint
            if (global_it > 0): # if model has already been trained
                # model = model_class.load_from_checkpoint(os.path.join(checkpoint_simple_path_exp,"epoch=" + str(self.epochStar) + "-step=" + str(self.epochStar)) + ".ckpt", config=config, method=method, all_images_DIP = all_images_DIP, global_it = global_it, param1_scale_im_corrupt=param1_scale_im_corrupt, param2_scale_im_corrupt=param2_scale_im_corrupt, scaling_input=scaling_input,root=self.root,subroot=self.subroot, fixed_hyperparameters_list=self.fixed_hyperparameters_list, hyperparameters_list=self.hyperparameters_list, debug=self.debug, suffix=self.suffix) # Load previous model in checkpoint        
                model = model_class.load_from_checkpoint(os.path.join(checkpoint_simple_path_exp,'last.ckpt'), config=config, method=method, all_images_DIP = all_images_DIP, global_it = global_it, param1_scale_im_corrupt=param1_scale_im_corrupt, param2_scale_im_corrupt=param2_scale_im_corrupt, scaling_input=scaling_input,root=self.root,subroot=self.subroot, fixed_hyperparameters_list=self.fixed_hyperparameters_list, hyperparameters_list=self.hyperparameters_list, debug=self.debug, suffix=self.suffix, override_input = self.override_input, scanner = self.scanner) # Load previous model in checkpoint
            elif (global_it == 0 and not config["unnested_1st_global_iter"]):
                # model = model_class.load_from_checkpoint(os.path.join(checkpoint_simple_path_exp,"epoch=" + str(self.epochStar) + "-step=" + str(self.epochStar)) + ".ckpt", config=config, method=method, all_images_DIP = all_images_DIP, global_it = global_it, param1_scale_im_corrupt=param1_scale_im_corrupt, param2_scale_im_corrupt=param2_scale_im_corrupt, scaling_input=scaling_input,root=self.root,subroot=self.subroot, fixed_hyperparameters_list=self.fixed_hyperparameters_list, hyperparameters_list=self.hyperparameters_list, debug=self.debug, suffix=self.suffix) # Load previous model in checkpoint        
                model = model_class.load_from_checkpoint(os.path.join(checkpoint_simple_path_exp,'last.ckpt'), config=config, method=method, all_images_DIP = all_images_DIP, global_it = global_it, param1_scale_im_corrupt=param1_scale_im_corrupt, param2_scale_im_corrupt=param2_scale_im_corrupt, scaling_input=scaling_input,root=self.root,subroot=self.subroot, fixed_hyperparameters_list=self.fixed_hyperparameters_list, hyperparameters_list=self.hyperparameters_list, debug=self.debug, suffix=self.suffix, override_input = self.override_input, scanner = self.scanner) # Load previous model in checkpoint
                #model = model_class.load_from_checkpoint(self.subroot_data + 'Data/initialization/' + 'last.ckpt', config=config, method=method, all_images_DIP = all_images_DIP, global_it = global_it, param1_scale_im_corrupt=param1_scale_im_corrupt, param2_scale_im_corrupt=param2_scale_im_corrupt, scaling_input=scaling_input,root=self.root,subroot=self.subroot, fixed_hyperparameters_list=self.fixed_hyperparameters_list, hyperparameters_list=self.hyperparameters_list, debug=self.debug) # enable to avoid pre iteratio
            elif (global_it == 0 and config["unnested_1st_global_iter"]):
                model = model_class.load_from_checkpoint(os.path.join(checkpoint_simple_path_exp,'last.ckpt'), config=config, method=method, all_images_DIP = all_images_DIP, global_it = global_it, param1_scale_im_corrupt=param1_scale_im_corrupt, param2_scale_im_corrupt=param2_scale_im_corrupt, scaling_input=scaling_input,root=self.root,subroot=self.subroot, fixed_hyperparameters_list=self.fixed_hyperparameters_list, hyperparameters_list=self.hyperparameters_list, debug=self.debug, suffix=self.suffix, override_input = self.override_input, scanner = self.scanner) # Load previous model in checkpoint
            elif (global_it == -100): # post reco was not already launched once
                if (os.path.isfile(os.path.join(checkpoint_simple_path_exp,'last.ckpt'))):
                    model = model_class.load_from_checkpoint(os.path.join(checkpoint_simple_path_exp,'last.ckpt'), config=config, method=method, all_images_DIP = all_images_DIP, global_it = global_it, param1_scale_im_corrupt=param1_scale_im_corrupt, param2_scale_im_corrupt=param2_scale_im_corrupt, scaling_input=scaling_input,root=self.root,subroot=self.subroot, fixed_hyperparameters_list=self.fixed_hyperparameters_list, hyperparameters_list=self.hyperparameters_list, debug=self.debug, suffix=self.suffix, override_input = self.override_input, scanner = self.scanner) # Load previous model in checkpoint
        elif (finetuning == 'best'): # best model saved in checkpoint
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
        if ("scaling_all_init" in config):
            if (config["scaling_all_init"]):
                image_corrupt_input_scale_not_used,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt = self.rescale_imag(self.image_corrupt_init,self.scaling_input) # Scaling of first x_label image
                image_corrupt_input_scale,param1_scale_im_corrupt_avant,param2_scale_im_corrupt_avant = self.rescale_imag(self.image_corrupt,self.scaling_input + "_init",param1=self.param1_scale_im_corrupt,param2=self.param2_scale_im_corrupt) # Scaling of current x_label image
                # image_corrupt_input_scale_normal,self.param1_scale_im_corrupt_normal,self.param2_scale_im_corrupt_normal = self.rescale_imag(self.image_corrupt,self.scaling_input) # Scaling of current x_label image
            else:
                image_corrupt_input_scale,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt = self.rescale_imag(self.image_corrupt,self.scaling_input) # Scaling of current x_label image
        else:
            image_corrupt_input_scale,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt = self.rescale_imag(self.image_corrupt,self.scaling_input) # Scaling of current x_label image

        # Corrupted image x_label, numpy --> torch float32
        self.image_corrupt_torch = Tensor(self.several_DIP_inputs*[image_corrupt_input_scale])
        # Adding dimensions to fit network architecture
        if (self.PETImage_shape[2] == 1): # if 3D but with dim3 = 1 -> 2D
            self.image_corrupt_torch = self.image_corrupt_torch.view(self.several_DIP_inputs,1,self.PETImage_shape[0],self.PETImage_shape[1],self.PETImage_shape[2])
            self.image_corrupt_torch = self.image_corrupt_torch[:,:,:,:,0]
        else: #3D
            self.image_corrupt_torch = self.image_corrupt_torch.view(1,1,self.PETImage_shape[2],self.PETImage_shape[1],self.PETImage_shape[0])
        # Training model with sub_iter_DIP iterations
        model = self.train_process(self.param1_scale_im_corrupt, self.param2_scale_im_corrupt, self.scaling_input, self.suffix, config, config["finetuning"], self.processing_unit, self.sub_iter_DIP, self.method, self.global_it, self.image_net_input_torch, self.image_corrupt_torch, self.net, self.PETImage_shape, self.experiment, self.checkpoint_simple_path, self.name_run, self.subroot, self.all_images_DIP) # Not useful to make iterations, we just want to initialize writer. global_it must be set to -1, otherwise seeking for a checkpoint file...
        if (self.net == 'DIP_VAE'):
            out, mu, logvar, z = model(self.image_net_input_torch)
        else:
            out = model(self.image_net_input_torch)

        self.sub_iter_DIP_already_done = model.sub_iter_DIP_already_done
        self.sub_iter_DIP_this_global_it = model.sub_iter_DIP_this_global_it
        self.sub_iter_DIP = self.sub_iter_DIP_already_done

        self.DIP_early_stopping = model.DIP_early_stopping
        if self.DIP_early_stopping:
            self.epochStar = model.epochStar
            #self.windowSize = model.windowSize
            self.patienceNumber = model.patienceNumber
            self.VAR_recon = model.VAR_recon
            self.MSE_WMV = model.MSE_WMV
            self.PSNR_WMV = model.PSNR_WMV
            self.SSIM_WMV = model.SSIM_WMV
            self.SUCCESS = model.SUCCESS
            if (self.SUCCESS and self.epochStar!= self.sub_iter_DIP - self.patienceNumber): # ES point is reached
                #if (self.all_images_DIP == "Last"):
                # self.sub_iter_DIP = self.epochStar + 1
                #else:
                self.sub_iter_DIP = self.epochStar + self.patienceNumber + 1

        '''
        # Descaling like at the beginning
        out_descale = self.descale_imag(out,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)
        # Saving image output
        self.save_img(out_descale, self.net_outputs_path)
        '''

        # Write descaled images in files
        if (self.all_images_DIP == "True"):
            if (self.global_it == -100 or self.global_it == -1):
                epoch_values = arange(0,self.sub_iter_DIP)
            else:
                epoch_values = arange(self.sub_iter_DIP - config["sub_iter_DIP"],self.sub_iter_DIP)
                epoch_values = arange(self.sub_iter_DIP - self.sub_iter_DIP_this_global_it,self.sub_iter_DIP)
        elif (self.all_images_DIP == "False"):
            #epoch_values = np.arange(0,self.sub_iter_DIP,max(self.sub_iter_DIP//10,1))
            epoch_values = arange(self.sub_iter_DIP//10,self.sub_iter_DIP+self.sub_iter_DIP//10,max(self.sub_iter_DIP//10,1)) - 1
        elif (self.all_images_DIP == "Last"):
            epoch_values = array([self.sub_iter_DIP-1])

        for epoch in epoch_values:
            # if (config["finetuning"] == "ES"):
            #     net_outputs_path = self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + "/ES_out_" + self.net + format(self.global_it) + '_epoch=' + format(epoch) + '.img'
            # else:
            net_outputs_path = self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + format(self.global_it) + '_epoch=' + format(epoch) + '.img'
            out = self.fijii_np(net_outputs_path,shape=(self.PETImage_shape),type_im='<f')
            #out = torch.from_numpy(out)
            # Descale like at the beginning
            out_descale = self.descale_imag(out,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)
            """
            print("descale")
            print(np.mean(out_descale))
            print(np.min(out_descale))
            print(np.max(out_descale))
            """
            #'''
            # Saving image output
            net_outputs_path = self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + format(self.global_it) + '_epoch=' + format(epoch) + '.img'
            os.system("mv " + net_outputs_path + " " + self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + format(self.global_it) + '_epoch=' + format(epoch)  + 'scaled.img')
            self.save_img(out_descale, net_outputs_path)
            # Squeeze image by loading it
            out_descale = self.fijii_np(net_outputs_path,shape=(self.PETImage_shape),type_im='<f') # loading DIP output
            # Saving (now DESCALED) image output
            self.save_img(out_descale, net_outputs_path)

            '''
            # Compute IR metric (different from others with several replicates)
            classResults.compute_IR_bkg(self.PETImage_shape,out_descale,epoch,classResults.IR_bkg_recon,self.phantom)
            classResults.writer.add_scalar('Image roughness in the background (best : 0)', classResults.IR_bkg_recon[epoch], epoch+1)
            # Write images over epochs
            classResults.writeEndImagesAndMetrics(epoch,self.sub_iter_DIP,self.PETImage_shape,out_descale,self.suffix,self.phantom,self.net,pet_algo="to fit",iteration_name="(post reconstruction)",all_images_DIP=all_images_DIP)
            '''

        batch_idx = "MR_forward"
        net_forward_MR = self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + 'DIP' + format(self.global_it) + '_epoch=' + format(self.sub_iter_DIP_already_done-1) + ('_batchidx=' + format(batch_idx))*(batch_idx!=-1) + '.img'
        if (self.several_DIP_inputs > 1):
        # if (os.path.isfile(net_forward_MR)):
            out = self.fijii_np(net_forward_MR,shape=(self.PETImage_shape),type_im='<f')
            # Descale like at the beginning
            out_descale = self.descale_imag(out,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)
            # Saving image output
            os.system("mv " + net_forward_MR + " " + self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + 'DIP' + format(self.global_it) + '_epoch=' + format(self.sub_iter_DIP_already_done - 1) + ('_batchidx=' + format(batch_idx))*(batch_idx!=-1) + 'scaled.img')
            self.save_img(out_descale, net_forward_MR)
            # Squeeze image by loading it
            out_descale = self.fijii_np(net_forward_MR,shape=(self.PETImage_shape),type_im='<f') # loading DIP output
            # Saving (now DESCALED) image output
            self.save_img(out_descale, net_forward_MR)


    def choose_net(self, net, param1_scale_im_corrupt, param2_scale_im_corrupt, scaling_input, config, method, all_images_DIP, global_it, PETImage_shape, suffix, override_input):
        if (net == 'DIP'): # Loading DIP architecture
            if(PETImage_shape[2] == 1): # 2D
                model = DIP_2D(param1_scale_im_corrupt, param2_scale_im_corrupt, scaling_input, self.config,self.root,self.subroot,method,all_images_DIP,global_it, self.fixed_hyperparameters_list, self.hyperparameters_list, self.debug, suffix, override_input, self.scanner, self.sub_iter_DIP_already_done, self.override_SC_init)
                model_class = DIP_2D
            else: # 3D
                model = DIP_3D(param1_scale_im_corrupt, param2_scale_im_corrupt, scaling_input, self.config,self.root,self.subroot,method,all_images_DIP,global_it, self.fixed_hyperparameters_list, self.hyperparameters_list, self.debug, suffix, override_input, self.scanner, self.sub_iter_DIP_already_done, self.override_SC_init)
                model_class = DIP_3D
        elif (net == "DIP_Xin"):
            self.embed_dim = 16
            self.kernel_size = 3
            self.skip = 3
            self.num_layers = 3
            self.depths = 2
            self.mode = "bilinear"
            from models.modules_Xin import DIP_skip_add # DIP Xin
            model = DIP_skip_add(1,self.embed_dim,1,self.kernel_size,self.skip,self.num_layers,self.depths,self.mode,config,suffix,param1_scale_im_corrupt, param2_scale_im_corrupt, scaling_input, self.config,self.root,self.subroot,method,all_images_DIP,global_it, self.fixed_hyperparameters_list, self.hyperparameters_list, self.debug, suffix, override_input, self.scanner, self.sub_iter_DIP_already_done, self.override_SC_init)
            model_class = DIP_skip_add
        elif (net == "Swin_Unetr"):
            self.embed_dim = 16
            self.kernel_size = 3
            self.skip = 3
            self.num_layers = 3786541
            self.mode = "bilinear"
            
            self.depths = (2,2,2,2) #tune.grid_search([(2,2,2,2),(4,4,4,4)]),
            self.num_heads = (3,6,12,24) #tune.grid_search([(3,6,12,24),(6,12,24,48)]),
            self.embed_dim = 24 #tune.grid_search([48]),
            self.use_v2 = True #tune.grid_search([True,False]),
            self.sigma_p = 0
            from models.modules_Xin import Swin_Unetr # Swin Unetr
            model = Swin_Unetr(self.num_heads,self.embed_dim,1,self.kernel_size,self.skip,self.num_layers,self.depths,self.mode,config,suffix,param1_scale_im_corrupt, param2_scale_im_corrupt, scaling_input, self.config,self.root,self.subroot,method,all_images_DIP,global_it, self.fixed_hyperparameters_list, self.hyperparameters_list, self.debug, suffix, override_input, self.scanner, self.sub_iter_DIP_already_done, self.override_SC_init)
            model_class = Swin_Unetr
        elif (net == 'DIP_VAE'): # Loading DIP VAE architecture
            model = VAE_DIP_2D(config)
            model_class = VAE_DIP_2D
        elif (net == 'DD'): # Loading Deep Decoder architecture
                #model = DD_2D(config,self.subroot,method,all_images_DIP,global_it, self.fixed_hyperparameters_list, self.hyperparameters_list, self.debug, suffix)
                model = DD_2D(param1_scale_im_corrupt, param2_scale_im_corrupt, scaling_input, self.config,self.root,self.subroot,method,all_images_DIP,global_it, self.fixed_hyperparameters_list, self.hyperparameters_list, self.debug, suffix)
                model_class = DD_2D
        elif (net == 'DD_AE'): # Loading Deep Decoder based autoencoder architecture
            model = DD_AE_2D(config) 
            model_class = DD_AE_2D

        return model, model_class
    
    def generate_nn_output(self, net, param1_scale_im_corrupt, param2_scale_im_corrupt, scaling_input, config, method, image_net_input_torch, PETImage_shape, finetuning, global_it, experiment, suffix, subroot, all_images_DIP):
        # Loading using previous model
        model, model_class = self.choose_net(net, config, method, all_images_DIP, global_it, PETImage_shape, suffix)
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
    

from torch.utils.data import Dataset
class ImagePairDataset(Dataset):
    def __init__(self, image_pairs):
        self.image_pairs = image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        return self.image_pairs[idx]