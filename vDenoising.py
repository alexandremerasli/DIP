## Python libraries

# Pytorch
import torch

# Local files to import
from utils.utils_func import *
from vReconstruction import vReconstruction
from vGeneral import vGeneral

import abc
class vDenoising(vGeneral):
    @abc.abstractmethod
    def __init__(self,config,root):
        print('__init__')

    def initializeSpecific(self, hyperparameters_config,root):
        self.createDirectoryAndConfigFile(hyperparameters_config)
        # Specific hyperparameters for denoising module (Do it here to have raytune hyperparameters_config hyperparameters selection)
        if hyperparameters_config["input"] == 'uniform': # Do not standardize or normalize if uniform, otherwise NaNs
            hyperparameters_config["scaling"] = "nothing"
        self.scaling_input = hyperparameters_config["scaling"]
        if self.method == 'Gong':
            # hyperparameters_config["scaling"] = 'positive_normalization' # Will still introduce bias as mu can be negative
            hyperparameters_config["scaling"] = "nothing"

        # Loading DIP input
        # Creating random image input for DIP while we do not have CT, but need to be removed after
        create_input(self.net,self.PETImage_shape,hyperparameters_config) # to be removed when CT will be used instead of random input. DO NOT PUT IT IN BLOCK 2 !!!
        # Loading DIP input (we do not have CT-map, so random image created in block 1)
        self.image_net_input = load_input(self.net,self.PETImage_shape,hyperparameters_config) # Scaling of network input. DO NOT CREATE RANDOM INPUT IN BLOCK 2 !!! ONLY AT THE BEGINNING, IN BLOCK 1    
        #image_atn = fijii_np(self.subroot+'Data/database_v2/' + self.phantom + '/' + self.phantom + '_atn.raw',shape=(self.PETImage_shape))
        #write_image_tensorboard(writer,image_atn,"Attenuation map (FULL CONTRAST)",self.suffix,image_gt,0,full_contrast=True) # Attenuation map in tensorboard
        image_net_input_scale = rescale_imag(self.image_net_input,self.scaling_input)[0] # Rescale of network input
        # DIP input image, numpy --> torch
        self.image_net_input_torch = torch.Tensor(image_net_input_scale)
        # Adding dimensions to fit network architecture
        if (self.net == 'DIP' or self.net == 'DIP_VAE'):
            self.image_net_input_torch = self.image_net_input_torch.view(1,1,self.PETImage_shape[0],self.PETImage_shape[1]) # For DIP
        else:
            if (self.net == 'DD'):
                input_size_DD = int(self.PETImage_shape[0] / (2**hyperparameters_config["d_DD"])) # if original Deep Decoder (i.e. only with decoder part)
                self.image_net_input_torch = self.image_net_input_torch.view(1,hyperparameters_config["k_DD"],input_size_DD,input_size_DD) # For Deep Decoder, if original Deep Decoder (i.e. only with decoder part)
            elif (self.net == 'DD_AE'):
                input_size_DD = self.PETImage_shape[0] # if auto encoder based on Deep Decoder
                self.image_net_input_torch = self.image_net_input_torch.view(1,1,input_size_DD,input_size_DD) # For Deep Decoder, if auto encoder based on Deep Decoder
        torch.save(self.image_net_input_torch,self.subroot + 'Data/initialization/image_' + self.net + '_input_torch.pt')

    def runComputation(self,config,hyperparameters_config,root):
        # Scaling of x_label image
        image_corrupt_input_scale,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt = rescale_imag(self.image_corrupt,self.scaling_input) # Scaling of x_label image

        # Corrupted image x_label, numpy --> torch
        self.image_corrupt_torch = torch.Tensor(image_corrupt_input_scale)
        # Adding dimensions to fit network architecture
        self.image_corrupt_torch = self.image_corrupt_torch.view(1,1,self.PETImage_shape[0],self.PETImage_shape[1])

        # Training model with sub_iter_DIP iterations
        model = self.train_process(hyperparameters_config, self.finetuning, self.processing_unit, self.sub_iter_DIP, self.method, self.admm_it, self.image_net_input_torch, self.image_corrupt_torch, self.net, self.PETImage_shape, self.test, self.checkpoint_simple_path, self.name_run, self.subroot) # Not useful to make iterations, we just want to initialize writer. admm_it must be set to -1, otherwise seeking for a checkpoint file...
        if (self.net == 'DIP_VAE'):
            out, mu, logvar, z = model(self.image_net_input_torch)
        else:
            out = model(self.image_net_input_torch)

        # Descaling like at the beginning
        out_descale = descale_imag(out,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)
        # Saving image output
        save_img(out_descale, self.net_outputs_path)

    def train_process(self, hyperparameters_config, finetuning, processing_unit, sub_iter_DIP, method, admm_it, image_net_input_torch, image_corrupt_torch, net, PETImage_shape, test, checkpoint_simple_path, name_run, subroot):
        # Implements Dataset
        train_dataset = torch.utils.data.TensorDataset(image_net_input_torch, image_corrupt_torch)
        # train_dataset = Dataset(image_net_input_torch, image_corrupt_torch)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1) # num_workers is 0 by default, which means the training process will work sequentially inside the main process

        # Choose network architecture as model
        model, model_class = choose_net(net, hyperparameters_config, method)

        #checkpoint_simple_path = 'runs/' # To log loss in tensorboard thanks to Logger
        checkpoint_simple_path_exp = subroot+'Block2/checkpoint/'+format(test)  + '/' + suffix_func(hyperparameters_config) + '/'

        model = load_model(image_net_input_torch, hyperparameters_config, finetuning, admm_it, model, model_class, method, checkpoint_simple_path_exp, training=True)

        # Start training
        print('Starting optimization, iteration',admm_it)
        trainer = create_pl_trainer(finetuning, processing_unit, sub_iter_DIP, admm_it, net, checkpoint_simple_path, test, checkpoint_simple_path_exp,name=name_run)

        trainer.fit(model, train_dataloader)

        return model