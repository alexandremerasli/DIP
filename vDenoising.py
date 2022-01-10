## Python libraries

# Pytorch
import torch

# Useful
import os
from datetime import datetime
from functools import partial
from ray import tune

# Math
import numpy as np
import matplotlib.pyplot as plt

# Local files to import
from utils.utils_func import *
from vReconstruction import vReconstruction

import abc
class vDenoising(abc.ABC):
    def __init__(self,config,args,root):
        vReconstruction.__init__(self,config,args,root)

    def initializeSpecific(self, config, args, root):
        vReconstruction.initializeSpecific(self,config,args,root)

    def runRayTune(self,config,args,root):
        config_combination = 1
        for i in range(len(config)):
            config_combination *= len(list(list(config.values())[i].values())[0])

        if args.proc == 'CPU':
            resources_per_trial = {"cpu": 1, "gpu": 0}
        elif args.proc == 'GPU':
            resources_per_trial = {"cpu": 0, "gpu": 0.1} # "gpu": 1 / config_combination
            #resources_per_trial = {"cpu": 0, "gpu": 1} # "gpu": 1 / config_combination
        elif args.proc == 'both':
            resources_per_trial = {"cpu": 10, "gpu": 1} # not efficient

        #reporter = CLIReporter(
        #    parameter_columns=['lr'],
        #    metric_columns=['mse'])

        # Start tuning of hyperparameters = start each admm computation in parallel
        #try: # resume previous run (if it exists)
        #    anaysis_raytune = tune.run(partial(admm_loop,args=args,root=os.getcwd()), config=config, local_dir = os.getcwd() + '/runs', name=suffix_func(config) + str(args.max_iter), resources_per_trial = resources_per_trial, resume = "ERRORED_ONLY")#, progress_reporter = reporter)
        #except: # do not resume previous run because there is no previous one
        #   anaysis_raytune = tune.run(partial(admm_loop,args=args,root=os.getcwd()), config=config, local_dir = os.getcwd() + '/runs', name=suffix_func(config) + "_max_iter=" + str(args.max_iter), resources_per_trial = resources_per_trial)#, progress_reporter = reporter)



        tune.run(partial(self.do_everything,args=args,root=root), config=config,local_dir = os.getcwd() + '/runs', resources_per_trial = resources_per_trial)#, progress_reporter = reporter)
    
    def do_everything(self,config,args,root):
        self.initializeSpecific(config,args,root)
        self.runDenoiser(root)

    def runDenoiser(self,root):
        # Scaling of x_label image
        image_corrupt_input_scale,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt = rescale_imag(self.image_corrupt,self.scaling_input) # Scaling of x_label image

        # Corrupted image x_label, numpy --> torch
        self.image_corrupt_torch = torch.Tensor(image_corrupt_input_scale)
        # Adding dimensions to fit network architecture
        self.image_corrupt_torch = self.image_corrupt_torch.view(1,1,self.PETImage_shape[0],self.PETImage_shape[1])

        # Training model with sub_iter_DIP iterations
        model = self.train_process(self.config, self.finetuning, self.processing_unit, self.sub_iter_DIP, self.admm_it, self.image_net_input_torch, self.image_corrupt_torch, self.net, self.PETImage_shape, self.test, self.checkpoint_simple_path, self.name_run, self.subroot) # Not useful to make iterations, we just want to initialize writer. admm_it must be set to -1, otherwise seeking for a checkpoint file...
        if (self.net == 'DIP_VAE'):
            out, mu, logvar, z = model(self.image_net_input_torch)
        else:
            out = model(self.image_net_input_torch)

        # Descaling like at the beginning
        out_descale = descale_imag(out,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)
        # Saving image output
        save_img(out_descale, self.net_outputs_path)
        
    def train_process(self, config, finetuning, processing_unit, sub_iter_DIP, admm_it, image_net_input_torch, image_corrupt_torch, net, PETImage_shape, test, checkpoint_simple_path, name_run, subroot):
        # Implements Dataset
        train_dataset = torch.utils.data.TensorDataset(image_net_input_torch, image_corrupt_torch)
        # train_dataset = Dataset(image_net_input_torch, image_corrupt_torch)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1) # num_workers is 0 by default, which means the training process will work sequentially inside the main process

        # Choose network architecture as model
        model, model_class = choose_net(net, config)

        #checkpoint_simple_path = 'runs/' # To log loss in tensorboard thanks to Logger
        checkpoint_simple_path_exp = subroot+'Block2/checkpoint/'+format(test)  + '/' + suffix_func(config) + '/'

        model = load_model(image_net_input_torch, config, finetuning, admm_it, model, model_class, subroot, checkpoint_simple_path_exp, training=True)

        # Start training
        print('Starting optimization, iteration',admm_it)
        trainer = create_pl_trainer(finetuning, processing_unit, sub_iter_DIP, admm_it, net, checkpoint_simple_path, test, checkpoint_simple_path_exp,name=name_run)

        trainer.fit(model, train_dataloader)

        return model