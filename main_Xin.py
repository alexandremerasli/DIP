

# 模型相关
import torch
from models.DIP_2D_Xin import DIP_2D
import pytorch_lightning as pl
from config_Xin import *

# 数据处理相关
import numpy as np
import matplotlib.pyplot as plt
import os
from functools import partial
from ray import tune

# 自定义函数
from utils.pre_utils_Xin import *


def main_computation(config,root):

    # write random seed in a file to get it in network architectures
    os.system("rm -rf " + os.getcwd() +"/seed.txt")
    file_seed = open(os.getcwd() + "/seed.txt","w+")
    file_seed.write(str(settings_config["random_seed"]))
    file_seed.close()

    # 定义各种文件的路径 
    path_noisy="/home/meraslia/workspace_reco/nested_admm/data/Algo/Data/initialization/image4_0/BSREM_30it/replicate_1/BSREM_it30.img" # 含噪图片位置
    path_output = "/home/meraslia/workspace_reco/nested_admm/data/Algo/Test_Xin"            # 输出图片位置
    PETImage_shape=(112,112,1)  # 输入图片的大小
    path_input = "/home/xzhang/Documents/我的模型/images/noise_images/" # 输入图片位置


    # 读取不同的含噪声文件，目前只有一个，将其做rescale和格式转换tensor 1,1,112,112
    image_corrupt=fijii_np(path_noisy,PETImage_shape) # 读取图片并将图片转换成numpy array
    image_corrupt_input_scaled,param1_scale_im_corrupt,param2_scale_im_corrupt = rescale_imag(image_corrupt,"positive_normalization") # 标准化图片, 减去平均值，除以标准差，参数1是mean，参数2是std
    image_corrupt_torch = torch.Tensor(image_corrupt_input_scaled)
    image_corrupt_torch = image_corrupt_torch.view(1,1,PETImage_shape[0],PETImage_shape[1],PETImage_shape[2])
    image_corrupt_torch = image_corrupt_torch[:,:,:,:,0]

    # ground_truth = np.zeros(PETImage_shape)
    #ground_truth = np.load("/home/xzhang/Documents/我的模型/images/noise_images/image4_0.npy")
    ground_truth = fijii_np("/home/meraslia/workspace_reco/nested_admm/data/Algo/Data/initialization/image4_0/BSREM_30it/replicate_1/BSREM_it30.img",shape=(PETImage_shape),type_im='<f')

    # 读取不同的噪声文件，用于输入。并且同样经过rescale处理
    image_net_input =np.random.uniform(PETImage_shape)
    image_net_input_torch = torch.Tensor(ground_truth)

    image_net_input_torch = image_net_input_torch.view(1,1,PETImage_shape[0],PETImage_shape[1],PETImage_shape[2])
    image_net_input_torch = image_net_input_torch[:,:,:,:,0]

    # 加载数据
    train_dataset = torch.utils.data.TensorDataset(image_net_input_torch, image_corrupt_torch)
    # train_dataset = [image_net_input_torch,image_corrupt_torch]
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1) 

    # 加载模型
    model = DIP_2D(param1_scale_im_corrupt, param2_scale_im_corrupt, config,root,
                "nested",all_images_DIP="True",global_it=-100, suffix='aaa',last_iter=-1,ground_truth=ground_truth, scaling_input="positive_normalization")
    model_class = DIP_2D

    # 定义tensorboard
    #checkpoint_simple_path = '/home/xzhang/Documents/我的模型/lightning_logs'
    checkpoint_simple_path = os.getcwd() + '/lightning_logs'
    # experiment = 24
    # name = 'my_model'

    logger = pl.loggers.TensorBoardLogger(save_dir=checkpoint_simple_path)#version=format(experiment), name=name)
    # Early stopping callback
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    early_stopping_callback = EarlyStopping(monitor="SUCCESS", mode="max",stopping_threshold=0.9,patience=np.inf) # SUCCESS will be 1 when ES if found, which is greater than stopping_threshold = 0.9
    trainer = pl.Trainer(max_epochs=config["sub_iter_DIP"],log_every_n_steps=1, logger=logger, callbacks=[early_stopping_callback])    
    # trainer = pl.Trainer(max_epochs=config["sub_iter_DIP"],log_every_n_steps=1,logger=logger)#, callbacks=[checkpoint_callback, tuning_callback, early_stopping_callback], logger=logger,gpus=gpus, accelerator=accelerator, profiler="simple")

    # 训练模型
    trainer.fit(model, train_dataloader)
    out = model(image_net_input_torch)



    image_out = out.view(PETImage_shape[0],PETImage_shape[1],PETImage_shape[2]).detach().numpy()
    image_concat = np.concatenate((image_corrupt, destand_numpy_imag(image_out,param1_scale_im_corrupt,param2_scale_im_corrupt)), axis=1)
    image_reversed =np.max(image_concat)-image_concat

    # plt.imshow(image_reversed, cmap='gray')
    # plt.show()  

root = os.getcwd()

# tune.run(partial(main_computation,root=root), config=config_tune,local_dir = os.getcwd() + '/lightning_logs')
main_computation(config,root)