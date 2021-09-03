import torch
import torch.nn as nn
import pytorch_lightning as pl
import os

class DD_2D_lightning(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.lr = config['lr']
        self.opti_DIP = config['opti_DIP']

        d = config["d_DD"] # Number of layers
        k = config['k_DD'] # Number of channels, depending on how much noise we mant to remove. Small k = less noise, but less fit
        self.num_channels_up = [k]*(d+1) + [1]
        
        self.layers = nn.ModuleList([])
        for i in range(len(self.num_channels_up)-2):       
            self.layers.append(nn.Sequential(
                               #nn.ReplicationPad2d(1), # if kernel size = 3
                               nn.Conv2d(self.num_channels_up[i], self.num_channels_up[i+1], 1, stride=1),
                               nn.Upsample(scale_factor=2, mode='bilinear'),
                               nn.ReLU(),
                               nn.BatchNorm2d(self.num_channels_up[i+1])))

        self.last_layers = nn.Sequential(nn.Conv2d(self.num_channels_up[-2], self.num_channels_up[-1], 1, stride=1))
        
        self.positivity = nn.ReLU() # Final ReLU to enforce positivity of ouput image
        # self.positivity = nn.SiLU() # Final SiLU, smoother than ReLU but not positive
        # self.positivity = nn.Softplus() # Final SiLU to enforce positivity of ouput image, smoother than ReLU

    def forward(self, x):
        out = x
        for i in range(len(self.num_channels_up)-2):
            out = self.layers[i](out)
        out = self.last_layers(out)
        out = self.positivity(out)
        return out

    def DIP_loss(self, out, image_corrupt_torch):
        return torch.nn.MSELoss()(out, image_corrupt_torch) # for DIP and DD

    def training_step(self, train_batch, batch_idx):
        image_net_input_torch, image_corrupt_torch = train_batch
        out = self.forward(image_net_input_torch)
        loss = self.DIP_loss(out, image_corrupt_torch)
        self.log('loss_monitor', loss)
        return loss

    def configure_optimizers(self):
        # Optimization algorithm according to command line

        """
        Optimization of the DNN with SGLD
        """

        if (self.opti_DIP == 'Adam'):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5E-8) # Optimizing using Adam
        elif (self.opti_DIP == 'LBFGS' or self.opti_DIP is None): # None means no argument was given in command line
            optimizer = torch.optim.LBFGS(self.parameters(), lr=self.lr, history_size=10, max_iter=4) # Optimizing using L-BFGS
        return optimizer