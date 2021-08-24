import torch
import torch.nn as nn
import pytorch_lightning as pl
import os

class ConvNet3D_real_lightning(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.lr = config['lr']
        self.opti_DIP = config['opti_DIP']

        L_relu = 0.2
        num_channel = [16, 32, 64, 128]
        pad = [0, 0]

        num_groups = [2,4,8,16]
        self.deep1 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(1, num_channel[0], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[0]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[0], num_channel[0], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[0]),
                                   nn.LeakyReLU(L_relu))

        self.down1 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[0], num_channel[0], 3, stride=(2, 2), padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[0]),
                                   nn.LeakyReLU(L_relu))

        self.deep2 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[0], num_channel[1], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[1]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[1], num_channel[1], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[1]),
                                   nn.LeakyReLU(L_relu))

        self.down2 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[1], num_channel[1], 3, stride=(2, 2), padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[1]),
                                   nn.LeakyReLU(L_relu))

        self.deep3 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[1], num_channel[2], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[2]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[2], num_channel[2], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[2]),
                                   nn.LeakyReLU(L_relu))

        self.down3 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[2], num_channel[2], 3, stride=(2, 2), padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[2]),
                                   nn.LeakyReLU(L_relu))

        self.deep4 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[2], num_channel[3], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[3]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[3], num_channel[3], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[3]),
                                   nn.LeakyReLU(L_relu))

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False),
                                 nn.ReplicationPad2d(1),
                                 nn.Conv2d(num_channel[3], num_channel[2], 3, stride=(1, 1), padding=pad[0]),
                                 nn.BatchNorm2d(num_channel[2]),
                                 nn.LeakyReLU(L_relu))

        self.deep5 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[2], num_channel[2], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[2]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[2], num_channel[2], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[2]),
                                   nn.LeakyReLU(L_relu))

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False),
                                 nn.ReplicationPad2d(1),
                                 nn.Conv2d(num_channel[2], num_channel[1], 3, stride=(1, 1), padding=pad[0]),
                                 nn.BatchNorm2d(num_channel[1]),
                                 nn.LeakyReLU(L_relu))

        self.deep6 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[1], num_channel[1], (3, 3), stride=1, padding=pad[0]),
                                   nn.BatchNorm2d(num_channel[1]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[1], num_channel[1], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[1]),
                                   nn.LeakyReLU(L_relu))

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False),
                                 nn.ReplicationPad2d(1),
                                 nn.Conv2d(num_channel[1], num_channel[0], 3, stride=(1, 1), padding=pad[0]),
                                 nn.BatchNorm2d(num_channel[0]),
                                 nn.LeakyReLU(L_relu))

        self.deep7 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[0], num_channel[0], (3, 3), stride=1, padding=pad[0]),
                                   nn.BatchNorm2d(num_channel[0]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[0], 1, (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(1))

        self.positivity = nn.ReLU() # Final ReLU to enforce positivity of ouput image

    def forward(self, x):
        out1 = self.deep1(x)
        out = self.down1(out1)
        out2 = self.deep2(out)
        out = self.down2(out2)
        out3 = self.deep3(out)
        out = self.down3(out3)
        out = self.deep4(out)

        out = self.up1(out)
        out_skip1 = out3 + out
        out = self.deep5(out_skip1)
        out = self.up2(out)
        out_skip2 = out2 + out
        out = self.deep6(out_skip2)
        out = self.up3(out)
        out_skip3 = out1 + out
        out = self.deep7(out_skip3)
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
        elif (self.opti_DIP == 'LBFGS' or opti_DIP is None): # None means no argument was given in command line
            optimizer = torch.optim.LBFGS(self.parameters(), lr=self.lr, history_size=10, max_iter=4) # Optimizing using L-BFGS
        return optimizer