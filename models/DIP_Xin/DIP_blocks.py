import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
import pytorch_lightning as pl

import functools

def init_weights(net,init_type='normal',init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
        
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

class UnetDeep(nn.Module):
    
    def __init__(self,in_ch,out_ch,depths=2,res=False):
        super().__init__()
        
        self.deep = nn.ModuleList([])
        self.res = res
        self.deep.append(nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=3,stride=1,padding=0),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2)
        ))
        for i in range(depths-2):
            self.deep.append(nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(out_ch, out_ch, kernel_size=3,stride=1,padding=0),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2)
            ))
        if (depths-1):
            self.deep.append(nn.Sequential(
                nn.ReplicationPad2d(1),
                nn.Conv2d(out_ch, out_ch, kernel_size=3,stride=1,padding=0),
                nn.BatchNorm2d(out_ch),
                ))
    def forward(self,x):
        
        for i,blk in enumerate(self.deep):
            x=blk(x)
            if i==0:
                residual = x
        if self.res:
            x += residual
        x = nn.LeakyReLU(0.2)(x)
        return x

class UnetDown(nn.Module):
    
    def __init__(self,in_ch,out_ch,kernel_size=3):
        super().__init__()
    
        self.down = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_ch,out_ch,kernel_size=kernel_size,stride=2,padding=0),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self,x):
        return self.down(x)

class UnetUp(nn.Module):
    
    def __init__(self,in_ch,out_ch,kernel_size=3,mode='bilinear'):
        
        super().__init__()
        
        if mode == 'bilinear':
            self.up = nn.Sequential(nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False),
                                 nn.ReplicationPad2d(1),
                                 nn.Conv2d(in_ch, out_ch, 3, stride=(1, 1), padding=0),
                                 nn.BatchNorm2d(out_ch),
                                 nn.LeakyReLU(0.2))
        elif mode == 'convt':
            self.up = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1))
    def forward(self,x):
        return self.up(x)

class UnetSkipAdd(nn.Module):
    
    def __init__(self,in_ch=1,embed_ch=16,out_ch=1,kernel_size=3,skip=3,num_layer=3,depths=2,mode='bilinear'):
        super().__init__()
        self.num_layer = num_layer
        self.skip = skip
        self.encoder_deep = nn.ModuleList([])
        self.encoder_down = nn.ModuleList([])
        self.decoder_deep = nn.ModuleList([])
        self.decoder_up = nn.ModuleList([])
        self.encoder_deep.append(UnetDeep(in_ch,embed_ch,depths=depths))
        for i in range(num_layer):
            self.encoder_down.append(UnetDown(embed_ch*(2**(i)),embed_ch*(2**(i)),kernel_size=kernel_size))
            self.encoder_deep.append(UnetDeep(embed_ch*(2**(i)),embed_ch*(2**(i+1)),res=False))
            self.decoder_up.append(UnetUp(embed_ch*(2**(num_layer-i)),embed_ch*(2**(num_layer-i-1)),mode=mode))
            self.decoder_deep.append(UnetDeep(embed_ch*(2**(num_layer-i-1)),embed_ch*(2**(num_layer-i-1)),res=False))
        self.conv_last = nn.Conv2d(embed_ch,out_ch,kernel_size=1,padding=0)
        
    def forward(self,x):
        out = x
        encoder_out = []
        for i in range(self.num_layer):
            out = self.encoder_deep[i](out)
            encoder_out.append(out)
            out = self.encoder_down[i](out)
        out = self.encoder_deep[self.num_layer](out)
       
            
        
        for i in range(self.num_layer):

            out = self.decoder_up[i](out)
            
            if self.skip>i :
               
                out = out + encoder_out[self.num_layer-1-i]
            
            out = self.decoder_deep[i](out)
        
        out = self.conv_last(out)
        return out
    
            
        



