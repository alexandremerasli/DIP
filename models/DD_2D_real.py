#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:24:14 2020

@author: alexandre
"""

import torch
import torch.nn as nn

class DD_2D_real(nn.Module):
    def __init__(self,
        num_output_channels=1, # gray image 
        num_channels_up=[32]*3, # k0=k1=k2=32 channels 
        filter_size_up=1,
        need_sigmoid=True, 
        pad ='reflection', 
        upsample_mode='bilinear', 
        act_fun=nn.ReLU(),
        bn_before_act = False,
        bn_affine = True,
        upsample_first = True):

        super(DD_2D_real, self).__init__()

        self.num_output_channels = num_output_channels
        self.num_channels_up = num_channels_up
        self.filter_size_up = filter_size_up
        self.need_sigmoid = need_sigmoid
        self.pad = pad
        self.upsample_mode = upsample_mode
        self.act_fun = act_fun
        self.bn_before_act = bn_before_act
        self.bn_affine = bn_affine
        self.upsample_first = upsample_first
    
        self.num_channels_up = self.num_channels_up + [self.num_channels_up[-1],self.num_channels_up[-1]]
        n_scales = len(self.num_channels_up) 
        
        if not (isinstance(self.filter_size_up, list) or isinstance(self.filter_size_up, tuple)) :
            self.filter_size_up   = [self.filter_size_up]*n_scales
        layers = []

        
        for i in range(len(self.num_channels_up)-1):
            
            if self.upsample_first:
                layers.append(self.conv( self.num_channels_up[i], self.num_channels_up[i+1],  self.filter_size_up[i], 1, pad=self.pad))
                if self.upsample_mode!='none' and i != len(self.num_channels_up)-2:
                    layers.append(nn.Upsample(scale_factor=2, mode=self.upsample_mode))
                #layers.append(nn.functional.interpolate(size=None,scale_factor=2, mode=self.upsample_mode)) 
            else:
                if self.upsample_mode!='none' and i!=0:
                    layers.append(nn.Upsample(scale_factor=2, mode=self.upsample_mode))
                #layers.append(nn.functional.interpolate(size=None,scale_factor=2, mode=self.upsample_mode)) 
                layers.append(self.conv( self.num_channels_up[i], self.num_channels_up[i+1],  self.filter_size_up[i], 1, pad=self.pad))        
            
            if i != len(self.num_channels_up)-1: 
                if(self.bn_before_act): 
                    layers.append(nn.BatchNorm2d( self.num_channels_up[i+1] ,affine=self.bn_affine))
                layers.append(self.act_fun)
                if(not self.bn_before_act): 
                    layers.append(nn.BatchNorm2d( self.num_channels_up[i+1], affine=self.bn_affine))
          
        layers.append(self.conv( self.num_channels_up[-1], self.num_output_channels, 1, pad=self.pad))
        #if self.need_sigmoid:
        #    layers.append(nn.Sigmoid())

        # Replace Sigmoid by ReLU to ensure positive image ?
        layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)
        

    def forward(self, x):
        return self.model(x)

    def add_module(self, module):
        self.add_module(str(len(self) + 1), module)

    def conv(self,in_f, out_f, kernel_size, stride=1, pad='zero'):
        padder = None
        to_pad = int((kernel_size - 1) / 2)
        if self.pad == 'reflection':
            padder = nn.ReflectionPad2d(to_pad)
            to_pad = 0
      
        convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=False)

        layers = filter(lambda x: x is not None, [padder, convolver])
        return nn.Sequential(*layers)