import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import pandas as pd
import numpy as np
from collections import OrderedDict



class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        def cnn_block(self, channels, kernels, strides, block, stage):
            layers_id = OrderedDict([
                (('conv'+block+str(stage)), nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=kernels[0], stride=strides[0], padding=1)),
                (('bn'+block+str(stage)), nn.BatchNorm2d(channels[1])),
                (('relu'+block+str(stage)), nn.ReLU(inplace = True)),

                (('conv'+block+str(stage+1)), nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=kernels[1], stride=strides[1], padding=1)),
                (('bn'+block+str(stage+1)), nn.BatchNorm2d(channels[2])),
                (('relu'+block+str(stage+1)), nn.ReLU(inplace = True)),

                (('conv'+block+str(stage+2)), nn.Conv2d(in_channels=channels[2], out_channels=channels[3], kernel_size=kernels[2], stride=strides[2], padding=1)),
                (('bn'+block+str(stage+2)), nn.BatchNorm2d(channels[3]))
            ]) 
            return nn.Sequential(layers_id)
        
        store_size = 16
        def calc_filter_size(size, filters, strides, pad):
            in_size = size
            size = ((size - filters[0] + 4)/strides[0]) + 1
            for i in range(1, len(filters)):
                size = ((size - filters[i] + 2)/strides[i]) + 1
            filter_size = in_size + 2*pad - (size - 1)*2
            global store_size
            store_size = size
            
            return int(filter_size)
        
        kernels = [3,3,3]
        strides = [2,1,1]
        
        self.pre_conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pre_bn = nn.BatchNorm2d(32)
        
        self.cnn1 = cnn_block(self, channels = [32,32,64,64], kernels = kernels, strides=strides, block = '1', stage = 1)
        self.res1 = nn.Sequential(nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = calc_filter_size(store_size, kernels, strides, 2), padding = 1, stride = 2),
                                    nn.BatchNorm2d(64))
        
        self.cnn2 = cnn_block(self, channels = [64,128,128,256], kernels = kernels, strides=strides, block = '2', stage = 1)
        self.res2 = nn.Sequential(nn.Conv2d(in_channels = 64, out_channels = 256, kernel_size = calc_filter_size(store_size, kernels, strides, 1), padding = 0, stride = 2),
                                    nn.BatchNorm2d(256))
        
        self.cnn3 = cnn_block(self, channels = [256,512,512,1024], kernels = kernels, strides=strides, block = '3', stage = 1)
        self.res3 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 1024, kernel_size = calc_filter_size(store_size, kernels, strides, 2), padding = 1, stride = 2),
                                    nn.BatchNorm2d(1024))
        
        
        linear_layers = OrderedDict([
            ('avgpool', nn.AvgPool2d(kernel_size = 2)),
            ('flatten', nn.Flatten(start_dim = 1)),
            
            ('fc1', nn.Linear(in_features = 1024, out_features = 512)),
            ('Lrelu1', nn.ReLU(inplace = True)),
            
            ('fc2', nn.Linear(in_features = 512, out_features = 256)),
            ('Lrelu2', nn.ReLU(inplace = True)),
            
            ('fc3', nn.Linear(in_features = 256, out_features = 60)),
            ('Lrelu3', nn.ReLU(inplace = True)),
            ('out', nn.Linear(in_features = 60, out_features = 10))
        ])
        self.linear = nn.Sequential(linear_layers)
        
        
    def forward(self, t):
        t = F.relu(self.pre_bn(self.pre_conv(t)))
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)
    
        x = self.cnn1(t)
        x += self.res1(t)
        x = F.relu(x)
        #x = F.dropout(x, p = 0.25)
        
        t_conv2 = self.res2(x)
        x = self.cnn2(x)
        x += t_conv2
        x = F.relu(x)
        #x = F.dropout(x, p = 0.25)
        
        t_conv3 = self.res3(x)
        x = self.cnn3(x)
        x += t_conv3
        x = F.relu(x)
        
        x = self.linear(x)
        
        return x
        