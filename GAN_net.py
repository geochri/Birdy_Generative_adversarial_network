# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 16:55:05 2019

@author: WT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class D_net(nn.Module):
    def __init__(self):
        super(D_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(1)
        self.drop = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(8*8, 1)
        
    def forward(self, x):
        x = self.lrelu(self.bn1(self.conv1(x))); #print(x.size())
        x = self.drop(x)
        x = self.lrelu(self.bn2(self.conv2(x))); #print(x.size())
        x = self.lrelu(self.bn3(self.conv3(x))); #print(x.size())
        x = self.drop(x)
        x = self.lrelu(self.bn4(self.conv4(x))); #print(x.size())
        x = self.lrelu(self.bn5(self.conv5(x))); #print(x.size())
        x = x.view(-1, 8*8)
        x = torch.sigmoid(self.fc1(x))
        return x
    
class G_net(nn.Module):
    def __init__(self, input_size=8*8, batch_size=25):
        super(G_net, self).__init__()
        self.batch_size = batch_size
        self.lrelu = nn.LeakyReLU(0.2)
        self.dconv1 = nn.ConvTranspose2d(1, 512, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.dconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.dconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.dconv5 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = x.view(-1, 1, 8, 8)
        x = self.lrelu(self.bn1(self.dconv1(x)));# print(x.size())
        x = self.lrelu(self.bn2(self.dconv2(x)));# print(x.size())
        x = self.lrelu(self.bn3(self.dconv3(x)));# print(x.size())
        x = self.lrelu(self.bn4(self.dconv4(x)));# print(x.size())
        x = F.tanh(self.dconv5(x, output_size=torch.Size([self.batch_size, 3, 128, 128])))
        return x