import torch
from torch import nn
import models
from collections import OrderedDict
from argparse import Namespace
import yaml
import os


class Model(nn.Module):
    def __init__(self, encode_layer, linear_layer):
        super().__init__()
        self.encode_layer =encode_layer
        self.linear_layer = linear_layer

    def forward(self, x, out='z'):
        return self.linear_layer(self.encode_layer(x,out))
    

class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False


class EncodeProject(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        cifar_head = (hparams.data ==
                      'cifar' or hparams.data == 'cifar100') #and hparams.encoder_type == 'cifar_head'

        if hparams.arch == 'ResNet50':
            self.convnet = models.resnet.ResNet50(
                cifar_head=cifar_head)
            self.encoder_dim = 2048
        elif hparams.arch == 'ResNet18':
            self.convnet = models.resnet.ResNet18(
                cifar_head=cifar_head)
            self.encoder_dim = 512
        elif hparams.arch == 'ResNet32':
            self.convnet = models.resnet.ResNet32()
            self.encoder_dim = 64
        else:
            raise NotImplementedError

        num_params = sum(p.numel()
                         for p in self.convnet.parameters() if p.requires_grad)

        print(
            f'======> {num_params/1e6:.3f}M parameters')


    def forward(self, x, out='h'):
        return self.convnet(x)
        
