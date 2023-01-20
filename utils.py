# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, models, transforms
#from resnet import resnet50
from torchvision.models import resnet50, resnet101, resnet152, inception_v3
import random
import time
import argparse
import math
import json
import pickle
import copy
import numpy as np
from PIL import ImageFile
import torch.utils.model_zoo as model_zoo  
from efficientnet import EfficientNet
ImageFile.LOAD_TRUNCATED_IMAGES = True


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def source_model_imagenet(base_model):
    if base_model.startswith('efficient'):
        model = EfficientNet.from_pretrained(base_model)
    else:
        model = eval(base_model)(pretrained = True)
    return model

def source_model_place365(base_model, model_path='./resnet50_places365_python36.pth.tar'):
    assert base_model == 'resnet50'
    model = resnet50(pretrained = False, num_classes = 365)
    state_dict = torch.load(model_path, pickle_module=pickle)['state_dict']
    state_dict_new = {}
    for k, v in state_dict.items():
        state_dict_new[k[len('module.'):]] = v
    model.load_state_dict(state_dict_new)
    return model

def transform_train(image_size, keep_aspect):
    crop_size = {299: 320, 224: 256}[image_size]
    operations = []
    if keep_aspect:
        operations.append(transforms.Resize(crop_size))
        #operations.append(transforms.CenterCrop((crop_size, crop_size)))
    else:
        operations.append(transforms.Resize((crop_size, crop_size)))
    operations.append(transforms.RandomHorizontalFlip())
    operations.append(transforms.RandomCrop((image_size, image_size)))
    operations.append(transforms.ToTensor())
    operations.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return transforms.Compose(operations)

def transform_test(image_size, keep_aspect):
    crop_size = {299: 320, 224: 256}[image_size]
    operations = []
    if keep_aspect:
        operations.append(transforms.Resize(crop_size))
        operations.append(transforms.CenterCrop((image_size, image_size)))
    else:
        operations.append(transforms.Resize((crop_size, crop_size)))
        operations.append(transforms.CenterCrop((image_size, image_size)))
    operations.append(transforms.ToTensor())
    operations.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return transforms.Compose(operations)
                                                     
class AdvancedImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, sample_rate=1, seed=0):
        super(AdvancedImageFolder, self).__init__(root, transform=transform)
        random.seed(seed)
        random.shuffle(self.samples)
        self.samples = self.samples[:(1+int(sample_rate * len(self.samples)))]
        self.targets = [s[1] for s in self.samples]
        self.imgs = self.samples

    def __getitem__(self, index):
        return super(AdvancedImageFolder, self).__getitem__(index), self.imgs[index]#return image path

def set_frozen(models):
    for model in models:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

def set_trainable(models):
    for model in models:
        model.train()
        for param in model.parameters():
            param.requires_grad = True

def set_lr_cosine(lr_init, epoch, num_epochs, optimizer):
    lr = 0.5 * lr_init * (1 + math.cos(math.pi * epoch / num_epochs))
    optimizer.param_groups[0]['lr'] = lr
    return lr 

def set_lr_step(lr_init, epoch, num_epochs, optimizer):
    lr = lr_init if epoch < (2 * num_epochs / 3) else lr_init * 0.1
    optimizer.param_groups[0]['lr'] = lr
    return lr 

