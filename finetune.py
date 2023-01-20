# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import sys
import time
import argparse
import configparser
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from PIL import ImageFile
#from visulize import save_heatmap, tsne
from utils import source_model_imagenet, source_model_place365, AdvancedImageFolder, transform_train, transform_test, set_frozen, set_trainable, set_lr_step, set_lr_cosine
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description = 'Inductive Transfer Learning Algorithms')
parser.add_argument('--task_name')
parser.add_argument('--data_dir')
parser.add_argument('--base_model', default = 'resnet50')
parser.add_argument('--task_conf', default = './task_config.ini')
parser.add_argument('--max_iters', type = int, default = 9000)
parser.add_argument('--sample_rate', type = float, default = 1)
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--image_size', type = int, default = 224)
parser.add_argument('--batch_size', type = int, default = 48)
parser.add_argument('--lr_init', type = float, default = 0.01)
parser.add_argument('--nesterov', type = int, default = 1)
parser.add_argument('--wd', type = float, default = 1e-4)
parser.add_argument('--regularizer', choices = ['fe', 'l2', 'l2sp', 'bss', 'delta', 'mixup', 'cutmix', 'rifle', 'smile'], default = 'l2')

args = parser.parse_args()

config = configparser.ConfigParser()
args = parser.parse_args()
print(torch.__version__)
config.read(args.task_conf)

sec = args.task_name.split('-')[0]
args.source_task = config.get(sec, 'source_task')
args.pretrained_weight = config.get(sec, 'pretrained_weight')
args.keep_aspect = config.getboolean(sec, 'keep_aspect')
args.alpha_l2sp = config.getfloat(sec, 'alpha_l2sp')
args.alpha_delta = config.getfloat(sec, 'alpha_delta')
args.alpha_bss = config.getfloat(sec, 'alpha_bss')
args.alpha_rifle = config.getint(sec, 'alpha_rifle')

print(args)

device = torch.device("cuda:0")

data_transforms = {'train': transform_train(args.image_size, args.keep_aspect), 
                    'test': transform_test(args.image_size, args.keep_aspect)}
set_names = list(data_transforms.keys())
image_datasets = {x: AdvancedImageFolder(os.path.join(args.data_dir, x),data_transforms[x], sample_rate=args.sample_rate if x == 'train' else 1, seed=args.seed)
                  for x in set_names}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=(args.batch_size if x == 'train' else 8),
                                             shuffle=(x=='train'), num_workers=2)
              for x in set_names}
dataset_sizes = {x: len(image_datasets[x]) for x in set_names}
num_classes = len(image_datasets['train'].classes)
criterion = nn.CrossEntropyLoss()

#usef for CutMix
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixup_data(x, y, index, alpha=1.0, use_cuda=True):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    if lam < 0.5:
        lam = 1 - lam
    batch_size = x.size()[0]
    if index is None:
        index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion_hard(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

logsoftmax = nn.LogSoftmax(dim = 1)
def mixup_criterion_soft(pred, y_a, y_b, lam):
    log_probs = logsoftmax(pred)
    loss_a = (-y_a * log_probs).mean(0).sum()   
    loss_b = (-y_b * log_probs).mean(0).sum()
    loss = lam * loss_a + (1 - lam) * loss_b   
    return loss

base_model = args.base_model
if args.source_task == 'imagenet':
    model_source = source_model_imagenet(base_model).to(device)
    model_target = source_model_imagenet(base_model).to(device)
elif args.source_task == 'places365':
    model_source = source_model_place365(base_model, args.pretrained_weight).to(device)
    model_target = source_model_place365(base_model, args.pretrained_weight).to(device)
set_frozen([model_source])

feature_dim = model_source.fc.weight.shape[1]
print('feature_dim = ', feature_dim)
model_target.fc = nn.Linear(feature_dim, num_classes)
model_target.to(device)

layer_outputs_source = []
layer_outputs_target = []
def for_hook_source(module, input, output):
    layer_outputs_source.append(output)
def for_hook_target(module, input, output):
    layer_outputs_target.append(output)

hook_layers = {'efficientnet-b4':['_blocks.31'], 'resnet50':['layer4']}
def register_hook(model, func, hook_layers):
    for name, layer in model.named_modules():
        if name in hook_layers:
            print('hooked:', name)
            layer.register_forward_hook(func)

if args.regularizer in ['bss', 'smile', 'delta']: 
    register_hook(model_source, for_hook_source, hook_layers[base_model])
    register_hook(model_target, for_hook_target, hook_layers[base_model])


class TrainIter():
    def __init__(self, args):
        pass

    def loss(self, model, inputs, labels, iter_id):
        raise NotImplementedError('No valid algorithm specified')

class FinetuneBase(TrainIter):
    def __init__(self, args):
        pass

    def loss(self, model, inputs, labels, iter_id):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        return loss, outputs

class FinetuneMixup(TrainIter):
    def __init__(self, args):
        self.alpha = 0.2

    def loss(self, model, inputs, labels, iter_id):
        index_perm = torch.randperm(inputs.shape[0]).cuda()
        inputs_mix, targets_a, targets_b, lam = mixup_data(inputs, labels, index_perm, self.alpha)
        outputs = model(inputs_mix)
        loss = mixup_criterion_hard(criterion, outputs, targets_a, targets_b, lam)
        return loss, outputs

class FinetuneCutMix(TrainIter):
    def __init__(self, args):
        self.alpha = 0.2

    def loss(self, model, inputs, labels, iter_id):
        r = np.random.rand(1)
        if r < 0.5:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        else:
            lam = np.random.beta(self.alpha, self.alpha)
            rand_index = torch.randperm(inputs.shape[0]).cuda()
            target_a = labels
            target_b = labels[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            # compute output
            outputs = model(inputs)
            loss = mixup_criterion_hard(criterion, outputs, target_a, target_b, lam)
        return loss, outputs

class FinetuneL2SP(TrainIter):
    def __init__(self, args):
        self.model_source_weights = {}
        for name, param in model_source.named_parameters():
            self.model_source_weights[name] = param.detach()
        self.reg_weight = args.alpha_l2sp

    def reg_l2sp(self, model):
        loss = torch.tensor(0.).to(device)
        for name, param in model.named_parameters():
            if not name.startswith('fc.'):
                loss += 0.5 * torch.norm(param - self.model_source_weights[name]) ** 2
        return loss
    
    def loss(self, model, inputs, labels, iter_id):
        outputs = model(inputs)
        loss_ce = criterion(outputs, labels)
        loss_reg = self.reg_l2sp(model)
        loss = loss_ce + self.reg_weight * loss_reg
        return loss, outputs

class FinetuneDELTA(TrainIter):
    def __init__(self, args):
        self.attention_weight = self.load_attention_weight(args)
        self.reg_weight = args.alpha_delta

    def load_attention_weight(self, args):
        attention_weight = []
        cws = np.load('attention_weight/%s.npy' % args.task_name, allow_pickle = True)
        cw = torch.from_numpy(np.array(cws[3])).float().to(device)
        cw = F.softmax(cw / 5).detach()
        attention_weight.append(cw)
        return attention_weight

    def reg_delta(self, inputs):
        _ = model_source(inputs)
        fea_loss = torch.tensor(0.).to(device)
        for i, (fm_src, fm_tgt) in enumerate(zip(layer_outputs_source, layer_outputs_target)):
            b, c, h, w = fm_src.shape
            fm_src = fm_src.detach().reshape(b, c, h*w)
            fm_tgt = fm_tgt.reshape(b, c, h*w)
            distance = torch.norm(fm_tgt - fm_src, 2, 2)
            distance = torch.mul(self.attention_weight[i], distance ** 2) / (h * w)
            fea_loss += torch.sum(distance)
        return fea_loss

    def loss(self, model, inputs, labels, iter_id):
        outputs = model(inputs)
        loss_ce = criterion(outputs, labels)
        loss_reg = self.reg_delta(inputs)
        loss = loss_ce + self.reg_weight * loss_reg
        return loss, outputs

class FinetuneBSS(TrainIter):
    def __init__(self, args):
        self.reg_weight = args.alpha_bss

    def reg_bss(self):
        x = layer_outputs_target[0]
        x = F.adaptive_avg_pool2d(x, (1, 1))
        u, s, v = torch.svd(x.squeeze(2).squeeze(2))
        loss = torch.sum(s[-1] * s[-1])
        return loss

    def loss(self, model, inputs, labels, iter_id):
        outputs = model(inputs)
        loss_ce = criterion(outputs, labels)
        loss_reg = self.reg_bss()
        loss = loss_ce + self.reg_weight * loss_reg
        return loss, outputs

class FinetuneRIFLE(TrainIter):
    def __init__(self, args):
        self.reinit_num = args.alpha_rifle
        model_target.fc.reset_parameters()

    def loss(self, model, inputs, labels, iter_id):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if (iter_id+1) % (args.max_iters // self.reinit_num) == 0 and (iter_id+1) < args.max_iters:
            print('re-init fc layer at iter %d' % (iter_id + 1))
            model_target.fc.reset_parameters()
        return loss, outputs

class AuxFC(nn.Module):
    def __init__(self, input_dim, out_dim1, out_dim2):
        super(AuxFC, self).__init__()
        self.fc1 = nn.Linear(input_dim, out_dim1)
        self.fc2 = nn.Linear(input_dim, out_dim2)
    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        return x1, x2

class FinetuneSMILE(TrainIter):
    def __init__(self, args):
        self.alpha = 0.2
        self.ema_decay = 0.999
        self.reg_weight_fea = 0.01
        self.reg_weight_fc = 0.1
        source_cls_num = 365 if args.source_task == 'places365' else 1000
        model_target.fc = AuxFC(feature_dim, num_classes, source_cls_num).to(device)
        model_target.fc.fc2.load_state_dict(model_source.fc.state_dict())

    def feature_mixup(self, lam, index):
        fm_src = layer_outputs_source[0].detach()
        fm_tgt = layer_outputs_target[0]
        b, c, h, w = fm_src.shape
        fm_src = lam * fm_src + (1 - lam) * fm_src[index,:,:,:]
        fea_loss = torch.norm(fm_src.detach() - fm_tgt) / (h * w)
        return fea_loss

    def source_label_mixup(self, lam, index, outputs_aux, outputs_src):
        fea_loss = mixup_criterion_soft(outputs_aux, outputs_src, outputs_src[index,:], lam)
        return fea_loss

    def mean_teacher_update(self):
        for name, src_param in model_source.named_parameters():
            if name.startswith('fc.'):
                continue
            tgt_param = model_target.state_dict()[name]
            src_param.data.mul_(self.ema_decay).add_(1 - self.ema_decay, tgt_param.data)

    def loss(self, model, inputs, labels, iter_id):
        outputs_src = F.softmax(model_source(inputs), dim = 1)
        
        index_perm = torch.randperm(inputs.shape[0]).cuda()
        inputs_mix, targets_a, targets_b, lam = mixup_data(inputs, labels, index_perm, self.alpha)
        outputs_tgt, outputs_aux = model(inputs_mix)

        loss_tgt_label_mixup = mixup_criterion_hard(criterion, outputs_tgt, targets_a, targets_b, lam)
        loss_tgt_fea_mixup = self.feature_mixup(lam, index_perm)
        loss_src_label_mixup = self.source_label_mixup(lam, index_perm, outputs_aux, outputs_src)
        loss = loss_tgt_label_mixup + self.reg_weight_fea * loss_tgt_fea_mixup + self.reg_weight_fc * loss_src_label_mixup

        if iter_id % 10 == 0:
            self.mean_teacher_update()

        return loss, outputs_tgt


def train(model, criterion, optimizer, iter_id, iter_tgt):
    since = time.time()
    model.train()

    (inputs, labels), (imgs, _) = iter_tgt.next()
    inputs = inputs.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    loss, outputs = algo.loss(model, inputs, labels, iter_id)
 
    _, preds = torch.max(outputs, 1)

    loss.backward()
    optimizer.step()
    layer_outputs_source.clear()
    layer_outputs_target.clear()
    if iter_id % 10 == 0:
        time_elapsed = time.time() - since
        corr_sum = torch.sum(preds == labels.data)
        step_acc = corr_sum.double() / len(labels)
        print('step: %d/%d, reg: %s, loss = %.4f, time = %.2f, top1 = %.4f' %(iter_id, args.max_iters, args.regularizer, loss, time_elapsed, step_acc))

def test(model, criterion, optimizer, iter_id, iter_max):
    since = time.time()
    model.eval()  # Set model to training mode
    phase = 'test'
    running_corrects = 0
    nstep = len(dataloaders[phase])
    for i, ((inputs, labels), (imgs, _)) in enumerate(dataloaders[phase]):
        inputs = inputs.to(device)
        labels = labels.to(device)
        if args.regularizer == 'smile':
            outputs = model(inputs)[0]
        else:
            outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        layer_outputs_source.clear()
        layer_outputs_target.clear()
    epoch_acc = running_corrects.double() / dataset_sizes[phase]
    if iter_id < iter_max:
        print('{} iter: {:d} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, iter_id, 0, epoch_acc))
    else:
        print('{} iter: last Loss: {:.4f} Acc: {:.4f}'.format(
                phase, 0, epoch_acc))
        print()
    return model

if args.regularizer in ['fe', 'l2']:
    algo = FinetuneBase(args)
elif args.regularizer == 'mixup':
    algo = FinetuneMixup(args)
elif args.regularizer == 'cutmix':
    algo = FinetuneCutMix(args)
elif args.regularizer == 'l2sp':
    algo = FinetuneL2SP(args)
elif args.regularizer == 'delta':
    algo = FinetuneDELTA(args)
elif args.regularizer == 'bss':
    algo = FinetuneBSS(args)
elif args.regularizer == 'rifle':
    algo = FinetuneRIFLE(args)
elif args.regularizer == 'smile':
    algo = FinetuneSMILE(args)
else:
    raise NotImplementedError('No valid algorithm specified')

if args.regularizer == 'fe':
    for name, param in model_target.named_parameters():
        if not name.startswith('fc.'):
            param.requires_grad = False
            print('parameter frozen:', name)
    optimizer_main = optim.SGD(model_target.fc.parameters(), lr=args.lr_init, momentum=0.9, weight_decay=args.wd)
else:
    optimizer_main = optim.SGD(model_target.parameters(), lr=args.lr_init, momentum=0.9, nesterov=(args.nesterov == 1), weight_decay=args.wd)
len_tgt = len(dataloaders['train']) - 1
iter_tgt = iter(dataloaders['train'])

for iter_id in range(args.max_iters):
    if iter_id % len_tgt == 0:
        iter_tgt = iter(dataloaders['train'])
    lr = set_lr_step(args.lr_init, iter_id, args.max_iters, optimizer_main)
    train(model_target, criterion, optimizer_main, iter_id, iter_tgt)

    if iter_id == 0 or ((iter_id+1) % 1000 == 0 and (iter_id+1) < args.max_iters):
        test(model_target, criterion, optimizer_main, iter_id+1, args.max_iters)

test(model_target, criterion, optimizer_main, args.max_iters, args.max_iters)
