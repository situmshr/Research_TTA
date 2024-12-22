from logging import debug
import os
import shutil
import time
import argparse
import json
import random
import math
import tqdm

from utils.utils import get_logger
from utils.cli_utils import *
from dataset.selectedRotateImageFolder import prepare_test_data

import torch    
import torch.nn.functional as F
import numpy as np

import tent

from models.Res import ResNetCifar as Resnet


def validate(val_loader, model, criterion, args, mode='eval'):
    batch_time = AverageMeter('Time', ':6.3f')
    top = AverageMeter('Acc@', ':6.2f')

    # with torch.no_grad():
    end = time.time()
    print(f"Corruption: {args.corruption}")
    for i, dl in enumerate(tqdm.tqdm(val_loader)):
        images, target = dl[0], dl[1]
        if args.gpu is not None:
            images = images.cuda()
        if torch.cuda.is_available():
            target = target.cuda()
        output = model(images)
        # measure accuracy and record loss
        acc = accuracy(output, target)
        top.update(acc[0], images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

    return top.avg


def copy_params_except_prelu(model, original_model):
    original_model_state_dict = original_model.state_dict()
    model_state_dict = model.state_dict()

    for name, param in original_model_state_dict.items():
        if "prelu" in name:
            continue

        if name in model_state_dict:
            if model_state_dict[name].shape == param.shape:
                model_state_dict[name].copy_(param)
                print(f"Copied parameter: {name}")
            else:
                print(f"Shape mismatch for {name}: {param.shape} != {model_state_dict[name].shape}")
        else:
            print(f"Parameter {name} not found in target model.")

    model.load_state_dict(model_state_dict)

def print_act_params(model, args):
    if args.act_type == "prelu" and args.act_flag:
        for name, param in model.named_parameters():
            if "prelu" in name:
                print(f"Name: {name}, Param: {param}")

def get_args():

    parser = argparse.ArgumentParser(description='PyTorch Cifar10-C Testing')

    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--dataroot", type=str, default="/work/masahiro-s/Research/experiments/data")
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument("--algorithm", type=str, default="tent")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--act_type", type=str, default="relu")
    parser.add_argument("--learning_rate", type=float, default=5e-3)
    parser.add_argument("--if_shuffle", type=bool, default=True)
    parser.add_argument("--act_flag", type=bool, default=False)
    parser.add_argument("--bn_flag", type=bool, default=False)

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    # set random seeds
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    subnet = Resnet(26, 1, 10, 3, nn.BatchNorm2d, args.act_type)

    if args.act_type == "frelu":
        state = torch.load("./checkpoint/frelu_ckpt.pth")
        subnet.load_state_dict(state['net'])
    elif args.act_type == "prelu":
        state = torch.load("./checkpoint/ckpt.pth")
        original_subnet = Resnet(26, 1, 10, 3, nn.BatchNorm2d, "relu")
        original_subnet.load_state_dict(state['net'])
        copy_params_except_prelu(subnet, original_subnet)
    else:
        state = torch.load("./checkpoint/ckpt.pth")
        subnet.load_state_dict(state['net'])

    if args.gpu is not None:
        subnet = subnet.cuda()
    
    common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                          'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 
                          'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

    if args.algorithm == 'tent':
        subnet = tent.configure_model(subnet)
        # params, param_names = tent.collect_params(subnet, args)
        params = subnet.parameters()
        optimizer = torch.optim.SGD(params, args.learning_rate, momentum=0.9)
        adapt_model = tent.Tent(subnet, optimizer)
    else:
        adapt_model = tent.configure_model(subnet)

    acc_dict = {}

    for corrupt in common_corruptions:
        if args.algorithm == 'tent':
            adapt_model.reset()
        
        args.corruption = corrupt

        val_dataset, val_loader = prepare_test_data(args)

        top1 = validate(val_loader, adapt_model, None, args, mode='eval')
        print(f"Corruption: {corrupt}, Top1: {top1}")

        acc_dict[corrupt] = top1.item()

    print(acc_dict)

    
