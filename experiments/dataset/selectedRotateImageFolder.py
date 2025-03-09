import os
import copy
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils.data

NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
te_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*NORM)
])
tr_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*NORM)
])
mnist_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

common_corruptions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]

def prepare_test_data(cfg, use_transforms=True):
    if cfg.dataset == 'cifar10':
        tesize = 10000
        if not hasattr(cfg, 'corruption') or cfg.corruption == 'original':
            print('Test on the original test set')
            teset = torchvision.datasets.CIFAR10(
                root=cfg.data.input,
                train=False,
                download=True,
                transform=te_transforms
            )
        elif cfg.corruption in common_corruptions:
            print('Test on %s level %d' % (cfg.corruption, cfg.level))
            teset_raw = np.load(os.path.join(cfg.data.input, 'CIFAR-10-C', f"{cfg.corruption}.npy"))
            teset_raw = teset_raw[(cfg.level - 1) * tesize: cfg.level * tesize]
            teset = torchvision.datasets.CIFAR10(
                root=cfg.data.input,
                train=False,
                download=True,
                transform=te_transforms
            )
            teset.data = teset_raw
        else:
            raise Exception('Corruption not found!')
    else:
        raise Exception('Dataset not found!')

    if not hasattr(cfg, 'workers'):
        cfg.workers = 1
    teloader = torch.utils.data.DataLoader(
        teset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.workers
    )
    return teset, teloader

def prepare_train_data(cfg):
    print('Preparing data...')
    if cfg.dataset == 'cifar10':
        trset = torchvision.datasets.CIFAR10(
            root=cfg.data.input,
            train=True,
            download=True,
            transform=tr_transforms
        )
    else:
        raise Exception('Dataset not found!')

    if not hasattr(cfg, 'workers'):
        cfg.workers = 1
    trloader = torch.utils.data.DataLoader(
        trset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers
    )
    return trset, trloader
