import logging
import os
import time
import csv
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import tent

from models.Res import FReLU, ResNetCifar as Resnet
from utils.cli_utils import AverageMeter, accuracy
from dataset.selectedRotateImageFolder import prepare_test_data
from utils.io_utils import save_results
from utils.checkpoint_utils import get_checkpoint_path

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

def validate(val_loader, model, cfg, mode='eval'):
    top = AverageMeter('Acc@', ':6.2f')
    logger.info(f"Evaluating corruption: {cfg.corruption}")
    with torch.no_grad():
        start = time.time()
        for dl in val_loader:
            images, target = dl[0], dl[1]
            if cfg.gpu is not None and torch.cuda.is_available():
                images = images.cuda()
                target = target.cuda()
            output = model(images)
            acc = accuracy(output, target)
            top.update(acc[0], images.size(0))
        comp_time = time.time() - start
    return top.avg, comp_time



def setup_source(model):
    """Use the base model without adaptation."""
    model = tent.configure_model(model)
    logger.info("Using source model without adaptation.")
    return model

def setup_tent(model, cfg):
    """Set up test-time adaptation using Tent."""
    if cfg.ext_flag:
        logger.info("Configuring model with extractor parameters (TENT).")
        model = tent.configure_ext_model(model)
        params, param_names = tent.collect_ext_params(model)
    else:
        model = tent.configure_model(model)
        params, param_names = tent.collect_params(model, cfg)
    optimizer = setup_optimizer(params, cfg)
    logger.info(f"Adaptation parameters: {param_names}")
    return tent.Tent(model, optimizer, cfg)

def setup_optimizer(params, cfg):
    """Optimizer configuration for Tent (using SGD here)."""
    return optim.SGD(params, lr=cfg.learning_rate, momentum=cfg.momentum)


@hydra.main(config_path="conf", config_name="config_test")
def evaluate(cfg: DictConfig):
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Fix the random seed
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")

    # Build the model and load the checkpoint
    subnet = Resnet(26, 1, 10, 3, nn.BatchNorm2d, cfg.act_type)
    
    if cfg.gpu is not None and torch.cuda.is_available():
        subnet = subnet.cuda()

    ckpt_path = get_checkpoint_path(cfg)
    state = torch.load(ckpt_path, map_location='cpu')
    
    subnet.load_state_dict(state['net'])

    project_root = os.path.dirname(os.path.abspath(__file__))
    cfg.data.input = os.path.join(project_root, cfg.data.input)

    # Configure the model according to the adaptation method
    if cfg.algorithm == 'tent':
        logger.info("Test-time adaptation: TENT")
        model = setup_tent(subnet, cfg)
    else:
        logger.info("No adaptation (source model).")
        model = setup_source(subnet)

    # Use the corruption list from the config file or the default list
    common_corruptions = cfg.get("corruptions", [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ])

    acc_dict = {}
    comp_time_dict = {}
    readapt_acc_dict = {}
    readapt_comp_time_dict = {}

    # Evaluate each corruption
    for corruption in common_corruptions:
        if cfg.algorithm == 'tent':
            try:
                model.reset()
                logger.info("Resetting model for new corruption.")
            except Exception as e:
                logger.warning("Model reset not supported: " + str(e))
        cfg.corruption = corruption

        _, val_loader = prepare_test_data(cfg)
        top, comp_time = validate(val_loader, model, cfg, mode='eval')
        logger.info(f"Corruption: {corruption}, Accuracy: {top}")
        acc_dict[corruption] = top.item()
        comp_time_dict[corruption] = comp_time

        if cfg.readapt:
            cfg.corruption = "original"

            _, val_loader = prepare_test_data(cfg)
            readapt_top, readapt_comp_time = validate(val_loader, model, cfg, mode='eval')
            logger.info(f"Readapted Accuracy: {readapt_top}, Computation Time: {readapt_comp_time}")
            readapt_acc_dict[corruption] = readapt_top.item()
            readapt_comp_time_dict[corruption] = readapt_comp_time

            cfg.corruption = corruption

    # Output the results to CSV files
    output_file, readapt_output_file = save_results(
        cfg, common_corruptions, acc_dict, comp_time_dict,
        readapt_acc_dict if cfg.readapt else None,
        readapt_comp_time_dict if cfg.readapt else None
    )
    logger.info(f"Output written to {output_file}")
    if readapt_output_file:
        logger.info(f"Readapted output written to {readapt_output_file}")


if __name__ == '__main__':
    evaluate()
