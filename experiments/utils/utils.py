import os
import sys
import logging
import random

import numpy as np
import torch

import torch.nn as nn

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def access_Kdim_element(param, index):
    dims = param.shape
    index_list = []
    for i in range(len(dims)-1):
        i_k = index // np.prod(dims[i+1:])
        index_list.append(i_k)
        index = index - i_k * np.prod(dims[i+1:])
    index_list.append(index)

    return index_list

def get_module_by_name(model, module_name):
    """
    Function to retrieve a specified module by name.

    Args:
        model (torch.nn.Module): The target PyTorch model.
        module_name (str): The name of the module to retrieve (dot-separated names are supported).

    Returns:
        torch.nn.Module: The module specified by the name.
    """
    try:
        # Access nested modules using dot-separated names
        module = model
        for name in module_name.split('.'):
            module = getattr(module, name)
        return module
    except AttributeError:
        raise ValueError(f"Module '{module_name}' not found in the model.")

def create_mask_params(model, params_list):
    mask_params = {}

    for nm, m in model.named_modules():
        for np, p in m.named_parameters():
            if np in ['weight', 'bias']:
                mask_params[f"{nm}.{np}"] = torch.ones_like(p)
    
    for param_str in params_list:
        module_name, param_name, index = param_str.split('___')
        param = get_module_by_name(model, module_name).get_parameter(param_name)
        index_list = access_Kdim_element(param, int(index))
        mask_params[f"{module_name}.{param_name}"][tuple(index_list)] = 0

    return mask_params

def count_params(model, args):
    k = 0
    for nm, m in model.named_modules():
        if args.act_type == "frelu":
            if isinstance(m, FReLU):
                # Add the number of parameters in the depthwise convolution (f_conv)
                k += sum(p.numel() for p in m.f_conv.parameters())
                # Add the number of parameters in the BatchNorm (bn)
                k += sum(p.numel() for p in m.bn.parameters())
        elif args.act_type == "prelu":
            if isinstance(m, nn.PReLU):
                k += sum(p.numel() for p in m.parameters())
    
    return k

def select_random_params(model, args):
    # Gather all trainable parameters from the model
    all_params = []
    
    for nm, m in model.named_modules():
        if "fc" in nm:
            continue
        for np, p in m.named_parameters():
            if np in ['weight', 'bias']:
                for i in range(p.numel()):
                    all_params.append(f"{nm}___{np}___{i}")

    # Determine how many parameters to select (k) based on the activation parameters
    k = count_params(model, args)
    print(f"Number of parameters to be randomly selected (k): {k}")

    # Randomly sample k parameters
    selected_params = random.sample(all_params, len(all_params)-k)

    return selected_params

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

def mean(items):
    return sum(items)/len(items)


def max_with_index(values):
    best_v = values[0]
    best_i = 0
    for i, v in enumerate(values):
        if v > best_v:
            best_v = v
            best_i = i
    return best_v, best_i


def shuffle(*items):
    example, *_ = items
    batch_size, *_ = example.size()
    index = torch.randperm(batch_size, device=example.device)

    return [item[index] for item in items]


def to_device(*items):
    return [item.to(device=device) for item in items]


def set_reproducible(seed=0):
    '''
    To ensure the reproducibility, refer to https://pytorch.org/docs/stable/notes/randomness.html.
    Note that completely reproducible results are not guaranteed.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, output_directory: str, log_name: str, debug: str) -> logging.Logger:
    logger = logging.getLogger(name)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s: %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if output_directory is not None:
        file_handler = logging.FileHandler(os.path.join(output_directory, log_name))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.propagate = False
    return logger
    

def _sign(number):
    if isinstance(number, (list, tuple)):
        return [_sign(v) for v in number]
    if number >= 0.0:
        return 1
    elif number < 0.0:
        return -1


def compute_flops(module: nn.Module, size, args, device):
    # print(module._auxiliary)
    def size_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        *_, h, w = output.shape
        module.output_size = (h, w)
    hooks = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            # print("init hool for", name)
            hooks.append(m.register_forward_hook(size_hook))
    with torch.no_grad():
        training = module.training
        module.eval()
        module(torch.rand(size).to(device))
        module.train(mode=training)
        # print(f"training={training}")
    for hook in hooks:
        hook.remove()

    flops = 0
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            # print(name)
            h, w = m.output_size
            kh, kw = m.kernel_size
            flops += h * w * m.in_channels * m.out_channels * kh * kw / m.groups
        if isinstance(module, nn.Linear):
            flops += m.in_features * m.out_features

    # Count FLOPs fwd + bwd
    if args.backward:
        flops += 2 * flops
    else:
        return flops

    if args.ext_flag:
        return flops

    # Count FLOPs for tent(BN+FReLU)
    if args.act_flag or args.bn_flag:
        for name, m in module.named_modules():
            if isinstance(m, nn.Conv2d):
                if args.act_type in name:
                    continue
                h, w = m.output_size
                kh, kw = m.kernel_size
                flops -= h * w * m.in_channels * m.out_channels * kh * kw / m.groups
    # Count FLOPs for tent(BN)
    if not args.act_flag:
        for name, m in module.named_modules():
            if isinstance(m, nn.Conv2d):
                if args.act_type in name:
                    h, w = m.output_size
                    kh, kw = m.kernel_size
                    flops -= h * w * m.in_channels * m.out_channels * kh * kw / m.groups

    # Subtract FLOPs for the first conv layer(Δconv_output)
    head_conv = getattr(module, "conv1")
    h, w = head_conv.output_size
    kh, kw = head_conv.kernel_size
    flops -= h * w * head_conv.in_channels * head_conv.out_channels * kh * kw / head_conv.groups

    return flops

def compute_nparam(module: nn.Module, skip_pattern):
    n_param = 0
    for name, p in module.named_parameters():
        if skip_pattern not in name:
            n_param += p.numel()
    return n_param