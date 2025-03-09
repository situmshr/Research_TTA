"""
Copyright to Tent Authors ICLR 2021 Spotlight
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
from torch.autograd import Variable

from models.Res import FReLU


class Tent(nn.Module):
    """
    Tent adapts a model by entropy minimization during testing.
    Once tented, the model updates itself on every forward pass.
    """
    def __init__(self, model, optimizer, cfg, mask_params=None, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.mask_params = mask_params
        self.steps = steps
        if steps <= 0:
            raise ValueError("Tent requires at least 1 step to forward and update")
        self.episodic = episodic

        # Save initial states to allow reset after adaptation.
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()
        # Perform adaptation steps and return final output.
        outputs = None
        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer, self.cfg, self.mask_params)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("Cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)

    def reset_steps(self, new_steps):
        self.steps = new_steps


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Compute the entropy of the softmax distribution from logits."""
    temperature = 1  # fixed temperature
    x = x / temperature
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.jit.script
def energy(x: torch.Tensor) -> torch.Tensor:
    """Calculate energy from logits."""
    temperature = 1
    x = -temperature * torch.logsumexp(x / temperature, dim=1)
    if torch.rand(1) > 0.95:
        print(x.mean(0).item())
    return x


@torch.enable_grad()  # Ensure gradients are enabled even in a no_grad context
def forward_and_adapt(x, model, optimizer, cfg, mask_params=None):
    """
    Forward a batch of data and adapt the model.
    Computes the softmax entropy loss, backpropagates, and updates parameters.
    """
    outputs = model(x)
    loss = softmax_entropy(outputs).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


def collect_ext_params(model):
    """
    Collect parameters excluding those from layers with 'fc' in their name.
    Returns a tuple (parameters, parameter names).
    """
    params = []
    names = []
    for name, param in model.named_parameters():
        if "fc" not in name:
            params.append(param)
            names.append(name)
    return params, names


def collect_params(model, cfg):
    """
    Collect affine scale and shift parameters from batch normalization layers,
    and, if configured, from activation modules.
    Returns a tuple (parameters, parameter names).
    """
    params = []
    names = []
    for name, module in model.named_modules():
        # BatchNorm2d parameters
        if cfg.bn_flag and isinstance(module, nn.BatchNorm2d) and "frelu" not in name:
            for param_name, param in module.named_parameters():
                if param_name in ['weight', 'bias']:
                    params.append(param)
                    names.append(f"{name}.{param_name}")
        # Activation-specific parameters
        if cfg.act_type != "relu" and cfg.act_flag != False:
            if cfg.act_type == "frelu" and isinstance(module, FReLU):
                if "bn" in cfg.frelu_update_params:
                    for param_name, param in module.bn.named_parameters():
                        params.append(param)
                        names.append(f"{name}.bn.{param_name}")
                if "conv" in cfg.frelu_update_params:
                    for param_name, param in module.f_conv.named_parameters():
                        params.append(param)
                        names.append(f"{name}.conv.{param_name}")
            elif cfg.act_type == "prelu" and isinstance(module, nn.PReLU):
                for param_name, param in module.named_parameters():
                    params.append(param)
                    names.append(f"{name}.{param_name}")
    return params, names



def copy_model_and_optimizer(model, optimizer):
    """
    Create copies of the model and optimizer states for later resetting.
    """
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """
    Restore the model and optimizer states from saved copies.
    """
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_ext_model(model):
    """
    Configure the model for external adaptation.
    Sets the model to training mode, disables gradients globally,
    then enables gradients for all modules except for Linear layers.
    """
    model.train()
    model.requires_grad_(False)
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            module.requires_grad_(True)
    return model


def configure_model(model):
    """
    Configure the model for use with Tent.
    Puts the model in train mode and disables gradients globally,
    then selectively enables gradients for layers to be adapted.
    BatchNorm layers and specific activation modules are reconfigured.
    """
    model.train()
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(True)
            # Force use of batch statistics in both train and eval modes.
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
        if isinstance(module, FReLU):
            module.requires_grad_(True)
            module.bn.track_running_stats = False
            module.bn.running_mean = None
            module.bn.running_var = None
        if isinstance(module, nn.PReLU):
            module.requires_grad_(True)
    return model


def check_model(model):
    """
    Check the model's compatibility with Tent.
    Ensures the model is in training mode, that some—but not all—
    parameters require gradients, and that at least one BatchNorm2d layer exists.
    """
    if not model.training:
        raise AssertionError("Tent requires the model to be in train mode: call model.train()")
    param_grads = [p.requires_grad for p in model.parameters()]
    if not any(param_grads):
        raise AssertionError("Tent requires some parameters to update (check grad requirements)")
    if all(param_grads):
        raise AssertionError("Tent should not update all parameters (check grad requirements)")
    if not any(isinstance(m, nn.BatchNorm2d) for m in model.modules()):
        raise AssertionError("Tent requires normalization (BatchNorm2d) for optimization")


def apply_random_params(model, cfg, mask_params):
    """
    If cfg.random_params is enabled, apply the corresponding mask to each parameter's gradient.
    
    Parameters:
        model (torch.nn.Module): The model whose parameters will be updated.
        cfg: Configuration object that includes 'random_params' flag.
        mask_params (dict): A dictionary where keys are parameter names and values are mask tensors.
    """
    if cfg.random_params and mask_params is not None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in mask_params and param.grad is not None:
                    param.grad.mul_(mask_params[name])
