import logging
import torch.optim as optim
import ../tent

logger = logging.getLogger(__name__)

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