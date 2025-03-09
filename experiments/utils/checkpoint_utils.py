import os

def get_checkpoint_path(cfg):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = os.path.join(project_root, "checkpoint", cfg.dataset)
    if cfg.act_type == "frelu":
        ckpt_file = "frelu_ckpt.pth"
    elif cfg.act_type == "prelu":
        ckpt_file = "prelu_ckpt.pth"
    else:
        ckpt_file = "relu_ckpt.pth"
    ckpt_fullpath = os.path.join(checkpoint_path, ckpt_file)
    return ckpt_fullpath
    