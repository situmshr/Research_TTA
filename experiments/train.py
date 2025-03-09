import os
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
import hydra

from models.Res import ResNetCifar  # Your custom model definition

# Import the data preparation functions
from experiments.dataset.selectedRotateImageFolder import prepare_train_data, prepare_test_data

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_net(cfg: DictConfig):
    """
    Load an original model checkpoint (without the desired activation type),
    then create a new model with cfg.act_type and copy matching weights.
    """
    ori_net = ResNetCifar(26, 1, 10, 3, nn.BatchNorm2d)
    checkpoint = torch.load(cfg.ori_model_path, map_location='cpu')
    ori_net.load_state_dict(checkpoint['net'])
    
    # Create new model with desired activation type
    net = ResNetCifar(26, 1, 10, 3, nn.BatchNorm2d, act_type=cfg.act_type)
    net_state = net.state_dict()
    for name, param in ori_net.state_dict().items():
        if name in net_state:
            net_state[name].copy_(param)
    net.load_state_dict(net_state)
    return net.to(device)

def get_frelu_params(net):
    """
    Return parameters whose names contain "frelu".
    """
    return [param for name, param in net.named_parameters() if "frelu" in name]

def test_model(net, testloader):
    """
    Evaluate the network on the test dataset.
    """
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

def warm_up(cfg: DictConfig, net):
    """
    Fine-tune only the FReLU parameters using the training data.
    """
    _, trainloader = prepare_train_data(cfg)
    _, testloader = prepare_test_data(cfg)
    
    criterion = nn.CrossEntropyLoss()
    frelu_params = get_frelu_params(net)
    optimizer = optim.SGD(frelu_params, lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)
    
    for epoch in range(cfg.warmup_epochs):
        net.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(inputs), labels)
            loss.backward()
            optimizer.step()
        acc = test_model(net, testloader)
        scheduler.step()
        print(f"Warm-up Epoch {epoch+1}/{cfg.warmup_epochs}, Test Acc: {acc:.2f}%")
    return net

def full_train(cfg: DictConfig, net):
    """
    Train all network parameters using the full training loop.
    """
    _, trainloader = prepare_train_data(cfg)
    _, testloader = prepare_test_data(cfg)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=0.1)
    
    for epoch in range(cfg.epochs):
        net.train()
        epoch_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(inputs), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(trainloader)
        print(f"Epoch {epoch+1}/{cfg.epochs}, Train Loss: {epoch_loss:.4f}")
        acc = test_model(net, testloader)
        scheduler.step()
    return net

@hydra.main(config_path="conf", config_name="config_train")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)
    
    if cfg.mode == "warmup":
        net = get_net(cfg)
        net = warm_up(cfg, net)
        ckpt_name = f"warm_up_{cfg.act_type}_ckpt.pth"
    elif cfg.mode == "train":
        # Instantiate a new model for full training (from scratch or random initialization)
        net = ResNetCifar(26, 1, 10, 3, nn.BatchNorm2d, act_type=cfg.act_type).to(device)
        net = full_train(cfg, net)
        ckpt_name = f"full_train_{cfg.act_type}_ckpt.pth"
    else:
        raise Exception(f"Unknown mode: {cfg.mode}")
    
    ckpt_dir = os.path.join(cfg.out_f, cfg.dataset)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    torch.save({'net': net.state_dict()}, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

if __name__ == '__main__':
    main()
