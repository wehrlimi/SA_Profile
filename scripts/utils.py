import torch

def define_device(cfg):
    if cfg.device.type == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(cfg.device.gpu_id))
    return device