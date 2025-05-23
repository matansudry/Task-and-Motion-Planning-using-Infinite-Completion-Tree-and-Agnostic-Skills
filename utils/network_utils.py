import torch
from torch import nn

def load_checkpoint_lightning(model:callable, checkpoint_path:str, device:str) -> nn.Module:
    """
    Load a checkpoint into a model

    Args:
        model (callable): _description_
        checkpoint_path (str): _description_
        device (str): _description_

    Returns:
        nn.Module: loaded network
    """
    folders = checkpoint_path.split("/")
    config_path = checkpoint_path.replace(folders[-1], "config.yml")
    network = model.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config_path).to(device)
    return network
