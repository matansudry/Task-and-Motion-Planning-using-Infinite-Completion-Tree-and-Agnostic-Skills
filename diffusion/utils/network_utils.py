import torch
from torch import nn

def load_checkpoint_lightning(model:nn.Module, checkpoint_path:str):
    """_summary_

    Args:
        model (nn.Module): _description_
        checkpoint_path (str): _description_

    Returns:
        _type_: _description_
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=lambda storage,
        loc: storage
        )['state_dict']
    new_checkpoint = {}
    for key in list(checkpoint.keys()):
        new_key = key.replace("model.", "",1)
        if new_key in list(model.state_dict()):
            new_checkpoint[new_key] = checkpoint[key]
    model.load_state_dict(new_checkpoint)
    model = model.cuda()

    return model