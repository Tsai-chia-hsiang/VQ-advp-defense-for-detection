from typing import Literal
import torch
import torch.nn as nn


def generate_patch(ptype:Literal['gray', 'random']="gray", psize:int=300, device:torch.device = torch.device("cpu")) -> nn.Parameter:
    """
    Generate a random patch as a starting point for optimization.
    """
    adv_patch:torch.Tensor=None
    match ptype:
        case 'gray':
            adv_patch = torch.full((3, psize, psize), 0.5, device=device)
        case 'random':
            adv_patch = torch.rand((3, psize, psize), device=device)
    
    return nn.Parameter(adv_patch)