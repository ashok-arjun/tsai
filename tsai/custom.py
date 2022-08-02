import torch

__all__ = ['total_params']

def total_params(model):
    """
    Counts the number of paramters
    """
    total_params = 0
    params_no_grad = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            param = parameter.numel() 
            params_no_grad += param
            continue
        param = parameter.numel()
        total_params+=param
    return total_params