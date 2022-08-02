import torch
import random
import numpy as np
import os
import re

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

def set_seed(seed, non_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if not non_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)  

def numeric_key(string):
    splitted = string.split(' ')
    if splitted[0].isdigit():
        return int(splitted[0])
    return -1

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]