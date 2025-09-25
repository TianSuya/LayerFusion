import torch
import torch.nn as nn

def LayerAvg(weight_list):

    length = len(weight_list)

    weight_tensor = torch.stack(weight_list, dim=0)
    weight_tensor = torch.sum(weight_tensor, dim=0)

    weight_tensor = weight_tensor / length

    return weight_tensor


