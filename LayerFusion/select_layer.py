import torch
import torch.nn as nn

def SelectLayer(weight_list, select_index=0):

    first_weight_tensor = weight_list[select_index]

    return first_weight_tensor


