import torch
import torch.nn as nn

def clamp(x, min_ratio=0, max_ratio=0):
    if len(x.size())==1:
        d = x.size(0)
        sorted_x, _ = torch.sort(x)
        min=sorted_x[int(d * min_ratio)]
        max=sorted_x[int(d * (1-max_ratio)-1)]
    else:
        d = x.size(1)
        sorted_x, _ = torch.sort(x, dim=1)
        min=sorted_x[:, int(d * min_ratio)].unsqueeze(1)
        max=sorted_x[:, int(d * (1-max_ratio)-1)].unsqueeze(1)
    clamped_x= torch.clamp(x, min, max)
    return clamped_x

def LaCO(weight_list, target_weight, laco_ratio) -> torch.Tensor:

    target_tensor = target_weight
    weight_tensor = torch.stack(weight_list, dim=0)
    diff_tensor = weight_tensor.data - target_tensor.data
    sum_diff_tensor = torch.sum(diff_tensor, dim=0)

    target_tensor.add_(sum_diff_tensor.data * laco_ratio)

    return target_tensor

