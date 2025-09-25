import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F

def normalize(x, dim=0):
    min_values, _ = torch.min(x, dim=dim, keepdim=True)
    max_values, _ = torch.max(x, dim=dim, keepdim=True)
    y = (x - min_values) / (max_values - min_values)
    return y

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

def act(x):
    y = torch.tanh(x)  # torch.relu(x)
    return y

def PCB_merge(flat_task_checks, pcb_ratio=0.1):
    all_checks = flat_task_checks.clone()
    n, d = all_checks.shape   
    all_checks_abs = clamp(torch.abs(all_checks), min_ratio=0.0001, max_ratio=0.0001)
    clamped_all_checks = torch.sign(all_checks)*all_checks_abs
    # print('All Check Abs', all_checks_abs.mean(), all_checks_abs.std())
    self_pcb = normalize(all_checks_abs, 1)**2
    # print('Self PCB', self_pcb.mean(), self_pcb.std())
    self_pcb_act = torch.exp(n*self_pcb)
    # print('Self PCB Act', self_pcb_act.mean(), self_pcb_act.std())
    cross_pcb = all_checks * torch.sum(all_checks, dim=0)
    cross_pcb_act = act(cross_pcb)
    task_pcb = self_pcb_act * cross_pcb_act
    # print('Task PCB', task_pcb.mean(), task_pcb.std())

    scale = normalize(clamp(task_pcb, 1-pcb_ratio, 0), dim=1)
    # print('Scale', scale.mean(), scale.std())
    tvs = clamped_all_checks
    # selected_tvs = tvs * ( scale != 0 ).float()
    merged_tv = torch.sum(tvs * scale, dim=0) / torch.clamp(torch.sum(scale, dim=0), min=1e-12)
    # merged_tv = torch.sum(selected_tvs, dim=0)
    # print('Merged TV', merged_tv.mean(), merged_tv.std())
    return merged_tv, clamped_all_checks, scale

def Balance(weight_list, target_weight, ratio, balance_ratio) -> torch.Tensor:

    target_tensor = target_weight
    diff_tensor = []
    for weight in weight_list:
        diff_tensor.append(weight.data - target_tensor.data)
    
    diff_tensor = pruning(diff_tensor, mask_ratio=balance_ratio)
    # print('Diff Tensor:', diff_tensor.shape)
    # diff_tensor = torch.sum(diff_tensor, dim=0)
    target_tensor.add_(diff_tensor.data * ratio)

    return target_tensor


