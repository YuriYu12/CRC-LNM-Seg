import torch
from typing import Callable
from torch import nn, Tensor
import torch.nn.functional as f
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
import numpy as np


class SoftCLDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_cldice: bool = False, smooth: float = 1., ddp: bool = True, iterations=10, do_bg=False):
        super(SoftCLDiceLoss, self).__init__()
        self.do_bg = do_bg
        self.is_ddp = ddp
        self.smooth = smooth
        self.iterations = iterations
        self.apply_nonlin = apply_nonlin
        self.batch_cldice = batch_cldice
        
    def forward(self, x, y, loss_mask=None):

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        
        x_all_labels = (torch.argmax(torch.softmax(x, dim=1), dim=1, keepdim=True) > 0) * 1.
        y_all_labels = (y > 0) * 1.
        
        pred_skeletion = soft_skeleton3d(x_all_labels, self.iterations)
        gt_skeletion = soft_skeleton3d(y_all_labels, self.iterations)
        if loss_mask is not None:
            pred_skeletion = pred_skeletion * loss_mask
            gt_skeletion = gt_skeletion * loss_mask
        
        t_prec = (torch.sum(torch.multiply(pred_skeletion, y_all_labels)) + self.smooth) /\
            (torch.sum(pred_skeletion) + self.smooth)
        t_sens = (torch.sum(torch.multiply(gt_skeletion, x_all_labels)) + self.smooth) /\
            (torch.sum(gt_skeletion) + self.smooth)
        
        if self.is_ddp and self.batch_cldice:
            t_prec = AllGatherGrad.apply(t_prec)
            t_sens = AllGatherGrad.apply(t_sens)
            
        if self.batch_cldice:
            t_prec = t_prec.sum(0)
            t_sens = t_sens.sum(0)
            
        cldice = 2 * (t_prec * t_sens + self.smooth) / (t_prec + t_sens + self.smooth)
        cldice = cldice.mean()
        return -cldice
        
        
def soft_erosion3d(inputs):
    p1 = -f.max_pool3d(-inputs, (3,1,1), (1,1,1), (1,0,0))
    p2 = -f.max_pool3d(-inputs, (1,3,1), (1,1,1), (0,1,0))
    p3 = -f.max_pool3d(-inputs, (1,1,3), (1,1,1), (0,0,1))
    return torch.min(p1, torch.min(p2, p3))


def soft_dilation3d(inputs):
    p = f.max_pool3d(inputs, (3,3,3), (1,1,1), (1,1,1))
    return p


def soft_opening3d(inputs):
    p = soft_dilation3d(soft_erosion3d(inputs))
    return p
    

def soft_skeleton3d(inputs: torch.Tensor, itr_):
    assert inputs.ndim == 5, f"expected 3D tensors (b, n, h, w, d), got tensor of ndim {inputs.ndim}"
    s = f.relu(inputs - soft_opening3d(inputs))
    for _ in range(itr_):
        inputs = soft_erosion3d(inputs)
        inputs_ = soft_opening3d(inputs)
        s_ = f.relu(inputs - inputs_)
        s = s + f.relu(s_ - s * s_)
    return s