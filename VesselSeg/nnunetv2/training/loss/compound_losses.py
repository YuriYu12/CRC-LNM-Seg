import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.training.loss.cldice import SoftCLDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
from functools import partial
from monai.losses import DiceLoss


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            target_regions = torch.clone(target[:, :-1])
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
    
    
class DC_and_clDC_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, soft_cldice_kwargs, alpha=0.3, ignore_label=None, dice_class=SoftDiceLoss) -> None:
        """
        apply both Dice and clDice loss constraint to the segmentation result to enhance connectivity
        this implementation is added according to paper http://arxiv.org/abs/2003.07311
        Args:
            alpha (float, optional): Defaults to 0.3.
        """
        super(DC_and_clDC_loss, self).__init__()
            
        self.weight_dice = 1 - alpha
        self.weight_cldc = alpha
        self.ignore_label = ignore_label
        self.cldc = SoftCLDiceLoss(**soft_cldice_kwargs)
        self.dice = SoftDiceLoss(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)
    
    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CLDice_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
        else:
            target_dice = target
            mask = None

        dc_loss = self.dice(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        cl_loss = self.cldc(net_output, target_dice, loss_mask=mask) \
            if self.weight_cldc != 0 else 0

        result = self.weight_cldc * cl_loss + self.weight_cldc * dc_loss
        return result
    
    
class CustomLoss(nn.Module):
    def __init__(self, ignore_label=None, ce_kwargs={}, soft_dice_kwargs={}, weighted=False, use_delta=False, alpha=10):
        super(CustomLoss, self).__init__()
        self.ignore_label = ignore_label
        ce_kwargs['reduction'] = 'none'
        if ignore_label is not None:
            ce_kwargs['ignore_label'] = ignore_label
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        # self.dc = DiceLoss(include_background=soft_dice_kwargs.get('do_bg', True),
        #                    softmax=False,
        #                    to_onehot_y=True,
        #                    reduction='none')
        self.dc = SoftDiceLoss(**soft_dice_kwargs)
        
        self.weighted = weighted
        self.use_delta = use_delta
        self.alpha = alpha
        
    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        b, *_ = net_output.shape
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(CustomLoss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
        else:
            target_dice = target
            mask = None
            
        net_output_var, net_output_gt = net_output.repeat(2, 1, 1, 1, 1).split([b, b], dim=0)
        net_output_var = net_output_var.softmax(1)
        random_var = torch.randint(0, 100, (1,), dtype=torch.float32, device=net_output_var.device) / 100
        net_output_app = torch.zeros_like(net_output_var)
        net_output_app[:, 0:1] = -net_output_var[:, 0:1] * random_var
        net_output_app[:, 1:] = net_output_var[:, 0:1] * random_var / (net_output_var.shape[1] - 1)
        net_output_var = net_output_var + net_output_app
        
        # dc_gt = self.dc(net_output_gt.softmax(1), target)
        dc_gt = self.dc(net_output_gt.softmax(1), target, loss_mask=mask, reduction='none')
        ce_gt = self.ce(net_output_gt, target)
        if self.use_delta:
            dc_var = self.dc(net_output_var.softmax(1), target, loss_mask=mask)
            loss_var = torch.clamp_min(dc_gt.mean() - dc_var, 0).mean()
        else:
            loss_var = 0
        
        if self.weighted:
            target_count = [torch.bincount(target[b].flatten().long(), minlength=7) for b in range(target.shape[0])]
            target_prop = torch.cat([(-self.alpha * (target_count[b] / target[b].numel())).exp()[None] for b in range(target.shape[0])]).softmax(1)
            loss_gt = (dc_gt * target_prop).mean() + (ce_gt * torch.cat([target_prop[b][target[b].long()][None] for b in range(target.shape[0])])).mean()
        else:
            loss_gt = dc_gt.mean() + ce_gt.mean()
        
        loss = loss_var + loss_gt
        return loss
        
        
ce_and_dice = DC_and_CE_loss
ce_and_bce_dice = DC_and_BCE_loss
weighted_ce_and_dice = partial(CustomLoss, weighted=True)  # proposedloss1
delta_ce_and_dice = partial(CustomLoss, use_delta=True)  # proposedloss2
delta_weighted_ce_and_dice = partial(CustomLoss, weighted=True, use_delta=True)  # proposedlossall
# delta_weighted_ce_and_dice = partial(CustomLoss, weighted=False, use_delta=False)  # proposedlossall

