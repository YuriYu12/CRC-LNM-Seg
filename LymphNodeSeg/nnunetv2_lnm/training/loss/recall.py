import torch
from torch import nn
from typing import Callable
import numpy as np
import scipy.ndimage as ndimage


class NoduleRecallLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, smooth: float = 1.,):
        super(NoduleRecallLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x, shp_y = x.shape, y.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        x = x[:, 1:]

        count = 0  # nodule count
        recall = 0  # batch recall

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                gt = y.long()
                y_onehot = torch.zeros(
                    shp_x, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)

            y_onehot = y_onehot[:, 1:]

            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    y_img = y_onehot[i, j]
                    x_img = x[i, j]
                    mask, num = ndimage.label(y_img.cpu())
                    count += num
                    for k in range(1, num + 1):
                        nodule = np.where(mask == k)
                        y_nodule = y_img[nodule]
                        x_nodule = x_img[nodule]

                        tp = (x_nodule * y_nodule).sum() if loss_mask is None else (
                            x_nodule * y_nodule * loss_mask[i, j][nodule]).sum()

                        tp_fn = y_nodule.sum() if loss_mask is None else (
                            y_nodule * loss_mask[i, j][nodule]).sum()

                        recall += tp / tp_fn

        recall = (recall + self.smooth) / (count + self.smooth)
        return -recall


if __name__ == '__main__':
    from nnunetv2_lnm.utilities.helpers import softmax_helper_dim1
    pred = torch.rand((2, 2, 3, 3, 3))
    ref = torch.randint(0, 2, (2, 3, 3, 3))

    rc = NoduleRecallLoss(apply_nonlin=softmax_helper_dim1, smooth=0)
    recall = rc(pred, ref)
    print(recall)
