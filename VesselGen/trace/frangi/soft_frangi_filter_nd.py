import torch
from torch import nn
from .gaussian_smoothing import GaussianSmoothing


class SoftFrangiFilter2D(nn.Module):

    def __init__(self, channels, kernel_size, sigmas, beta, c, device):
        """
        Apply Soft Frangi filter on a 3d tensor.
        Arguments:
            channels (int, sequence): Number of channels of the input tensors. Output will
                have this number of channels as well.
            kernel_size (int, sequence): Size of the gaussian kernel.
            sigmas (list, sequence): List of standard deviations of the gaussian kernels.
            beta (float, sequence): Beta parameter of Frangi filter.
            beta (с, sequence): С parameter of Frangi filter.
                Default value is 2 (spatial).
        """
        super(SoftFrangiFilter2D, self).__init__()
        self.sigmas = sigmas
        self.gaus_filters = []
        self.beta = beta
        self.c = c
        for sigma in sigmas:
            self.gaus_filters.append([])
            for order in ['xx', 'yy', 'xy']:
                self.gaus_filters[-1].append(GaussianSmoothing(channels, kernel_size, sigma, 2, order, device))

    def _calc_frangi_response(self, xx, yy, xy):
        """
        Calculate Frangi filter response give the second order derivatives.
        Arguments:
            xx (torch.Tensor, sequence): (bs, channels, h, w), second order derivative on x-axis
            yy (torch.Tensor, sequence): (bs, channels, h, w), second order derivative on y-axis
            xy (torch.Tensor, sequence): (bs, channels, h, w), second order derivative on xy-axes
        """
        lambda_t1 = ((xx + yy) + torch.sqrt((xx - yy) ** 2 + 4 * xy ** 2 + 1e-6)) / 2
        lambda_t2 = ((xx + yy) - torch.sqrt((xx - yy) ** 2 + 4 * xy ** 2 + 1e-6)) / 2
        lambdas = torch.stack((lambda_t1, lambda_t2), dim=0)
        lambdas_abs, sorted_ind = torch.sort(torch.abs(lambdas), dim=0)
        lambda2_sign = (lambdas * sorted_ind).sum(dim=0)
        lambda1 = lambdas_abs[0]
        lambda2 = lambdas_abs[1]
        blobness = torch.zeros(lambda1.shape, device=lambda1.device)
        blobness[lambda2 != 0] = blobness[lambda2 != 0] + torch.exp(
            -(lambda1[lambda2 != 0] / lambda2[lambda2 != 0]) ** 2 / (2 * self.beta ** 2))
        hess_struc = (1 - torch.exp(-torch.sqrt(lambda1 ** 2 + lambda2 ** 2) ** 2 / (2 * self.c ** 2)))
        vness = blobness * hess_struc
        vness[lambda2_sign > 0] = vness[lambda2_sign > 0] * 0
        return vness, blobness, hess_struc, lambda1, lambda2

    def forward(self, img):
        """
        Apply Soft Frangi filter on a batch of images.
        Arguments:
            img (torch.Tensor, sequence): Tensor of shape (bs, channels, h, w)
        """
        frangi_resp = torch.zeros((len(self.sigmas),) + img.shape, dtype=torch.float32, device=img.device)
        for i in range(len(self.sigmas)):
            xx = self.gaus_filters[i][0](img)
            yy = self.gaus_filters[i][1](img)
            xy = self.gaus_filters[i][2](img)
            vness, _, _, _, _ = self._calc_frangi_response(xx, yy, xy)
            frangi_resp[i] = vness
        max_frangi_resp = (torch.softmax(frangi_resp, dim=0) * frangi_resp).sum(dim=0)
        return max_frangi_resp
      

class SoftFrangiFilter3D(nn.Module):

    def __init__(self, channels, kernel_size, sigmas, beta, c, device):
        """
        Apply Soft Frangi filter on a 3d tensor.
        Arguments:
            channels (int, sequence): Number of channels of the input tensors. Output will
                have this number of channels as well.
            kernel_size (int, sequence): Size of the gaussian kernel.
            sigmas (list, sequence): List of standard deviations of the gaussian kernels.
            beta (float, sequence): Beta parameter of Frangi filter.
            beta (с, sequence): С parameter of Frangi filter.
                Default value is 2 (spatial).
        """
        super(SoftFrangiFilter3D, self).__init__()
        self.sigmas = sigmas
        self.gaus_filters = []
        self.beta = beta
        self.c = c
        for sigma in sigmas:
            self.gaus_filters.append([])
            for order in ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']:
                self.gaus_filters[-1].append(GaussianSmoothing(channels, kernel_size, sigma, 3, order, device))

    def _calc_frangi_response(self, xx, yy, zz, xy, xz, yz):
        """
        Calculate Frangi filter response give the second order derivatives.
        """
        # 1) numerically solve for lambdas
        alpha = xx + yy + zz
        beta = xy ** 2 + xz ** 2 + yz ** 2 - xy * xz - xy * yz - xz * yz
        gamma = xx * yy * zz + 2 * xy * yz * xz - xx * yz ** 2 - yy * xz ** 2 - zz * xy ** 2
        p = -(beta + alpha / 3) ** 2
        q = -(gamma + 2 * alpha ** 3 / 27 + alpha * beta / 3)
        cos_phi = -q / (torch.sqrt((torch.abs(p) / 3) ** 3) * 2)
        lambda_t1 = alpha / 3 + 2 * torch.sqrt(torch.abs(p) / 3) * torch.cos(torch.arccos(cos_phi) / 3)
        lambda_t2 = alpha / 3 - 2 * torch.sqrt(torch.abs(p) / 3) * torch.cos(torch.arccos(cos_phi) / 3 + torch.pi)
        lambda_t3 = alpha / 3 - 2 * torch.sqrt(torch.abs(p) / 3) * torch.cos(torch.arccos(cos_phi) / 3 - torch.pi)
        # lambda_t1 = ((xx + yy) + torch.sqrt((xx - yy) ** 2 + 4 * xy ** 2 + 1e-6)) / 2
        # lambda_t2 = ((xx + yy) - torch.sqrt((xx - yy) ** 2 + 4 * xy ** 2 + 1e-6)) / 2
        lambdas = torch.stack((lambda_t1, lambda_t2, lambda_t3), dim=0)
        lambdas_abs, sorted_ind = torch.sort(torch.abs(lambdas), dim=0)
        lambda2_sign = (lambdas * sorted_ind).sum(dim=0)
        lambda1 = lambdas_abs[0]
        lambda2 = lambdas_abs[1]
        blobness = torch.zeros(lambda1.shape, device=lambda1.device)
        blobness[lambda2 != 0] = blobness[lambda2 != 0] + torch.exp(
            -(lambda1[lambda2 != 0] / lambda2[lambda2 != 0]) ** 2 / (2 * self.beta ** 2))
        hess_struc = (1 - torch.exp(-torch.sqrt(lambda1 ** 2 + lambda2 ** 2) ** 2 / (2 * self.c ** 2)))
        vness = blobness * hess_struc
        vness[lambda2_sign > 0] = 0
        return vness, blobness, hess_struc, lambda1, lambda2

    def forward(self, img):
        """
        Apply Soft Frangi filter on a batch of images.
        Arguments:
            img (torch.Tensor, sequence): Tensor of shape (bs, channels, h, w, d)
        """
        frangi_resp = torch.zeros((len(self.sigmas),) + img.shape, dtype=torch.float32, device=img.device)
        for i in range(len(self.sigmas)):
            xx = self.gaus_filters[i][0](img)
            yy = self.gaus_filters[i][1](img)
            zz = self.gaus_filters[i][2](img)
            xy = self.gaus_filters[i][3](img)
            xz = self.gaus_filters[i][4](img)
            yz = self.gaus_filters[i][5](img)
            vness, _, _, _, _ = self._calc_frangi_response(xx, yy, zz, xy, xz, yz)
            frangi_resp[i] = vness
        max_frangi_resp = (torch.softmax(frangi_resp, dim=0) * frangi_resp).sum(dim=0)
        return max_frangi_resp