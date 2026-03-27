from typing import Union, Type, List, Tuple

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from nnunetv2_lnm.network.vessel_blocks import UNetDecoderAttn

# adapted from PlainConvUNet
class VesselDecAttnUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 vessel_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False,
                 attention_level: int = 3  # attention at the last three resolutions
                 ):
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"

        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        
        self.skip = PlainConvEncoder(vessel_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                     [1] * n_stages, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                     dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                     nonlin_first=nonlin_first, pool= 'max')
        
        self.decoder = UNetDecoderAttn(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                       nonlin_first=nonlin_first, attention_level=attention_level)
        
        self.attention_level = attention_level

    def forward(self, x):
        skips = self.encoder(x[:, :1, ...])
        vessels = self.skip(x[:, 1:, ...])
        # print(len(skips), len(vessels))

        outputs = self.decoder(skips, vessels[::-1][:self.attention_level])
        return outputs