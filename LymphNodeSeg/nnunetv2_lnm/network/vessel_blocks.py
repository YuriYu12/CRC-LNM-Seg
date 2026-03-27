import numpy as np
import torch
from torch import nn
from typing import Union, List, Tuple
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder

from torch import einsum
from einops import rearrange


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class CrossAttention(nn.Module):
    def __init__(self, dim, guide_dim=None, heads=8, dim_head=64, dropout=0.0, prenorm=True):
        super().__init__()
        guide_dim = default(guide_dim, dim)
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.guide_norm = nn.LayerNorm(guide_dim) if prenorm else nn.Identity()

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(guide_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(guide_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, guide=None):
        x_in = x.clone()
        guide = default(guide, x)
        
        x = rearrange(x, "b c f h w -> b (f h w) c")
        guide = rearrange(guide, "b c f h w -> b (f h w) c")

        x = self.norm(x)
        guide = self.guide_norm(guide)

        q = self.to_q(x) 
        k = self.to_k(guide)
        v = self.to_v(guide)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=self.heads), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.heads)
        out = self.to_out(out)

        b, c, f, h, w = x_in.shape
        out = rearrange(out, "b (f h w) c -> b c f h w", f=f, h=h, w=w)

        return x_in + out


# adapted from UNetDecoder
class UNetDecoderAttn(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False,
                 attention_level: int = 3  # attention at the last three resolutions
                 ):

        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        stages = []
        crossattns = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            if s <= attention_level:
                crossattns.append(CrossAttention(encoder.output_channels[-s]))
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))
            stages.append(StackedConvBlocks(
                n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs,
                encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs, nonlin_first
            ))

            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.crossattns = nn.ModuleList(crossattns)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.attention_level = attention_level

    def forward(self, skips, vessels):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            if s < self.attention_level:
                # print(lres_input.shape, vessels[s].shape)
                lres_input = self.crossattns[s](lres_input, guide=vessels[s])

            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output