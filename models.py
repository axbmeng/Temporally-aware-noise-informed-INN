import torch
import torch.nn as nn
import numpy as np
import sys
from torch.nn import functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint
from torch.autograd import Function



class ChannelAttention(nn.Module):
    def __init__(self, channel_in, reduction_ratio=3):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channel_in,channel_in//reduction_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel_in//reduction_ratio,channel_in)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.view(y.size(0),-1)
        y = self.fc2(self.relu(self.fc1(y)))
        y = self.sigmoid(y)
        y = y.view(y.size(0),y.size(1),1,1)
        
        return y

from simam_module import simam_module


class Ni2Dense(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=False):
        super(Ni2Dense, self).__init__()
        
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv1_1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        
        self.conv4 = nn.Conv2d(channel_in +  gc, gc, 3, 1, 1, bias=bias)
         
        self.conv5 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #self.ca = ChannelAttention(12)
        self.reset_params()
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        
        x1_1 = self.lrelu(self.conv1_1(x))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1_1), 1)))
        x5 = self.conv5(torch.cat((x, x3, x4), 1))

        #attn = self.ca(x5)
        #x5 = x5 + x5*attn
        return x5

from transformer import Attention,FeedForward
from Subnet_constructor import ResidualBlock_AT_skip
class Reformer(nn.Module):
    def __init__(self, dims_in, clamp=1.0, split=3):
        super().__init__()

        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])
        self.split_len1 = split
        self.split_len2 = channels - split

        self.clamp = clamp
        self.affine_eps = 0.0001
        
        self.rsca = ResidualBlock_AT_skip(12,12,gc=16)
        self.patch_embed = OverlapPatchEmbed(12)
        self.s2 = Attention(48,1,False)
        self.patch_unembed = nn.Conv2d(48,3,kernel_size=3,stride=1,padding=1,bias = False)
        
        self.trans = nn.Conv2d(3,48,kernel_size=3,stride=1,padding=1,bias = False)
        self.t2 = FeedForward(48,2.66,False)
        self.untrans = nn.Conv2d(48,12,kernel_size=3,stride=1,padding=1,bias = False)
     
    def e(self, s):
        return torch.exp(self.clamp * (torch.sigmoid(s) * 2 - 1)) + self.affine_eps

    def forward(self, x1, x2, rev=False):
        # noise = torch.randn(x[:, 0:1, :, :].size()).mul_(sigma / 255.0).cuda()
        if not rev:
            
            x2_c = self.patch_embed(self.rsca(x2))
            s2 = self.patch_unembed(self.s2(x2_c))
            y1 = x1 + s2
            
            t1 = self.untrans(self.t2(self.trans(y1)))
            #t1 = self.t2(y1)
            y2 = x2 + t1
            return y1,y2
        else:
            t1 = self.untrans(self.t2(self.trans(x1)))
            #t1 = self.t2(x1)
            y2 = x2 - t1
            
            x2_c = self.patch_embed(self.rsca(y2))
            s2 = self.patch_unembed(self.s2(x2_c))
            y1 = x1 - s2
            
            return y1,y2

from transformer import OverlapPatchEmbed
from noise_estimate import noise_estimate
import math
class RNVPCouplingBlock(nn.Module):
    def __init__(self, dims_in, subnet_constructor=None, clamp=1.0, split=3):
        super().__init__()

        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])
        self.split_len1 = split
        self.split_len2 = channels - split

        self.clamp = clamp
        self.affine_eps = 0.0001

        self.s1 = subnet_constructor(self.split_len1 + 1, self.split_len2)
        self.t1 = subnet_constructor(self.split_len1 + 1, self.split_len2)

        self.s2 = Reformer(dims_in)
        self.s22 = Reformer(dims_in)

    def e(self, s):
        return torch.exp(self.clamp * (torch.sigmoid(s) * 2 - 1)) + self.affine_eps

    def forward(self, x, noise=None, rev=False):
        if not rev:
            
            x1, x2 = x[:, 0: self.split_len1, :, :], x[:, self.split_len1:, :, :]
            x2_c = x2
            b, c, w, h = x2.shape
            
            y1,x2 = self.s2(x1,x2,rev)
            y1,x2 = self.s22(y1,x2,rev)

            if w == 128:
                sigma = noise_estimate(y1)* 255
                noise = torch.randn(x1[:, 0:1, :, :].size()).mul_(sigma / 255.0).cuda()
            else:
                sigma = noise_estimate(y1[:,:,math.ceil((w-128)/2):(math.ceil((w-128)/2)+128),math.ceil((h-128)/2):(math.ceil((h-128)/2)+128)])* 255
                
                noise = torch.randn(x1[:, 0:1, :, :].size()).mul_(sigma / 255.0).cuda()
            y1_c = torch.cat((y1, noise), 1)
            s1, t1 = self.s1(y1_c), self.t1(y1_c)
            y2 = self.e(s1) * x2 + t1
            return torch.cat((y1, y2), 1)
        else:
            x1, x2 = x[:, 0: self.split_len1, :, :], x[:, self.split_len1:, :, :]
            b, c, w, h = x1.shape
            if w == 128:
                sigma = noise_estimate(x1)* 255
                noise = torch.randn(x1[:, 0:1, :, :].size()).mul_(sigma / 255.0).cuda()
            else:
                sigma = noise_estimate(x1[:,:,math.ceil((w-128)/2):(math.ceil((w-128)/2)+128),math.ceil((h-128)/2):(math.ceil((h-128)/2)+128)])* 255
                
                noise = torch.randn(x1[:, 0:1, :, :].size()).mul_(sigma / 255.0).cuda()
            x1_c = torch.cat((x1, noise), 1)
            s1, t1 = self.s1(x1_c), self.t1(x1_c)
            y2 = (x2 - t1) / self.e(s1)
            y2_c = y2
            
            y1,y2 = self.s2(x1,y2_c,rev)
            y1,y2 = self.s22(y1,y2,rev)
      
            return torch.cat((y1, y2), 1)

class MV(nn.Module):
    def __init__(self, dim):
        super(MV, self).__init__()
        # device = torch.device('cuda')
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            GDN(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ConvTranspose2d(64, 64, 3, 1, 1, bias=False),
            GDN(64, inverse=True),
            nn.ConvTranspose2d(64, dim, 4, 2, 1, bias=False),
        )

    def forward(self, x):
        out = self.conv_block(x)

        return out 

class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """

    def __init__(self,
                 ch,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=0.1,
                 reparam_offset=2 ** -18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset ** 2
        self.beta_bound = ((self.beta_min + self.reparam_offset ** 2) ** 0.5)
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma)
        self.pedestal = self.pedestal

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta ** 2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma ** 2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs ** 2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs


def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output


class SPyNet(nn.Module):
    """SPyNet network structure.

    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017

    Args:
        pretrained (str): path for pre-trained SPyNet. Default: None.
    """

    def __init__(self, pretrained='https://download.openmmlab.com/mmediting/restorers/'
                                  'basicvsr/spynet_20210409-c6c1bd09.pth'):
        super().__init__()

        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(6)])

        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False)
        elif pretrained is not None:
            raise TypeError('[pretrained] should be str or None, '
                            f'but got {type(pretrained)}.')

        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        """Compute flow from ref to supp.

        Note that in this function, the images are already resized to a
        multiple of 32.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        n, _, h, w = ref.size()

        # normalize the input images
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        # generate downsampled frames
        for level in range(5):
            ref.append(
                F.avg_pool2d(
                    input=ref[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
            supp.append(
                F.avg_pool2d(
                    input=supp[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(
                    input=flow,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True,
                    recompute_scale_factor=False) * 2.0

            # add the residue to the upsampled flow
            flow = flow_up + self.basic_module[level](
                torch.cat([
                    ref[level],
                    flow_warp(
                        supp[level],
                        flow_up.permute(0, 2, 3, 1),
                        padding_mode='border'), flow_up
                ], 1))

        return flow

    def forward(self, ref, supp):
        """Forward function of SPyNet.

        This function computes the optical flow from ref to supp.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """

        # upsize to a multiple of 32
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(
            input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(
            input=supp,
            size=(h_up, w_up),
            mode='bilinear',
            align_corners=False)

        # compute flow, and resize back to the original resolution
        flow = F.interpolate(
            input=self.compute_flow(ref, supp),
            size=(h, w),
            mode='bilinear',
            align_corners=False)

        # adjust the flow values
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow


class SPyNetBasicModule(nn.Module):
    """Basic Module for SPyNet.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """

    def __init__(self):
        super().__init__()

        self.basic_module = nn.Sequential(
            ConvModule(
                in_channels=8,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=16,
                out_channels=2,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=None))

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].

        Returns:
            Tensor: Refined flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)


def warp_torch(x, flo):
    """
    Backward warp an image tensor (im2) to im1, according to the optical flow from im1 to im2

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # Mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid_ori = grid + flo
    vgrid = torch.zeros_like(vgrid_ori)

    # Scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid_ori[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid_ori[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(x, vgrid)
    mask = torch.ones(x.size())
    if x.is_cuda:
        mask = mask.cuda()
    mask = torch.nn.functional.grid_sample(mask, vgrid)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output, mask


spynet = SPyNet().cuda()

import os

import cv2
import torch


NUM_IN_FR_EXT = 5 # temporal size of patch


class INN(nn.Module):
    def __init__(self, dims_in=[[15, 128, 128]]):
        super(INN, self).__init__()

        self.current_dims = dims_in

        operations = []
        
        for j in range(3):
      
            b = RNVPCouplingBlock(self.current_dims, subnet_constructor=Ni2Dense,
                                   clamp=1.0, split=3)
            operations.append(b)
        self.operations = nn.ModuleList(operations)
        
        self.mv = MV(2)
        #self.spynet = SPyNet()

    def forward(self, x, noise_map):

        frs = []

        for i in range(5):
            frs.append(x[:, i * 3:i * 3 + 3, :, :])
        
        aligned = []
        
        for i in range(5):
            flow = spynet(frs[2], frs[i])
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
            flow = self.mv(flow)   
            warped_img, _ = warp_torch(frs[i], flow)
            aligned.append(warped_img)
            
        out = frs[2]
        for i in range(5):
            if i != 2:
                out = torch.cat((out, aligned[i]), 1)
        i=0
        for op in self.operations:
            out = op.forward(out, noise_map, False)

        con = out[:, 0:3, :, :]
        z = out[:, 3:, :, :]

        z_ = torch.randn_like(z)

        out = torch.cat((con, z_), 1)
        for op in reversed(self.operations):
            out = op.forward(out, noise_map, True)

        frs = []
        for i in range(5):
            frs.append(out[:, i * 3:i * 3 + 3, :, :])
        output = (frs[0] + frs[1] + frs[2] + frs[3] + frs[4]) / 5

        
        return output

