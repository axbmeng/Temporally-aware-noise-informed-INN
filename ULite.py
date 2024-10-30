import torch
import torch.nn as nn
from torch.autograd import Function
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

class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation = 1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups = dim, dilation = dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups = dim, dilation = dilation)

    def forward(self, x):
        x = x + self.dw_h(x) + self.dw_w(x)
        return x

class EncoderBlock(nn.Module):
    """Encoding then downsampling"""
    def __init__(self, in_c, out_c, mixer_kernel = (7, 7)):
        super().__init__()
        self.dw = AxialDW(in_c, mixer_kernel = (7, 7))
        self.bn = GDN(in_c)#nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=2,stride=2,padding=0)
        #self.down = nn.MaxPool2d((2,2))
        self.act = nn.GELU()

    def forward(self, x):
        skip = self.bn(self.dw(x))
        x = self.act(self.pw(skip))
        return x, skip

class DecoderBlock(nn.Module):
    """Upsampling then decoding"""
    def __init__(self, in_c, out_c, mixer_kernel = (7, 7)):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pw = nn.Conv2d(in_c + out_c, out_c,kernel_size=1)
        self.bn = GDN(out_c,inverse=True)#nn.BatchNorm2d(out_c)
        self.dw = AxialDW(out_c, mixer_kernel = (7, 7))
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.pw2(self.dw(self.bn(self.pw(x)))))
        return x
    
class BottleNeckBlock(nn.Module):
    """Axial dilated DW convolution"""
    def __init__(self, dim):
        super().__init__()

        gc = dim//4
        self.pw1 = nn.Conv2d(dim, gc, kernel_size=1)
        self.dw1 = AxialDW(gc, mixer_kernel = (3, 3), dilation = 1)
        self.dw2 = AxialDW(gc, mixer_kernel = (3, 3), dilation = 2)
        self.dw3 = AxialDW(gc, mixer_kernel = (3, 3), dilation = 3)

        self.bn = GDN(4*gc)#nn.BatchNorm2d(4*gc)
        self.pw2 = nn.Conv2d(4*gc, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.pw1(x)
        x = torch.cat([x, self.dw1(x), self.dw2(x), self.dw3(x)], 1)
        x = self.act(self.pw2(self.bn(x)))
        return x

class ULite(nn.Module):
    def __init__(self):
        super().__init__()

        """Encoder"""
        self.conv_in = nn.Conv2d(2, 16, kernel_size=7, padding='same')
        self.e1 = EncoderBlock(16, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)
        #self.e4 = EncoderBlock(128, 256)
        #self.e5 = EncoderBlock(256, 512)

        """Bottle Neck"""
        self.b5 = BottleNeckBlock(128)

        """Decoder"""
        #self.d5 = DecoderBlock(512, 256)
        #self.d4 = DecoderBlock(256, 128)
        self.d3 = DecoderBlock(128, 64)
        self.d2 = DecoderBlock(64, 32)
        self.d1 = DecoderBlock(32, 16)
        self.conv_out = nn.Conv2d(16, 2, kernel_size=1)

    def forward(self, x):
        """Encoder"""
        x = self.conv_in(x)
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        #x, skip4 = self.e4(x)
        #x, skip5 = self.e5(x)

        """BottleNeck"""
        x = self.b5(x)         # (512, 8, 8)

        """Decoder"""
        #x = self.d5(x, skip5)
        x# = self.d4(x, skip4)
        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)
        x = self.conv_out(x)
        return x
