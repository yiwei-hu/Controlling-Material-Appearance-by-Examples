import math
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from models.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
from models.utils import mycrop


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class CircularUpsample(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.factor=scale_factor

    def forward(self, input):

        _,_,h,w=input.shape

        t = torch.cat([input,input[:,:,:,0:1]],dim=-1)
        t = torch.cat([t,torch.cat([input[:,:,0:1,:],input[:,:,0:1,0:1]], dim=-1)],dim=-2)

        m = nn.Upsample(size=(self.factor*h+1,self.factor*w+1), mode='bilinear', align_corners=True)
        out = m(t)[:,:,:self.factor*h, :self.factor*w]

        return out


class CircularUpsample2(nn.Module):
    def __init__(self, kernel, factor=2, manual_pad=3):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)
        self.manual_pad = manual_pad

    def forward(self, input):
        _,_,h,w = input.shape

        input = F.pad(input, (self.manual_pad, self.manual_pad, self.manual_pad, self.manual_pad), mode ='circular')
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)
        out = out[:,:, self.manual_pad:self.factor*h+self.manual_pad, self.manual_pad:self.factor*w+self.manual_pad]

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)
        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)
        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)
        return out


class CircularBlur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1, manual_pad=3, up_crop=0):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)
        self.pad = pad
        self.manual_pad = manual_pad
        self.up_crop = up_crop

    def forward(self, input):
        input = F.pad(input, (self.manual_pad, self.manual_pad, self.manual_pad, self.manual_pad), mode ='circular')
        out = upfirdn2d(input, self.kernel, pad=self.pad)
        _,_,h,w = out.shape

        if self.up_crop!=0:
            out = out[:,:, 2*self.up_crop:h-2*self.up_crop, 2*self.up_crop:w-2*self.up_crop]
        else:
            out = out[:,:, self.manual_pad:h-self.manual_pad, self.manual_pad:w-self.manual_pad]
        return out


class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True, circular=False, downsample=False, circular2=False):
        super().__init__()

        self.weight = nn.Parameter( torch.randn(out_channel, in_channel, kernel_size, kernel_size) )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.circular = circular
        self.circular2 = circular2
        self.downsample = downsample
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding

        if kernel_size==4 and self.downsample:
            self.padding = int(padding/2)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        if self.circular or self.circular2:
            input = F.pad(input, (self.padding,self.padding,self.padding,self.padding), mode ='circular')
            out = F.conv2d(input, self.weight*self.scale, bias=self.bias, stride=self.stride)
        else:
            out = F.conv2d(input, self.weight*self.scale, bias=self.bias, stride=self.stride, padding=self.padding)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
                f'{self.weight.shape[2]}, stride={self.stride}, padding={self.padding})')


class EqualConvTranspose2d(nn.Module):
    def __init__( self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True, circular=False, circular2=False, manual_pad=3 ):
        super().__init__()

        if circular:
            self.weight = nn.Parameter( torch.randn(out_channel, in_channel, kernel_size, kernel_size) )
        else:
            self.weight = nn.Parameter( torch.randn(in_channel, out_channel, kernel_size, kernel_size) )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.circular=circular
        self.circular2=circular2
        self.stride = stride
        self.padding = padding
        self.up = CircularUpsample(scale_factor=2)
        self.manual_pad = manual_pad 
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        if self.circular:
            up_in = self.up(input)
            input = F.pad(up_in, (self.padding,self.padding,self.padding,self.padding), mode ='circular')
            out = F.conv2d( input, self.weight*self.scale, bias=self.bias )

        elif self.circular2:
            input = F.pad(input, (self.manual_pad, self.manual_pad, self.manual_pad, self.manual_pad), mode ='circular')
            out = F.conv_transpose2d( input, self.weight*self.scale, bias=self.bias, stride=self.stride, padding=self.padding )

        else:
            out = F.conv_transpose2d( input, self.weight*self.scale, bias=self.bias, stride=self.stride, padding=self.padding )
        return out

    def __repr__(self):
        return ( f"{self.__class__.__name__}({self.weight.shape[0]}, {self.weight.shape[1]},"
                 f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})" )


class EqualLinear(nn.Module):
    def __init__( self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None

        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):

        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear( input, self.weight * self.scale, bias=self.bias * self.lr_mul )

        return out

    def __repr__(self):
        return ( f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})' )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)
        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__( self, in_channel, out_channel, kernel_size, style_dim, demodulate=True, upsample=False, downsample=False, blur_kernel=[1,3,3,1], circular=False, circular2=False ):
        super().__init__()

        self.circular=circular
        self.circular2=circular2
        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        self.manual_pad = 3
        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            if self.circular2:
                self.blur = CircularBlur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor, manual_pad=0)
            else:
                self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            if self.circular2:
                self.blur = CircularBlur(blur_kernel, pad=(pad0, pad1))
            else:
                self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter( torch.randn(1, out_channel, in_channel, kernel_size, kernel_size) )
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        self.demodulate = demodulate

        if self.circular:
            self.up = CircularUpsample(scale_factor=2)

    def __repr__(self):
        "For print"
        return ( f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})' )

    def forward(self, input, style):
        batch, in_channel, in_height, in_width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view( batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size )

        if self.upsample:
            input = input.view(1, batch * in_channel, in_height, in_width)

            if self.circular:
                input = F.pad(self.up(input), (self.padding, self.padding, self.padding, self.padding), mode ='circular')
                out = F.conv2d(input, weight, groups=batch)
            elif self.circular2:
                input = F.pad(input, (self.manual_pad, self.manual_pad, self.manual_pad, self.manual_pad), mode ='circular')
                weight = weight.view( batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size )
                weight = weight.transpose(1, 2).reshape( batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size )
                out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            else:
                weight = weight.view( batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size )
                weight = weight.transpose(1, 2).reshape( batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size )
                out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)

            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            if not self.circular:
                out = self.blur(out)
                if self.circular2:
                    out = out[:,:, 2*self.manual_pad:2*in_height+2*self.manual_pad, 2*self.manual_pad:2*in_width+2*self.manual_pad]

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, in_height, in_width)
            if self.circular or self.circular2:
                input = F.pad(input, (self.padding, self.padding, self.padding, self.padding), mode ='circular')
                out = F.conv2d(input, weight, groups=batch)

            else:
                out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def __shift_noise(self, noise, shiftN):

        scale = 512/noise.shape[-1]

        shiftY = shiftN[0]
        shiftX = shiftN[1]

        scaled_shiftX = shiftX/scale
        scaled_shiftY = shiftY/scale

        scaled_shiftX_int = int(np.modf(scaled_shiftX)[1])
        scaled_shiftX_frac = np.modf(scaled_shiftX)[0]
        scaled_shiftY_int = int(np.modf(scaled_shiftY)[1])
        scaled_shiftY_frac = np.modf(scaled_shiftY)[0]

        noise_crop = mycrop(noise, noise.shape[-1], rand0=[scaled_shiftY_int, scaled_shiftX_int])
        size = noise_crop.shape[-1]

        input = F.pad(noise_crop, (1,1,1,1), mode='circular')

        scaled_shiftX_frac = scaled_shiftX_frac*2/(size+1)
        scaled_shiftY_frac = scaled_shiftY_frac*2/(size+1)

        dX = torch.linspace(-1+2/(size+1)+scaled_shiftX_frac, 1-2/(size+1)+scaled_shiftX_frac, size)
        dY = torch.linspace(-1+2/(size+1)+scaled_shiftY_frac, 1-2/(size+1)+scaled_shiftY_frac, size)

        meshx, meshy = torch.meshgrid((dY, dX))
        grid = torch.stack((meshy, meshx), 2)
        grid = grid.unsqueeze(0).cuda() # add batch dim

        output = torch.nn.functional.grid_sample(input, grid, align_corners=True)

        return output

    def forward(self, image, noise=None, shiftN=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        if shiftN is not None:
            noise = self.__shift_noise(noise, shiftN)

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return out


class StyledConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, upsample=False, blur_kernel=[1, 3, 3, 1], demodulate=True, circular=False, circular2=False ):
        super().__init__()

        self.conv = ModulatedConv2d(in_channel, out_channel, kernel_size, style_dim, upsample=upsample, blur_kernel=blur_kernel, demodulate=demodulate, circular=circular, circular2=circular2)
        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None, shiftN=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise, shiftN=shiftN)
        out = self.activate(out)
        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, out_channel=3, upsample=True, blur_kernel=[1, 3, 3, 1], circular=False, circular2=False):
        super().__init__()

        if upsample:
            if circular:
                self.upsample = CircularUpsample(scale_factor=2)
            elif circular2:
                self.upsample = CircularUpsample2(blur_kernel)
            else:
                self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, out_channel, 1, style_dim, demodulate=False, circular=circular, circular2=circular2)
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip
        return out


class ConvLayer(nn.Sequential):
    def __init__( self, in_channel, out_channel, kernel_size, upsample=False, downsample=False, blur_kernel=(1,3,3,1), bias=True, activate=True, padding="zero", circular=False, circular2=False):
        layers = []
        self.padding = 0
        stride = 1

        if downsample:
            if circular:
                self.padding = kernel_size // 2
                stride = 2
            elif circular2:
                factor = 2
                p = (len(blur_kernel) - factor) + (kernel_size - 1)
                pad0 = (p + 1) // 2
                pad1 = p // 2
                layers.append(CircularBlur(blur_kernel, pad=(pad0, pad1)))
                stride = 2                
            else:
                factor = 2
                p = (len(blur_kernel) - factor) + (kernel_size - 1)
                pad0 = (p + 1) // 2
                pad1 = p // 2
                layers.append(Blur(blur_kernel, pad=(pad0, pad1)))
                stride = 2

        if upsample:
            if circular:
                layers.append( EqualConvTranspose2d( in_channel, out_channel, kernel_size, padding=kernel_size // 2, stride=2, bias=bias and not activate, circular=circular) )
            elif circular2:
                manual_pad = 3
                layers.append( EqualConvTranspose2d( in_channel, out_channel, kernel_size, padding=0, stride=2, bias=bias and not activate, circular2=circular2, manual_pad=manual_pad) )
                factor = 2
                p = (len(blur_kernel) - factor) - (kernel_size - 1)
                pad0 = (p + 1) // 2 + factor - 1
                pad1 = p // 2 + 1
                layers.append(CircularBlur(blur_kernel, pad=(pad0, pad1), up_crop=manual_pad, manual_pad=0))                
            else:
                layers.append( EqualConvTranspose2d( in_channel, out_channel, kernel_size, padding=0, stride=2, bias=bias and not activate) )
                factor = 2
                p = (len(blur_kernel) - factor) - (kernel_size - 1)
                pad0 = (p + 1) // 2 + factor - 1
                pad1 = p // 2 + 1
                layers.append(Blur(blur_kernel, pad=(pad0, pad1)))
        else:
            if not downsample:
                if padding == "zero":
                    self.padding = (kernel_size - 1) // 2
                elif padding == "reflect":
                    padding = (kernel_size - 1) // 2
                    if padding > 0:
                        layers.append(nn.ReflectionPad2d(padding))
                    self.padding = 0
                elif padding != "valid":
                    raise ValueError('Padding should be "zero", "reflect", or "valid"')
            
            layers.append( EqualConv2d( in_channel, out_channel, kernel_size, padding=self.padding, stride=stride, bias=bias and not activate, circular=circular, downsample=downsample, circular2=circular2 ) )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))
            else:
                layers.append(ScaledLeakyReLU(0.2))
        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], circular=False, circular2=False, dk_size=3):
        super().__init__()
        self.conv1 = ConvLayer(in_channel, in_channel, 3, circular=circular, circular2=circular2)
        self.conv2 = ConvLayer(in_channel, out_channel, dk_size, downsample=True, circular=circular, circular2=circular2)
        self.skip = ConvLayer( in_channel, out_channel, 1, downsample=True, activate=False, bias=False, circular=circular, circular2=circular2)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)
        return out

