
#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
from .deform_conv_v2 import DeformConv2D


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()

        self.layernorm1 = nn.LayerNorm(c)
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)

        self.layernorm2 = nn.LayerNorm(c)
        self.fc1 = nn.Linear(c, 4 * c, bias=False)
        self.fc2 = nn.Linear(4 * c, c, bias=False)

        self.dropout = nn.Dropout(0.1)
        self.act = nn.ReLU(True)

    def forward(self, x):
        x1 = x
        x = self.layernorm1(x)
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x1

        x2 = x
        x = self.layernorm2(x)
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.fc2(x)) + x2

        return x


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1, )
        return self.conv(x)


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act, )
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act="silu", ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv

        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act="silu", ):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in range(n)]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        x = self.conv3(x)
        return x


class dConv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=3, d=3, g=4, act=True, ):  # ch_in, ch_out, kernel, stride, padding, groups
        super(dConv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, d, g, bias=False, )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class C3TR(CSPLayer):
    # C3 module with TransformerBlock()
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, e=0.5):
        super().__init__(in_channels, out_channels, n, shortcut, e)
        c_ = int(out_channels * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class TransformerBlock(nn.Module):
    # Vision Transformer
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
        return self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)


class CSPDarknet(nn.Module):
    def __init__(self, dep_mul, wid_mul, out_features=("dark3", "dark4", "dark5"), depthwise=False, act="silu"):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features

        # Load ResNet18 without the fully connected layer
        resnet = resnet18(weights=None)
        
        # Stem: ResNet18's initial conv + bn + relu + maxpool
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        # ResNet18 layers
        self.dark2 = resnet.layer1  # 64 channels
        self.dark3 = resnet.layer2  # 128 channels
        self.dark4 = resnet.layer3  # 256 channels
        self.dark5 = resnet.layer4  # 512 channels
        
        # Transformer and DeformConv2D layers (unchanged)
        self.swt1 = C3TR(512, 512)
        self.deconv = DeformConv2D(512, 512, kernel_size=3, padding=1, modulation=True)
        
        # Attention layers adjusted for ResNet18 channel sizes
        self.att1 = nn.Sequential(*[nn.Conv2d(64, 64, kernel_size=1), nn.Sigmoid()])
        self.att2 = nn.Sequential(*[nn.Conv2d(64, 64, kernel_size=1), nn.Sigmoid()])
        self.att3 = nn.Sequential(*[nn.Conv2d(128, 128, kernel_size=1), nn.Sigmoid()])
        self.att4 = nn.Sequential(*[nn.Conv2d(256, 256, kernel_size=1), nn.Sigmoid()])  # Fix: 256->256 not 512->512
        self.att5 = nn.Sequential(*[nn.Conv2d(512, 512, kernel_size=1), nn.Sigmoid()])
        
        # Convolution layers to adjust channels of inter features
        # Based on actual inter shapes from debug output:
        # inter[0]: 32 channels -> 64 channels (to match stem output)
        # inter[1]: 64 channels -> 64 channels (to match dark2 output)  
        # inter[2]: 128 channels -> 128 channels (to match dark3 output)
        # inter[3]: 256 channels -> 256 channels (to match dark4 output)
        # inter[4]: 512 channels -> 512 channels (to match dark5 output)
        self.inter0_conv = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.inter1_conv = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.inter2_conv = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False)  # 128->128
        self.inter3_conv = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)  # 256->256  
        self.inter4_conv = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)  # 512->512

    def forward(self, x, inter):
        outputs = {}
        
        x = self.stem(x)  # ~ 128x128x64
        outputs["stem"] = x
        
        # Process inter[0] and add to stem output
        inter0_adjusted = self.inter0_conv(F.interpolate(inter[0], size=x.shape[2:]))
        x = self.dark2(self.att1(x + inter0_adjusted) * x)  # 128x128x64
        outputs["dark2"] = x
        
        # Process inter[1] and add to dark2 output
        inter1_adjusted = self.inter1_conv(F.interpolate(inter[1], size=x.shape[2:]))
        x = self.dark3(self.att2(x + inter1_adjusted) * x)  # 64x64x128
        outputs["dark3"] = x
        
        # Process inter[2] and add to dark3 output
        inter2_adjusted = self.inter2_conv(F.interpolate(inter[2], size=x.shape[2:]))
        x = self.dark4(self.att3(x + inter2_adjusted) * x)  # 32x32x256
        outputs["dark4"] = x
        
        # Process inter[3] and add to dark4 output
        inter3_adjusted = self.inter3_conv(F.interpolate(inter[3], size=x.shape[2:]))
        x = self.dark5(self.att4(x + inter3_adjusted) * x)  # 16x16x512
        
        # Process inter[4] and add to dark5 output
        inter4_adjusted = self.inter4_conv(F.interpolate(inter[4], size=x.shape[2:]))
        x1 = self.deconv(self.att5(x + inter4_adjusted) * x)
        x = self.deconv(x1)
        x = self.swt1(x)
        outputs["dark5"] = x
        
        return {k: v for k, v in outputs.items() if k in self.out_features}


class SCConv(nn.Module):
    def __init__(self, planes, pooling_r):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(planes, planes, 3, 1, 1),
        )
        self.k3 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
        )
        self.k4 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        identity = x
        out = torch.sigmoid(
            torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:])))  # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out)  # k3 * sigmoid(identity + k2)
        out = self.k4(out)  # k4
        return out


class SCBottleneck(nn.Module):
    pooling_r = 4  # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, in_planes, planes):
        super(SCBottleneck, self).__init__()
        planes = int(planes / 2)
        self.conv1_a = nn.Conv2d(in_planes, planes, 1, 1)
        self.k1 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.conv1_b = nn.Conv2d(in_planes, planes, 1, 1)
        self.scconv = SCConv(planes, self.pooling_r)
        self.conv3 = nn.Conv2d(planes * 2, planes * 2, 1, 1)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x
        out_a = self.conv1_a(x)
        out_a = self.relu(out_a)
        out_a = self.k1(out_a)
        out_b = self.conv1_b(x)
        out_b = self.relu(out_b)
        out_b = self.scconv(out_b)
        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out += residual
        out = self.relu(out)
        return out


if __name__ == '__main__':
    print(CSPDarknet(1, 1))
