# C:\Users\user\anaconda3\envs\smoke_jetson\lib\site-packages\ultralytics\nn\smoke_backbone.py

import torch
import torch.nn as nn
import torch.nn.functional as F
# C2f 임포트가 필요 없으므로 삭제하고, 절대 경로로 수정
from ultralytics.nn.modules.conv import Conv, RepConv 

# -----------------------------------------------------------------
# 1. ECPConvBlock (백본 강화를 위한 모듈)
# -----------------------------------------------------------------

class ECPConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, ratio=0.25):
        super(ECPConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = int(in_channels * ratio)
        self.rest_k = in_channels - self.k
        self.channel_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            Conv(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
        self.rep_conv = RepConv(self.k, self.k, kernel_size, stride, padding=kernel_size//2, groups=self.k)
        self.rest_conv = Conv(self.rest_k, self.rest_k, 1, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        scores = self.channel_selector(x).squeeze()
        if not hasattr(self, 'fixed_indices'):
            scores_avg = scores.mean(dim=0)
            _, fixed_indices = torch.topk(scores_avg, self.in_channels)
            self.register_buffer('fixed_k_indices', fixed_indices[:self.k].sort()[0])
            self.register_buffer('fixed_rest_indices', fixed_indices[self.k:].sort()[0])
        selected_channels = torch.index_select(x, 1, self.fixed_k_indices)
        rest_channels = torch.index_select(x, 1, self.fixed_rest_indices)
        conv_result = self.rep_conv(selected_channels)
        rest_result = self.rest_conv(rest_channels)
        output = torch.empty_like(x)
        output.index_copy_(1, self.fixed_k_indices, conv_result)
        output.index_copy_(1, self.fixed_rest_indices, rest_result)
        return output

class EMA(nn.Module):
    def __init__(self, channels, groups=8):
        super(EMA, self).__init__()
        self.groups = groups
        self.c_per_g = channels // groups
        self.conv_1x1_h = nn.Conv2d(self.c_per_g, self.c_per_g, (1, 3), padding=(0, 1))
        self.conv_1x1_w = nn.Conv2d(self.c_per_g, self.c_per_g, (3, 1), padding=(1, 0))
        self.conv_3x3 = nn.Conv2d(self.c_per_g, self.c_per_g, 3, padding=1, groups=self.c_per_g)
        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(groups, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x_group = x.view(b * self.groups, self.c_per_g, h, w)
        x_h = F.adaptive_avg_pool2d(x_group, (h, 1))
        x_w = F.adaptive_avg_pool2d(x_group, (1, w))
        x_h = self.conv_1x1_h(x_h)
        x_w = self.conv_1x1_w(x_w)
        attn_1x1 = x_h + x_w
        attn_1x1 = self.sigmoid(attn_1x1)
        out_1x1 = x_group * attn_1x1
        out_3x3 = self.conv_3x3(x_group)
        fused = self.gn(out_1x1 + out_3x3)
        fused = self.sigmoid(fused)
        return fused.view(b, c, h, w) * x

class ECPConvBlock(nn.Module):
    """ ECPConvBlock: ResNetBlock 시그니처와 호환되도록 수정됨 """
    def __init__(self, c1, c2, s=1, e=4):  # ResNetBlock과 동일한 인자
        super().__init__()
        c_out = c2 * e  

        self.ecp_conv = ECPConv(c1, c1, k=3, s=s)
        self.pw_conv = Conv(c1, c_out, 1)
        self.ema = EMA(c_out)

        self.shortcut = nn.Identity()
        if s != 1 or c1 != c_out: 
            self.shortcut = Conv(c1, c_out, 1, s=s)

    def forward(self, x):
        return self.ema(self.pw_conv(self.ecp_conv(x)) + self.shortcut(x))

# MFFPN과 RCM 클래스는 여기서 삭제합니다.