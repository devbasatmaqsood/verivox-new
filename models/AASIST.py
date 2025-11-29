"""
AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import random
from typing import Union

from mamba_ssm import Mamba
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        # Forward Mamba
        self.mamba_fwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        # Backward Mamba
        self.mamba_bwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Dim)
        x_fwd = self.mamba_fwd(x)
        
        # Flip for backward pass to model bidirectional context
        x_rev = torch.flip(x, [1])
        x_bwd = self.mamba_bwd(x_rev)
        x_bwd = torch.flip(x_bwd, [1])
        
        # Combine
        x_out = x_fwd + x_bwd
        return self.norm(x_out)

class CONV(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    def __init__(self,
                 out_channels,
                 kernel_size,
                 sample_rate=16000,
                 in_channels=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 groups=1,
                 mask=False):
        super().__init__()
        if in_channels != 1:

            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (
                in_channels)
            raise ValueError(msg)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)

        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2,
                                  (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2*fmax/self.sample_rate) * \
                np.sinc(2*fmax*self.hsupp/self.sample_rate)
            hLow = (2*fmin/self.sample_rate) * \
                np.sinc(2*fmin*self.hsupp/self.sample_rate)
            hideal = hHigh - hLow

            self.band_pass[i, :] = Tensor(np.hamming(
                self.kernel_size)) * Tensor(hideal)

    def forward(self, x, mask=False):
        band_pass_filter = self.band_pass.clone().to(x.device)
        if mask:
            A = np.random.uniform(0, 20)
            A = int(A)
            A0 = random.randint(0, band_pass_filter.shape[0] - A)
            band_pass_filter[A0:A0 + A, :] = 0
        else:
            band_pass_filter = band_pass_filter

        self.filters = (band_pass_filter).view(self.out_channels, 1,
                                               self.kernel_size)

        return F.conv1d(x,
                        self.filters,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        bias=None,
                        groups=1)


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False
        self.mp = nn.MaxPool2d((1, 3))  # self.mp = nn.MaxPool2d((1,4))

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x
        out = self.conv1(x)

        # print('out',out.shape)
        out = self.bn2(out)
        out = self.selu(out)
        # print('out',out.shape)
        out = self.conv2(out)
        #print('conv2 out',out.shape)
        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out


class Model(nn.Module):
    def __init__(self, d_args):
        super().__init__()

        self.d_args = d_args
        filts = d_args["filts"]
        
        # --- Frontend (Same as AASIST) ---
        self.conv_time = CONV(out_channels=filts[0],
                              kernel_size=d_args["first_conv"],
                              in_channels=1)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))

        # --- Mamba Backend (Replaces Graph Logic) ---
        self.backend_dim = filts[-1][-1]
        
        # Spectral Mamba (Frequency Sequence)
        self.pos_S = nn.Parameter(torch.randn(1, 23, self.backend_dim)) 
        self.mamba_S = BiMambaBlock(d_model=self.backend_dim)

        # Temporal Mamba (Time Sequence)
        self.mamba_T = BiMambaBlock(d_model=self.backend_dim)
        
        # Pooling and Output
        self.pool_drop = nn.Dropout(0.3)
        self.out_layer = nn.Linear(self.backend_dim * 4, 2) # Concatenating Max/Avg from both streams

    def forward(self, x, Freq_aug=False):
        # x: (Batch, Samples)
        x = x.unsqueeze(1)
        x = self.conv_time(x, mask=Freq_aug)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # Encoder output: (Batch, Channels, Freq, Time)
        e = self.encoder(x)
        
        # --- Prepare Spectral Stream ---
        # Pool over time to get Frequency sequence
        e_S, _ = torch.max(torch.abs(e), dim=3) # (Batch, Channels, Freq)
        e_S = e_S.transpose(1, 2) # (Batch, Freq, Channels)
        
        # Add positional embedding if shapes match (simple robustness check)
        if e_S.shape[1] == self.pos_S.shape[1]:
             e_S = e_S + self.pos_S
        
        # Process with Mamba (Spectral)
        out_S = self.mamba_S(e_S) 
        
        # --- Prepare Temporal Stream ---
        # Pool over freq to get Time sequence
        e_T, _ = torch.max(torch.abs(e), dim=2) # (Batch, Channels, Time)
        e_T = e_T.transpose(1, 2) # (Batch, Time, Channels)
        
        # Process with Mamba (Temporal)
        out_T = self.mamba_T(e_T)
        
        # --- Readout (Pooling) ---
        # 1. Max and Avg Pooling for Spectral
        s_max, _ = torch.max(out_S, dim=1)
        s_avg = torch.mean(out_S, dim=1)
        
        # 2. Max and Avg Pooling for Temporal
        t_max, _ = torch.max(out_T, dim=1)
        t_avg = torch.mean(out_T, dim=1)
        
        # Combine all features
        feat = torch.cat([s_max, s_avg, t_max, t_avg], dim=1)
        
        feat = self.pool_drop(feat)
        output = self.out_layer(feat)

        return feat, output