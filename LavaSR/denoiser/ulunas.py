# Copyright (c) 2026 Xiaobin-Rong
# Licensed under the MIT license.

import torch
import numpy as np
import torch.nn as nn
from einops import rearrange


class ERB(nn.Module):
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft//2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(nfreqs-erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs-erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def hz2erb(self, freq_hz):
        erb_f = 21.4*np.log10(0.00437*freq_hz + 1)
        return erb_f

    def erb2hz(self, erb_f):
        freq_hz = (10**(erb_f/21.4) - 1)/0.00437
        return freq_hz

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        low_lim = erb_subband_1/nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points)/fs*nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        erb_filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) \
                                                / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2-2):
            erb_filters[i + 1, bins[i]:bins[i+1]] = (np.arange(bins[i], bins[i+1]) - bins[i] + 1e-12)\
                                                    / (bins[i+1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i+1]:bins[i+2]] = (bins[i+2] - np.arange(bins[i+1], bins[i + 2])  + 1e-12) \
                                                    / (bins[i + 2] - bins[i+1] + 1e-12)

        erb_filters[-1, bins[-2]:bins[-1]+1] = 1- erb_filters[-2, bins[-2]:bins[-1]+1]
        
        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))
    
    def bm(self, x):
        """x: (B,C,T,F)"""
        x_low = x[..., :self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1:])
        return torch.cat([x_low, x_high], dim=-1)
    
    def bs(self, x_erb):
        """x: (B,C,T,F_erb)"""
        x_erb_low = x_erb[..., :self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1:])
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


class AffinePReLU(nn.Module):
    def __init__(self, channels, width, init=0.25):
        super().__init__()
        self.affine_weight = nn.Parameter(torch.ones(channels, width))
        self.affine_bias = nn.Parameter(torch.zeros(channels, width))
        self.slope_weight = nn.Parameter(torch.empty(channels))
        nn.init.constant_(self.slope_weight, init)
    
    def forward(self, x):
        y = self.affine_weight[None,:,None,:] * x + self.affine_bias[None,:,None,:]
        y = y + torch.where(x>0, x, self.slope_weight.view(1,-1,1,1) * x)
        return y


class FA(nn.Module):
    def __init__(self, nfreq, freq_comp_ratio=4):
        super().__init__()
        self.r = freq_comp_ratio

        self.gru = nn.GRU(self.r, self.r, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*self.r, self.r)
        
        remainder = nfreq % self.r
        if remainder != 0:
            self.pad_len = self.r - remainder
        else:
            self.pad_len = 0

        self.F_pad = nfreq + self.pad_len
        self.H = self.F_pad // self.r

    def forward(self, x):
        B, C, T, F = x.shape
        x = torch.mean(x.pow(2), dim=1)  # (B,T,F)

        x = nn.functional.pad(x, (0, self.pad_len))
        x = x.view(B, T, self.H, self.r)

        x = rearrange(x, 'b t h c -> (b t) h c')
        x, _ = self.gru(x)  # (BT, H, 2C)
        x = self.fc(x)
        
        x = rearrange(x, '(b t) h c -> b t h c', b=B)
        
        x = x.reshape(B, T, self.F_pad)  # (B, T, F)

        if self.pad_len > 0:
            x = x[..., :F]

        return x


class cTFA(nn.Module):
    """causal time-frequency attention"""
    def __init__(self, channels, width):
        super().__init__()
        self.channels = channels
        self.ta_gru = nn.GRU(channels, channels*2, 1, batch_first=True)
        self.ta_fc = nn.Linear(channels*2, channels)
        
        self.fa = FA(width)

    def forward(self, x):
        """x: (B,C,T,F)"""
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        at = self.ta_gru(zt.transpose(1,2))[0]
        at = self.ta_fc(at).transpose(1,2)  # (B,C,T)
        at = torch.sigmoid(at)
        
        af = self.fa(x)
        af = torch.sigmoid(af)
        
        return at[...,None] * x * af[:, None]
    

class Shuffle(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """x: (B,2C,T,F)"""
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()  # (B,C,2,T,F)
        x = rearrange(x, 'b c g t f -> b (c g) t f')  # (B,2C,T,F)
        return x


class XConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        width,
        kernel_size,
        stride: int = 1,
        groups: int = 1,
        use_deconv: bool =False,
        is_last: bool=False,
    ):
        super().__init__()  
        self.g = groups
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        kt = kernel_size[0]
        
        if use_deconv:
            pt = kt - 1
            conv_module = nn.ConvTranspose2d
        else:
            pt = 0
            conv_module = nn.Conv2d
        
        pf = kernel_size[1]//2

        self.ops = nn.Sequential(
            nn.ZeroPad2d([0, 0, kt - 1, 0]),
            conv_module(in_channels, out_channels, kernel_size,
                        stride=(1, stride), padding=(pt, pf), groups=groups),
            nn.BatchNorm2d(out_channels),
            AffinePReLU(out_channels, width) if not is_last else nn.Identity(),
            cTFA(out_channels, width),
            Shuffle() if (not is_last and groups==2) else nn.Identity()
        )
    
    def forward(self, x):
        x = self.ops(x)
        return x


class XDWSBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        width,
        kernel_size,
        stride: int = 1,
        groups: int = 1,
        use_deconv: bool = False,
        is_last: bool =False,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        kt = kernel_size[0]
        
        if use_deconv:
            pt = kt - 1
            conv_module = nn.ConvTranspose2d
        else:
            pt = 0
            conv_module = nn.Conv2d
        
        pf = kernel_size[1]//2

        if stride == 2:
            if not use_deconv:
                in_width = width * 2 -1
            else:
                in_width = width // 2 + 1
        else:
            in_width = width

        self.pconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, groups=groups),
            nn.BatchNorm2d(out_channels),
            AffinePReLU(out_channels, in_width),
            Shuffle() if groups==2 else nn.Identity()
        )
        
        self.dconv = nn.Sequential(
            nn.ZeroPad2d([0, 0, kt - 1, 0]),
            conv_module(out_channels, out_channels, kernel_size,
                        stride=(1, stride), padding=(pt, pf), groups=out_channels),
            nn.BatchNorm2d(out_channels),
            AffinePReLU(out_channels, width) if not is_last else nn.Identity(),
            cTFA(out_channels, width)
        )     

        
    def forward(self, x):
        """x: (B, C, T, F)"""
        h = self.pconv(x)
        h = self.dconv(h)

        return h


class XMBBlocks(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        width,
        kernel_size,
        stride: int = 1,
        groups: int = 1,
        use_deconv: bool = False,
        is_last: bool = False,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        kt = kernel_size[0]
        
        if use_deconv:
            pt = kt - 1
            conv_module = nn.ConvTranspose2d
        else:
            pt = 0
            conv_module = nn.Conv2d
        
        pf = kernel_size[1]//2

        if stride == 2:
            if not use_deconv:
                in_width = width * 2 -1
            else:
                in_width = width // 2 + 1
        else:
            in_width = width

        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, groups=groups),
            nn.BatchNorm2d(out_channels),
            AffinePReLU(out_channels, in_width),
            Shuffle() if groups == 2 else nn.Identity()
        )
        self.dconv = nn.Sequential(
            nn.ZeroPad2d([0, 0, kt - 1, 0]),
            conv_module(out_channels, out_channels, kernel_size,
                        stride=(1, stride), padding=(pt, pf), groups=out_channels),
            nn.BatchNorm2d(out_channels),
            AffinePReLU(out_channels, width)
        )        
        self.pconv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, groups=groups),
            nn.BatchNorm2d(out_channels),
            cTFA(out_channels, width)
        )
        self.shuffle = Shuffle() if (not is_last and groups==2) else nn.Identity()

    def forward(self, x):
        input = x
        x = self.pconv1(x)
        x = self.dconv(x)
        x = self.pconv2(x)
        
        if x.shape == input.shape:
            x = x + input

        x = self.shuffle(x)

        return x
      

class GRNN(nn.Module):
    """Grouped RNN"""
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x, h=None):
        """
        x: (B, seq_length, input_size)
        h: (num_layers, B, hidden_size)
        """
        if h== None:
            if self.bidirectional:
                h = torch.zeros(self.num_layers*2, x.shape[0], self.hidden_size, device=x.device)
            else:
                h = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=x.device)
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)
        h1, h2 = h1.contiguous(), h2.contiguous()
        y1, h1 = self.rnn1(x1, h1)
        y2, h2 = self.rnn2(x2, h2)
        y = torch.cat([y1, y2], dim=-1)
        h = torch.cat([h1, h2], dim=-1)
        return y, h


class DPGRNN(nn.Module):
    """Grouped Dual-path RNN"""
    def __init__(self, input_size, width, hidden_size, **kwargs):
        super(DPGRNN, self).__init__(**kwargs)
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size

        self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size//2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, input_size)
        self.intra_ln = nn.LayerNorm((width, input_size), eps=1e-8)

        self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, input_size)
        self.inter_ln = nn.LayerNorm(((width, input_size)), eps=1e-8)
    
    def forward(self, x):
        """x: (B, C, T, F)"""
        ## Intra RNN
        x = x.permute(0, 2, 3, 1)  # (B,T,F,C)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (B*T,F,C)
        intra_x = self.intra_rnn(intra_x)[0]  # (B*T,F,C)
        intra_x = self.intra_fc(intra_x)      # (B*T,F,C)
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.input_size) # (B,T,F,C)
        intra_x = self.intra_ln(intra_x)
        intra_out = torch.add(x, intra_x)

        ## Inter RNN
        x = intra_out.permute(0,2,1,3)  # (B,F,T,C)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]) 
        inter_x = self.inter_rnn(inter_x)[0]  # (B*F,T,C)
        inter_x = self.inter_fc(inter_x)      # (B*F,T,C)
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.input_size) # (B,F,T,C)
        inter_x = inter_x.permute(0,2,1,3)   # (B,T,F,C)
        inter_x = self.inter_ln(inter_x) 
        inter_out = torch.add(intra_out, inter_x)
        
        dual_out = inter_out.permute(0,3,1,2)  # (B,C,T,F)
        
        return dual_out


class Encoder(nn.Module):
    def __init__(
        self,
        types,
        channels,
        widths,
        kernels,
        strides,
        groups
    ):
        super().__init__()
        block_types = [XConvBlock, XDWSBlock, XMBBlocks]
        n_blocks = len(types)
        en_convs = []
        in_channels = 1
        for i in range(n_blocks):
            module = block_types[types[i]]
            out_channels = channels[i]
            
            en_convs.append(module(in_channels, out_channels, widths[i],
                                   kernels[i], strides[i], groups=groups[i]))
            in_channels = out_channels
            
        self.en_convs = nn.ModuleList(en_convs)

    def forward(self, x):
        en_outs = []
        for i in range(len(self.en_convs)):
            x = self.en_convs[i](x)
            en_outs.append(x)
            # print(i, x.shape)

        return x, en_outs


class Decoder(nn.Module):
    def __init__(
        self,
        types,
        channels,
        widths,
        kernels,
        strides,
        groups,
        final_width
    ):
        super().__init__()
        block_types = [XConvBlock, XDWSBlock, XMBBlocks]
        n_blocks = len(types)
        de_convs = []
        in_channels = channels[-1]
        
        for i in range(n_blocks-1, 0, -1):
            module = block_types[types[i]]
            out_channels = channels[i-1]

            de_convs.append(module(in_channels, out_channels, widths[i-1],
                                   kernels[i], strides[i], groups[i], use_deconv=True))
            in_channels = out_channels

        module = block_types[types[0]]
        de_convs.append(module(in_channels, 1, final_width, kernels[0], strides[0], groups[0], use_deconv=True, is_last=True))
        
        self.de_convs = nn.ModuleList(de_convs)

    def forward(self, x, en_outs):
        n_blocks = len(self.de_convs)
        for i in range(n_blocks):
            x = self.de_convs[i](x + en_outs[n_blocks-i-1])
            # print(i, x.shape)
        x = torch.sigmoid(x)
        return x


class ULUNAS(nn.Module):
    def __init__(
        self,
        n_fft=512,
        hop_len=256,
        win_len=512,
        erb_low=65,
        erb_high=64,
        types=[0, 2, 1, 2, 1],
        strides=[2, 2, 1, 1, 1],
        groups=[1, 2, 2, 2, 2],
        channels=[12, 24, 24, 32, 16],
        kernels=[(3, 3), (2, 3), (2, 3), (1, 5), (1, 5)],
        widths=[65, 33, 33, 33, 33]
        
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        
        self.erb = ERB(erb_low, erb_high, nfft=n_fft, high_lim=8000, fs=16000)

        self.encoder = Encoder(types, channels, widths, kernels, strides, groups)
        
        self.dpgrnn = nn.Sequential(
                *[DPGRNN(channels[-1], widths[-1], channels[-1]) \
                    for i in range(2)]
            )
        
        self.decoder = Decoder(types, channels, widths, kernels, strides, groups, final_width=erb_low+erb_high)

    def forward(self, input):
        """
        input: (batch, n_samples)
        """
        device = input.device
        assert input.ndim == 2  # mono input
        n_samples = input.shape[1]
        
        stft_kwargs = {'n_fft': self.n_fft, 'hop_length': self.hop_len, 'win_length': self.win_len,
                       'window': torch.hann_window(self.win_len).to(device), 'onesided': True}
        
        spec = torch.stft(input,  **stft_kwargs, return_complex=True)
        spec = torch.view_as_real(spec)  # (B,F,T,2)

        spec = spec.permute(0,3,2,1)  # (B,2,T,F)
        feat = torch.log10(torch.norm(spec, dim=1, keepdim=True).clamp(1e-12))

        feat = self.erb.bm(feat)  # (B,4,T,129)
        
        feat, en_outs = self.encoder(feat)

        feat = self.dpgrnn(feat) # (B,16,T,33)

        m_feat = self.decoder(feat, en_outs)
        
        m = self.erb.bs(m_feat)

        spec_enh = spec * m
        spec_enh = spec_enh.permute(0,3,2,1)  # (B,F,T,2)
        
        spec_enh = torch.complex(spec_enh[...,0], spec_enh[...,1])
        output = torch.istft(spec_enh, **stft_kwargs)
        output = torch.nn.functional.pad(output, (0, n_samples-output.shape[1]))
        
        return output


if __name__ == "__main__":
    model = ULUNAS().eval()

    """complexity count"""
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (16000,), as_strings=False,
                                            print_per_layer_stat=False, verbose=False)
    params = 0
    for p in model.parameters():
        params += p.numel()
    print(f"The complexity of ULUNAS: MACs={macs/1e6:.2f} M, Params={params/1e3:.2f} k\n")


    """causality check"""
    a = torch.randn(1, 16000)
    b = torch.randn(1, 16000)
    c = torch.randn(1, 16000)
    x1 = torch.cat([a, b], dim=1)
    x2 = torch.cat([a, c], dim=1)

    y1 = model(x1)[0]
    y2 = model(x2)[0]

    stft_latency = 256*2
    err = (y1[:16000-stft_latency] - y2[:16000-stft_latency]).abs().max()
    if err < 1e-8:
        print("The model is causal, without any look ahead.")
