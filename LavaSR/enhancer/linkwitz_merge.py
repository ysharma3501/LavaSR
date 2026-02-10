## Proposed work
## Linkwitz inspired class that essentially merges low freq from original audio and high freq from upsampled audio
## Smooths out any metallic artifacts

import torch

class FastLRMerge:
    def __init__(self, sample_rate=48000, cutoff=4000, transition_bins=256, device="cpu"):
        self.sample_rate = sample_rate
        self.cutoff = cutoff
        self.transition_bins = transition_bins
        self.device = device

        # cache: (n_bins, ndim) -> complex mask
        self.mask_cache = {}

        # Precompute fade curve once
        x = torch.linspace(-1, 1, steps=transition_bins, device=device)
        t = (x + 1) / 2
        self.fade_template = (3 * t**2 - 2 * t**3).to(torch.complex64)
      
    def _get_mask(self, n_bins, ndim):
        key = (n_bins, ndim)
        if key in self.mask_cache:
            return self.mask_cache[key]

        cutoff_bin = int((self.cutoff / (self.sample_rate / 2)) * n_bins)

        mask = torch.ones(n_bins, device=self.device, dtype=torch.complex64)

        half = self.transition_bins // 2
        start = max(0, cutoff_bin - half)
        end = min(n_bins, cutoff_bin + half)

        fade = self.fade_template[: end - start]

        mask[:start] = 0
        mask[start:end] = fade
        mask[end:] = 1

        # pre-expand to broadcast shape once
        for _ in range(ndim - 1):
            mask = mask.unsqueeze(0)

        self.mask_cache[key] = mask
        return mask

    def __call__(self, a, b):
        spec1 = torch.fft.rfft(a, dim=-1)
        spec2 = torch.fft.rfft(b, dim=-1)

        mask = self._get_mask(spec1.size(-1), spec1.ndim)

        # merged = spec2 + (spec1 - spec2) * mask
        spec2 += (spec1 - spec2) * mask

        return torch.fft.irfft(spec2, n=a.size(-1), dim=-1)
