## Proposed work 
## This BWE model is based on Vocos, excellant speed with good quality.


import torch
import types
from vocos import Vocos
from torch.cuda.amp import autocast as autocast_func

## used to improve quality in end
from LavaSR.enhancer.linkwitz_merge import FastLRMerge

## quick monkey patch to improve quality slightly
def custom_forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of the ISTFTHead module.

    Args:
        x (Tensor): Input tensor of shape (B, L, H)

    Returns:
        Tensor: Reconstructed time-domain audio signal
    """
    x = self.out(x).transpose(1, 2)
    mag, p = x.chunk(2, dim=1)
    mag = torch.exp(mag)
    mag = torch.clip(mag, max=1e3)
    x_real = torch.cos(p)
    x_imag = torch.sin(p)
    S = mag * (x_real + 1j * x_imag)
    audio = self.istft(S)
    return audio
  
class LavaBWE:
    def __init__(self, model_path, device='cpu'):
      
        self.device = device
        self.lr_refiner = FastLRMerge(device=device)

        state_dict = torch.load(f"{model_path}/pytorch_model.bin", map_location="cpu")
        self.bwe_model = Vocos.from_hparams(f"{model_path}/config.yaml")

        self.bwe_model.load_state_dict(state_dict)
        self.bwe_model = self.bwe_model.eval().to(device)
    
        self.bwe_model.head.forward = types.MethodType(custom_forward, self.bwe_model.head)

        

    def infer(self, wav, autocast=False):
        """Inference function for bwe"""
      
        wav = wav.to(self.device)
        with torch.no_grad(), torch.autocast(self.device, dtype=torch.float16, enabled=autocast):
            features_input = self.bwe_model.feature_extractor(wav)
            features = self.bwe_model.backbone(features_input)
            pred_audio = self.bwe_model.head(features)
            with autocast_func(enabled=False):
                pred_audio = self.lr_refiner(pred_audio[:, :wav.shape[1]].float(), wav[:, :pred_audio.shape[1]].float())

        return pred_audio



