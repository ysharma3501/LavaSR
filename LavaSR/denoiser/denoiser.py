import torch
import torchaudio

from LavaSR.denoiser.ulunas import ULUNAS

class LavaDenoiser:
    def __init__(self, model_path, device='cpu'):

        self.device = device
        self.model = ULUNAS().to(device).eval()

        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)      

    def infer(self, wav):

        wav = wav.to(self.device)
        with torch.inference_mode():
            wav = self.model(wav)
          
        return wav
