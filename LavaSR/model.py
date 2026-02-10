import torch
import torchaudio
from huggingface_hub import snapshot_download

from LavaSR.enhancer.enhancer import LavaBWE
from LavaSR.denoiser.denoiser import LavaDenoiser
from LavaSR.utils import wav_to_1s_batches, load_wav
from LavaSR.enhancer.linkwitz_merge import FastLRMerge



class LavaEnhance:
    def __init__(self, model_path="YaTharThShaRma999/LavaSR", device='cpu'):

        if model_path == "YaTharThShaRma999/LavaSR":
            model_path = snapshot_download(model_path)

        self.device = device
        self.bwe_model = LavaBWE(f"{model_path}/enhancer", device=device) ## proposed work
        self.denoiser_model = LavaDenoiser(f'{model_path}/denoiser/denoiser.bin', device=device) ## based on UL-UNAS
        

    def enhance(self, wav, enhance=True, denoise=True, batch=False):
        pad_size = 0
        low_quality_audio = wav

        if batch:
            wav, pad_size = wav_to_1s_batches(wav, 16000)

        if denoise:
            with torch.inference_mode():
                wav = self.denoiser_model.infer(wav)
                wav = torchaudio.functional.resample(wav, 16000, 48000)
        else:
            wav = torchaudio.functional.resample(wav, 16000, 48000)
    
        if enhance:
            with torch.no_grad():
                wav = self.bwe_model.infer(wav).reshape(-1)
        else:
            wav = wav.reshape(-1)

        return wav

    def load_audio(self, file_path, input_sr=16000, cutoff=None):
        x = load_wav(file_path, resample_to=input_sr).to(self.device)
        
        if cutoff == None:
            cutoff = input_sr//2
          
        self.bwe_model.lr_refiner = FastLRMerge(device=self.device, cutoff=cutoff, transition_bins=1024)
      
        return x, input_sr
