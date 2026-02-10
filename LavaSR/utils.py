import math
import torch
import librosa
import torchaudio

def load_wav(audio_file, resample_to=16000):
    wav, sr = librosa.load(audio_file, sr=48000, duration=20)
    wav = torch.tensor(wav)

    x = torchaudio.functional.resample(wav, sr, resample_to)
    x = torchaudio.functional.resample(x, resample_to, 16000).unsqueeze(0)
    return x

def wav_to_1s_batches(wav: torch.Tensor, sr: int):

    if wav.dim() == 2:
        wav = wav.squeeze(0)

    T = wav.shape[0]
    chunk = sr

    # compute padding needed to reach full seconds
    remainder = T % chunk
    pad_size = (chunk - remainder) % chunk  # 0 if already divisible

    if pad_size > 0:
        # repeat pad (wrap padding)
        repeats = math.ceil(pad_size / T)
        pad_src = wav.repeat(repeats)[:pad_size]
        wav = torch.cat([wav, pad_src], dim=0)

    # reshape into batches of 1s
    chunks = wav.view(-1, chunk)

    return chunks, pad_size

