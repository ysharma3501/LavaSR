import math
import torch
import librosa
import torchaudio

def load_wav(audio_file, resample_to=16000, duration=1000):
    wav, sr = librosa.load(audio_file, sr=48000, duration=duration)
    wav = torch.tensor(wav)

    x = torchaudio.functional.resample(wav, sr, resample_to)
    x = torchaudio.functional.resample(x, resample_to, 16000).unsqueeze(0)
    return x

def wav_to_1s_batches(wav: torch.Tensor, sr: int):
    # Ensure 1D tensor [samples]
    if wav.dim() == 2:
        wav = wav.squeeze(0)

    T = wav.shape[0]
    # Define chunk size for 1.28 seconds
    chunk = int(1.28 * sr)

    # Compute padding needed to reach a multiple of 1.28s
    remainder = T % chunk
    pad_size = (chunk - remainder) % chunk 

    if pad_size > 0:
        # Repeat the original audio to fill the pad_size
        repeats = math.ceil(pad_size / T)
        pad_src = wav.repeat(repeats)[:pad_size]
        wav = torch.cat([wav, pad_src], dim=0)

    # Reshape into batches of [N, chunk_size]
    chunks = wav.view(-1, chunk)

    return chunks, pad_size

