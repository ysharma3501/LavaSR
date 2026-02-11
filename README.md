# 🌋 LavaSR


LavaSR is a lightweight and high quality speech enhancement model that enhances low quality audio with noise into clean crisp audio with speeds reaching roughly 4000x realtime on GPU and 50x realtime on CPU.

https://github.com/user-attachments/assets/988c3726-eb6f-4877-93b9-cd5f0f488f8e


## Main features
- Extremely fast: Reaches speeds over 4000x realtime on GPUs and 50x realtime on CPUs
- High quality: Quality is on par with diffusion based models.
- Efficency: Just uses 500mb vram and potentially less.
- Universal input: Supports any input sampling rate from 8khz to 48khz.

## Usage
You can try it locally, colab, or spaces.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17wzpZ1nH_BrDsSfZ0JiZNdf4OH-zsfs2?usp=sharing)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/YatharthS/LavaSR)

#### Simple 1 line installation:
```
uv pip install git+https://github.com/ysharma3501/LavaSR.git
```

#### Load model:
```python
from LavaSR.model import LavaEnhance 

## change device to your torch device type(cuda, mps, etc.)
device = 'cpu'
lava_model = LavaEnhance("YatharthS/LavaSR", device)
```

#### Simple inference
```python
import soundfile as sf
from IPython.display import Audio

input_audio, input_sr = lava_model.load_audio('input.wav')

## Enhance Audio
output_audio = lava_model.enhance(input_audio).cpu().numpy().squeeze()

## Save Audio(both input and output)
sf.write('input.wav', input_audio.cpu().numpy().squeeze(), 16000)
sf.write('output.wav', output_audio, 16000)
```

#### Advanced inference
```python
import soundfile as sf
from IPython.display import Audio

cutoff = None ## Default is roughly half your sampling rate. You can lower it for higher quality but might sound "metallic".
input_sr = 16000 ## Change to any sr you want(from 8khz-48khz).
denoise = False ## Change this to True only if your audio has noise you want to filter.
batch = False ## Change this to True if audio is very long.

## Load Audio
input_audio, input_sr = lava_model.load_audio('input.wav', input_sr=input_sr)

## Enhance Audio
output_audio = lava_model.enhance(input_audio, denoise=denoise, batch=batch).cpu().numpy().squeeze()

## Save Audio(both input and output)
sf.write('input.wav', input_audio.cpu().numpy().squeeze(), 16000)
sf.write('output.wav', output_audio, 16000)
```

## Info

Q: How is this novel?

A: It adapts Vocos based architecture for BWE(bandwidth extension/audio upsampling). We also propose a novel triphase loss and a linkwitz-riley inspired refiner to further significantly increase quality.

Q: How is it so fast?

A: Because it uses the Vocos architecture which is isotropic and single pass, it's much faster then time-domain based and diffusion based models.

## Roadmap

- [x] Release model and code
- [x] Huggingface spaces demo
- [ ] Release training code
- [ ] Release model trained on music and audio

## Acknowledgments

- [Vocos](https://github.com/gemelo-ai/vocos.git) for their excellant architecture.
- [UL-UNAS](https://github.com/Xiaobin-Rong/ul-unas.git) for their great denoiser model.

## Final Notes

The model and code are licensed under the Apache-2.0 license. See LICENSE for details.

Stars/Likes would be appreciated, thank you.

Email: yatharthsharma3501@gmail.com
  
