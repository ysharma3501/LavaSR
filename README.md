# 🌋 LavaSR
<p align="center">
  <a href="https://huggingface.co/YatharthS/LavaSR">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E" alt="Hugging Face Model">
  </a>
  &nbsp;
  <a href="https://huggingface.co/spaces/YatharthS/LavaSR">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue" alt="Hugging Face Space">
  </a>
  &nbsp;
  <a href="https://colab.research.google.com/drive/17wzpZ1nH_BrDsSfZ0JiZNdf4OH-zsfs2?usp=sharing">
    <img src="https://img.shields.io/badge/Colab-Notebook-F9AB00?logo=googlecolab&logoColor=white" alt="Colab Notebook">
  </a>
  &nbsp;
  <a href="https://arxiv.org/abs/2603.07285">
    <img src="https://img.shields.io/badge/arXiv-2603.07285-B31B1B.svg" alt="arXiv Paper">
  </a>
</p>

LavaSR is a lightweight and high quality speech enhancement model that enhances low quality audio with noise into clean crisp audio with speeds reaching roughly 5000x realtime on GPU and over 60x realtime on CPU.

**Interspeech acceptance**: LavaSR v2's paper has been accepted to main track Interspeech 2026!

**LavaSR v2 just released**: Massive increase in quality and speed, surpassing 6gb slow diffusion models. Check it out!

https://github.com/user-attachments/assets/1d11ae30-cb19-4c9b-ac46-52adbcac957f

## Main features
- Extremely fast: Reaches speeds over 5000x realtime on GPUs and 50x realtime on CPUs
- High quality: Quality surpasses diffusion models.
- Efficency: Just uses 500mb vram and potentially less.
- Universal input: Supports any input sampling rate from 8khz to 48khz.

### Why is this useful?
* Enhancing TTS: LavaSR can enhance TTS(text-to-speech) model quality considerably with nearly 0 computational cost.
* Real-time enhancement: LavaSR allows for on device enhancement of any low quality calls, audio, etc. while using little memory.
* Restoring datasets: LavaSR can enhance audio quality of any audio dataset.


### Comparisons

Quality comparisons using Log-Spectral-distance on VCTK validation. Lower is better(more similar to original 48khz file).
| Method              | 8→48 kHz | 16→48 kHz | 24→48 kHz |
|--------------------|----------------|-----------------|-----------------|
| Sinc upsampling     | 2.98           | 2.75            | 2.17            |
| AudioSR (diffusion) | 1.13           | 0.98            | 0.82            |
| NU-WAVE2(diffusion) | 1.10           | 0.94            | 0.87            |
| AP-BWE(previous best) | 0.86           | 0.74            | 0.64            |
| **Proposed model**  | **0.85**       | **0.72**        | **0.63**        |

Speed Comparisons were done on A100 gpu. Higher realtime means faster processing speeds.

| Model         | Speed (Real-Time) | Model Size |
| :------------ | :---------------- | :--------- |
| **LavaSR** | **5000x realtime** | **~50 MB** |
| AP-BWE      | 300x realtime        | ~70 MB     |
| FlowHigh      | 80x realtime        | ~450 MB     |
| FlashSR       | 14x realtime        | ~1000 MB     |
| AudioSR       | 0.6x realtime    | ~6000 MB     |

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
from LavaSR.model import LavaEnhance2 

## change device to your torch device type(cuda, mps, etc.)
device = 'cpu'
lava_model = LavaEnhance2("YatharthS/LavaSR", device)
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
sf.write('output.wav', output_audio, 48000)
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
sf.write('output.wav', output_audio, 48000)
```

## Community
- [ComfyUI node](https://github.com/marduk191/ComfyUI-LavaSR): Code to use LavaSR in comfyui.
- [GUI App](https://github.com/faxlab/LavaSR-Fast-Enhancer): GUI app for LavaSR.
- [C++ implementation](https://github.com/Topping1/LavaSRcpp): Purely C++ implementation for LavaSR.
- [Onnx implementation](https://github.com/Topping1/LavaSR-ONNX): Clean and simple Onnx implementation for LavaSR.
- [ONNX Embedded](https://github.com/idocinthebox/LavaSR-ONNX-Embedded): ONNX runtime with weights embedded in graph, DirectML/NPU support, and export script.
- [FalAI demo](https://fal.ai/models/fal-ai/lava-sr): LavaSR demo hosted by FalAI
  
Thanks to the community for building on top of the model!

## Info

Q: How is this novel?

A: It adapts Vocos based architecture for BWE(bandwidth extension/audio upsampling). We also propose linkwitz-riley inspired refiner to further significantly increase quality.

Q: How is it so fast?

A: Because it uses the Vocos architecture which is isotropic and single pass, it's much faster then time-domain based and diffusion based models. Vocos scales incredibly well as well.

Q: What is it trained on?

A: It starts off from a Vocos prior and trained with just 50k steps on VCTK dataset. Dataset is randomly resampled to 8khz/16khz/24khz and randomly noise is added.
## Roadmap

- [x] Release model and code
- [x] Huggingface spaces demo
- [x] Release model with no metallic issue.
- [ ] Release training code
- [ ] Release model trained on music and audio

## Acknowledgments

- [Vocos](https://github.com/gemelo-ai/vocos.git) for their excellant architecture.
- [UL-UNAS](https://github.com/Xiaobin-Rong/ul-unas.git) for their great denoiser model.

## Final Notes

Currently writing an Interspeech paper for LavaSR, receiving feedback from community would be great.

The model and code are licensed under the Apache-2.0 license. See LICENSE for details.

Stars/Likes would be appreciated, thank you.

Email: yatharthsharma3501@gmail.com
  
