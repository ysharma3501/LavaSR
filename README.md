# LavaSR
Fast speech restoration

```python
from LavaSR.model import LavaEnhance 

device = 'cpu'
lava_model = LavaEnhance("modelpath", device)
```

```python
from IPython.display import Audio
import torch

audio, input_sr = lava_model.load_audio('input.wav')

wav = lava_model.enhance(audio, denoise=False, batch=False)

## low quality audio
display(Audio(audio.cpu(), rate=16000))
## high quality audio
display(Audio(wav.cpu(), rate=48000))
```
