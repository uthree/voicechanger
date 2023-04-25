import sys
import json
import torchaudio
import os
import glob
from spectrogram import linear_spectrogram, linear_to_mel, plot_spectrogram
from cyclegan import Generator as Convertor
import torch

sys.path.append("./hifigan")
from inference import load_checkpoint
from models import Generator as Vocoder
from env import AttrDict

# Load Vocoder
checkpoint_dict = load_checkpoint("./g_02500000", 'cpu')
with open('./hifigan/config_v1.json') as f:
    json_config = json.loads(f.read())
h = AttrDict(json_config)
vocoder = Vocoder(h)
vocoder.load_state_dict(checkpoint_dict['generator'])

# Load convertor
convertor = Convertor()
convertor.load_state_dict(torch.load("./g_a2b.pt", map_location='cpu'))

if not os.path.exists("./outputs/"):
    os.mkdir("./outputs")

paths = glob.glob("./inputs/*.wav")
for i, path in enumerate(paths):
    wf, sr = torchaudio.load(path)
    wf = torchaudio.functional.resample(wf, sr, 22050)
    with torch.no_grad():
        lin_spec = linear_spectrogram(wf)
        plot_spectrogram(lin_spec[0], f"./outputs/{i}_input.png")
        lin_spec = convertor(lin_spec)
        plot_spectrogram(lin_spec[0], f"./outputs/{i}_output.png")
        spec = linear_to_mel(lin_spec)
    wf = vocoder(spec)
    wf = torchaudio.functional.resample(wf, 22050, sr)[0]
    wf = wf.detach()
    torchaudio.save(filepath=os.path.join("./outputs/", f"{i}.wav"), src=wf, sample_rate=sr)
