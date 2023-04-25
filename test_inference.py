import sys
import json
import torchaudio
import os
import glob
from spectrogram import linear_spectrogram, linear_to_mel, plot_spectrogram

sys.path.append("./hifigan")
from inference import load_checkpoint
from models import Generator
from env import AttrDict

# Load Generator
checkpoint_dict = load_checkpoint("./g_02500000", 'cpu')
with open('./hifigan/config_v1.json') as f:
    json_config = json.loads(f.read())
h = AttrDict(json_config)
generator = Generator(h)
generator.load_state_dict(checkpoint_dict['generator'])

if not os.path.exists("./outputs/"):
    os.mkdir("./outputs")

paths = glob.glob("./inputs/*.wav")
for i, path in enumerate(paths):
    wf, sr = torchaudio.load(path)
    wf = torchaudio.functional.resample(wf, sr, 22050)
    lin_spec = linear_spectrogram(wf)
    plot_spectrogram(lin_spec[0], f"./outputs/{i}.png")
    spec = linear_to_mel(lin_spec)
    wf = generator(spec)
    wf = torchaudio.functional.resample(wf, 22050, sr)[0]
    torchaudio.save(filepath=os.path.join("./outputs/", f"{i}.wav"), src=wf, sample_rate=sr)
