import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-mp', '--model-path', default='./g_a2b.pt')
parser.add_argument('-ig', '--input-gain', default=1.0, type=float)
parser.add_argument('-g', '--gain', default=1.0, type=float)
parser.add_argument('-ps', '--pitch-shift', default=0, type=int)

args = parser.parse_args()

device = torch.device(args.device)

# Load Vocoder
checkpoint_dict = load_checkpoint("./g_02500000", device)
with open('./hifigan/config_v1.json') as f:
    json_config = json.loads(f.read())
h = AttrDict(json_config)
vocoder = Vocoder(h)
vocoder = vocoder.to(device)
vocoder.load_state_dict(checkpoint_dict['generator'])

# Load convertor
convertor = Convertor()
convertor.load_state_dict(torch.load(args.model_path, map_location=device))
convertor = convertor.to(device)

if args.pitch_shift != 0:
    ps = torchaudio.transforms.PitchShift(22050, args.pitch_shift).to(device)
else:
    ps = torch.nn.Identity().to(device)


if not os.path.exists("./outputs/"):
    os.mkdir("./outputs")

paths = glob.glob("./inputs/*.wav")
for i, path in enumerate(paths):
    wf, sr = torchaudio.load(path)
    wf = wf.to(device)
    wf = torchaudio.functional.resample(wf, sr, 22050) * args.input_gain
    with torch.no_grad():
        print(f"converting {path}")
        wf = ps(wf)
        lin_spec = linear_spectrogram(wf)
        plot_spectrogram(lin_spec.detach().cpu()[0], os.path.join("./outputs/", f"{i}_input.png"))
        lin_spec = convertor(lin_spec)
        plot_spectrogram(lin_spec.detach().cpu()[0], os.path.join("./outputs/", f"{i}_output.png"))
        spec = linear_to_mel(lin_spec)
    wf = vocoder(spec)
    wf = torchaudio.functional.resample(wf, 22050, sr)[0] * args.gain
    wf = wf.cpu().detach()
    torchaudio.save(filepath=os.path.join("./outputs/", f"{i}.wav"), src=wf, sample_rate=sr)
