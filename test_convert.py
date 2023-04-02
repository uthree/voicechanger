import torch
import torchaudio
from cyclegan import Generator
import os
from tqdm import tqdm
import argparse
from spectrogram import pack_spec_and_phase, unpack_spec_and_phase, wave_to_spec_and_phase, spec_and_phase_to_wave

parser = argparse.ArgumentParser(description="Convert audio files")

parser.add_argument('-d', '--device', default='cpu', choices=['cpu', 'cuda', 'mps'],
                    help="Device setting. Set this option to cuda if you need to use ROCm.")
parser.add_argument('-i', '--input', default='./inputs',
                    help="Input directory")
parser.add_argument('-o', '--output', default='./outputs',
                    help="Output directory")
parser.add_argument('-cp', '--cycleganpath', default='./g_a2b.pt',
                    help="Path of generator model")
parser.add_argument('-ps', '--pitch-shift', default=0, type=int)
parser.add_argument('-g', '--gain', default=1.0, type=float)
parser.add_argument('-ig', '--input-gain', default=1.0, type=float)


args = parser.parse_args()
device_name = args.device

if device_name == 'cuda':
    if not torch.cuda.is_available():
        print("Error: cuda is not available in this environment.")
        exit()

if device_name == 'mps':
    if not torch.backends.mps.is_built():
        print("Error: mps is not available in this environment.")
        exit()

device = torch.device(device_name)

G = Generator()
G.to(device)
G.load_state_dict(torch.load(args.cycleganpath, map_location=device))
print(f"Loaded model from {args.cycleganpath}.")

if args.pitch_shift != 0:
    pitch_shift = torchaudio.transforms.PitchShift(22050, args.pitch_shift).to(device)
else:
    pitch_shift = torch.nn.Identity()

if not os.path.exists(args.output):
    os.mkdir(args.output)

for i, fname in enumerate(os.listdir(args.input)):
    if fname == ".DS_Store":
        continue
    print(f"Converting {fname}")
    with torch.no_grad():
        wf, sr = torchaudio.load(os.path.join(args.input, fname))
        wf = wf * args.input_gain
        s, p = wave_to_spec_and_phase(wf)
        s = G(s)
        wf = spec_and_phase_to_wave(s, p)
        wf = wf * args.gain
        wf = wf.to(torch.float32).to('cpu').detach()
        out_path = os.path.join(args.output, f"output_{fname}_{i}.wav")
        torchaudio.save(out_path, src=wf, sample_rate=sr)
