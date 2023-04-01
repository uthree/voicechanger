import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from tqdm import tqdm

from cyclegan import Generator, Discriminator
from stt import STTEncoder
from dataset import WaveFileDirectory
from spectrogram import wave_to_spec_and_phase, pack_spec_and_phase


def load_or_init_models(device=torch.device('cpu')):
    paths = ["./g_a2b.pt", "./g_b2a.pt", "./d_a.pt", "./d_b.pt", "./stt_encoder.pt"]
    model_classes = [Generator, Generator, Discriminator, Discriminator, STTEncoder]
    models = []
    for cls, p in zip(model_classes, paths):
        if os.path.exists(p):
            m = cls()
            m.load_state_dict(torch.load(p, map_location=device))
            m.to(device)
            models.append(m)
            print(f"Loaded model from {p}")
        else:
            models.append(cls().to(device))
            print(f"Initialized {p}")
    return models


def save_models(Gab, Gba, Da, Db):
    torch.save(Gab.state_dict(), "./g_a2b.pt")
    torch.save(Gba.state_dict(), "./g_b2a.pt")
    torch.save(Da.state_dict(), "./d_a.pt")
    torch.save(Db.state_dict(), "./d_b.pt")
    print("Saved models")


def cutmid(x):
    length = x.shape[2]
    s = length // 8
    e = length - (length // 8)
    return x[:, :, s:e]


parser = argparse.ArgumentParser(description="Train Cycle GAN")

parser.add_argument('dataset_path_a')
parser.add_argument('dataset_path_b')
parser.add_argument('-d', '--device', default='cpu', choices=['cpu', 'cuda', 'mps'],
                    help="Device setting. Set this option to cuda if you need to use ROCm.")
parser.add_argument('-e', '--epoch', default=60, type=int)
parser.add_argument('-b', '--batch', default=16, type=int)
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-m', '--maxdata', default=-1, type=int, help="max dataset size")
parser.add_argument('-lr', '--learningrate', default=2e-4, type=float)
parser.add_argument('-len', '--length', default=65536*2, type=int)
parser.add_argument('--consistency', default=10.0, type=float, help="weight of cycle-consistency loss")
parser.add_argument('--identity', default=1.0, type=float, help="weight of identity loss")
parser.add_argument('--stt', default=0.0, type=float, help="weight of STT")
parser.add_argument('--preview', default=False, type=bool, help="flag of writing preview during training")
parser.add_argument('-psa', '--pitch-shift-a', default=0, type=int)
parser.add_argument('-psb', '--pitch-shift-b', default=0, type=int)

args = parser.parse_args()
device_name = args.device

print(f"selected device: {device_name}")
if device_name == 'cuda':
    if not torch.cuda.is_available():
        print("Error: cuda is not available in this environment.")
        exit()

if device_name == 'mps':
    if not torch.backends.mps.is_built():
        print("Error: mps is not available in this environment.")
        exit()

if torch.cuda.is_available() and device_name != "cuda":
    print(f"Warning: CUDA is available in this environment, but selected device is {device_name}. training process will may be slow.")

device = torch.device(device_name)

Gab, Gba, Da, Db, STT = load_or_init_models(device)

ds_a = WaveFileDirectory(
        [args.dataset_path_a],
        length=args.length,
        max_files=args.maxdata)

ds_b = WaveFileDirectory(
        [args.dataset_path_b],
        length=args.length,
        max_files=args.maxdata)

dl_a = torch.utils.data.DataLoader(ds_a, batch_size=args.batch, shuffle=True)
dl_b = torch.utils.data.DataLoader(ds_b, batch_size=args.batch, shuffle=True)

OGab = optim.Adam(Gab.parameters(), lr=args.learningrate, betas=(0.5, 0.999))
OGba = optim.Adam(Gba.parameters(), lr=args.learningrate, betas=(0.5, 0.999))
ODa = optim.Adam(Da.parameters(), lr=args.learningrate, betas=(0.5, 0.999))
ODb = optim.Adam(Db.parameters(), lr=args.learningrate, betas=(0.5, 0.999))

if args.pitch_shift_a != 0:
    Ta = torchaudio.transforms.PitchShift(22050, args.pitch_shift_a).to(device)
else:
    Ta = nn.Identity().to(device)

if args.pitch_shift_b != 0:
    Tb = torchaudio.transforms.PitchShift(22050, args.pitch_shift_b).to(device)
else:
    Tb = nn.Identity().to(device)

scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
L1 = nn.L1Loss()

Lcyc = args.consistency
Lid = args.identity
Lstt = args.stt

for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=min(len(ds_a), len(ds_b)))
    for batch, (real_a, real_b) in enumerate(zip(dl_a, dl_b)):
        if real_a.shape[0] != real_b.shape[0]:
            continue
        N = real_a.shape[0]
        
        # Pitch Shift
        real_a = Ta(real_a.to(device))
        s, p = wave_to_spec_and_phase(real_a)
        real_a = pack_spec_and_phase(s, p).detach()
        real_b = Tb(real_b.to(device))
        s, p = wave_to_spec_and_phase(real_b)
        real_v = pack_spec_and_phase(s, p).detach()

        # Train G.
        OGab.zero_grad()
        OGba.zero_grad()

        with torch.cuda.amp.autocast(enabled=args.fp16):
            fake_b = Gab(real_a)
            fake_a = Gba(real_b)
            recon_a = Gba(fake_b)
            recon_b = Gab(fake_a)
            loss_G_cyc = L1(recon_b, real_b) + L1(recon_a, real_a)
            loss_G_id = L1(Gab(real_b), real_b) + L1(Gba(real_a), real_a)
            if args.stt != 0.0:
                loss_G_stt = STT.feat_loss(fake_b, real_a) + STT.feat_loss(fake_a, real_b)
            else:
                loss_G_stt = torch.tensor([0]).to(device)
            loss_G_adv = F.relu(-Db(cutmid(fake_b))).mean() +\
                F.relu(-Da(cutmid(fake_a))).mean()
            loss_G = loss_G_adv + loss_G_id * Lid + loss_G_cyc * Lcyc + loss_G_stt * Lstt

        scaler.scale(loss_G).backward()

        nn.utils.clip_grad_norm_(Gab.parameters(), max_norm=1.0, norm_type=2.0)
        nn.utils.clip_grad_norm_(Gba.parameters(), max_norm=1.0, norm_type=2.0)

        fake_a = fake_a.detach()
        fake_b = fake_b.detach()
        recon_a = recon_a.detach()
        recon_b = recon_b.detach()

        scaler.step(OGab)
        scaler.step(OGba)

        # Train D.
        ODa.zero_grad()
        ODb.zero_grad()

        with torch.cuda.amp.autocast(enabled=args.fp16):
            loss_D = F.relu(0.5 + Da(cutmid(fake_a))).mean() +\
                F.relu(0.5 + Db(cutmid(fake_b))).mean() +\
                F.relu(0.5 - Da(real_a)).mean() +\
                F.relu(0.5 - Db(real_b)).mean()

        scaler.scale(loss_D).backward()

        nn.utils.clip_grad_norm_(Da.parameters(), max_norm=1.0, norm_type=2.0)
        nn.utils.clip_grad_norm_(Db.parameters(), max_norm=1.0, norm_type=2.0)

        scaler.step(ODa)
        scaler.step(ODb)

        scaler.update()
        
        tqdm.write(f"Id: {loss_G_id.item():.4f}, Adv.: {loss_G_adv.item():.4f}, Cyc.: {loss_G_cyc.item():.4f}, STT: {loss_G_stt.item():.4f}")
        bar.set_description(desc=f"G: {loss_G.item():.4f}, D: {loss_D.item():.4f}")
        bar.update(N)

        if loss_D.isnan().any() or loss_G.isnan().any():
            exit()

        if batch % 100 == 0:
            save_models(Gab, Gba, Da, Db)
