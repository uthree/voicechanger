import torch


# waveform [N, L] -> spectrogram [N, C, L], phase [N, C, L]
def wave_to_spec_and_phase(x, n_fft=512, hop_length=128, downsample_rate=2):
    wh = n_fft // 8
    x = x[:, wh:-wh:downsample_rate]
    spec_complex = torch.stft(x, n_fft, return_complex=True, hop_length=hop_length)
    spec = spec_complex.abs()
    phase = spec_complex / (spec_complex.abs() + 1e-6)
    return spec, phase


# spectrogram [N, C, L], phase [N, C, L] -> waveform [N, L]
def spec_and_phase_to_wave(spec, phase, n_fft=512, hop_length=128, upsample_rate=2):
    x = torch.istft(spec * phase, n_fft=n_fft, hop_length=hop_length)
    x = upsample(x, upsample_rate)
    return x


def upsample(x, upsample_rate=1):
    x = x.unsqueeze(0)
    return torch.nn.functional.interpolate(x, [x.shape[2] * upsample_rate]).squeeze(0)


# Spec, Phase -> Tensor(Real Number) [N, C, L]
def pack_spec_and_phase(spec, phase):
    phase = torch.cat([phase.real, phase.imag], dim=1)
    x = torch.cat([spec, phase], dim=1)
    return x


# Tensor(Real Number) -> Spec, Phase
def unpack_spec_and_phase(x):
    l = x.shape[1] // 3
    spec = x[:, :l]
    phase_real = x[:, l:l*2]
    phase_imag = x[:, l*2:]
    phase = torch.complex(phase_real, phase_imag)
    phase = phase / (phase.abs() + 1e-6)
    return spec, phase
