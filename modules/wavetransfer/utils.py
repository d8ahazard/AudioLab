import torch
import torchaudio
import numpy as np
from modules.wavetransfer.params import get_default_params
from librosa.filters import mel as librosa_mel_fn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

# from https://github.com/jik876/hifi-gan/blob/master/utils.py
def plot_spectrogram(spectrogram):
  fig, ax = plt.subplots(figsize=(10, 2))
  im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                  interpolation='none')
  plt.colorbar(im, ax=ax)

  fig.canvas.draw()
  plt.close()

  return fig

def plot_audio(audio, sr):
  fig = plt.figure(figsize=(10, 2))
  plt.plot(np.arange(len(audio)) / sr, audio)
  return fig 

def len_audio(spec):
  sr = get_default_params().sample_rate
  hop = get_default_params().hop_samples
  n_mels = get_default_params().n_mels
  win = hop * 4
  n_fft = 2**((win-1).bit_length())
  fmax = get_default_params().fmax
  fmin = get_default_params().fmin
  mel_basis = librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
  mel_basis = torch.from_numpy(mel_basis).float().to(spec.device)
  fb_pseudo_inverse = torch.linalg.pinv(mel_basis)
  specgram = fb_pseudo_inverse @ torch.exp(spec)
  iSpec = torchaudio.transforms.InverseSpectrogram(n_fft=n_fft, win_length=win, hop_length=hop, normalized=False, pad_mode='reflect',
                                                   center=True, onesided=True).to(spec.device)
  specgram = torch.complex(specgram, specgram * 0.0)
  audio_len = iSpec(specgram).shape[-1]
  return audio_len
