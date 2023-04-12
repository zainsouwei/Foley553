import torch
import librosa
from scipy.io.wavfile import read as loadwav
import numpy as np
import datasets
import matplotlib.pyplot as plt
from scipy import signal
import pywt
from librosa.filters import mel

import warnings

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}
wavelet_basis = {}
mel_fb = {}


def mel_spectrogram_hifi(
    audio, n_fft, n_mels, sample_rate, hop_length, fmin, fmax, center=False
):
    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)

    if torch.min(audio) < -1.0:
        print('min value is ', torch.min(audio))
    if torch.max(audio) > 1.0:
        print('max value is ', torch.max(audio))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel_fb = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[str(fmax) + '_' + str(audio.device)] = (
            torch.from_numpy(mel_fb).float().to(audio.device)
        )
        hann_window[str(audio.device)] = torch.hann_window(n_fft).to(audio.device)

    audio = torch.nn.functional.pad(
        audio.unsqueeze(1),
        (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
        mode='reflect',
    )
    audio = audio.squeeze(1)

    spec = torch.stft(
        audio,
        n_fft,
        hop_length=hop_length,
        window=hann_window[str(audio.device)],
        center=center,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    print("fft shape")
    #print(mel_basis[str(fmax) + '_' + str(audio.device)].shape)  
    print(spec.shape)
    mel = torch.matmul(mel_basis[str(fmax) + '_' + str(audio.device)], spec)
    mel = spectral_normalize_torch(mel).numpy()

    # pad_size = math.ceil(mel.shape[2] / 8) * 8 - mel.shape[2]
    #
    # mel = np.pad(mel, ((0, 0), (0, 0), (0, pad_size)))

    return mel


def mel_dwt(audio, sample_rate, window_length=0.025, window_step=0.01, n_filters=40, n_levels=4):
    # Step 1: Divide the speech signal into blocks using overlapping smooth windows such as Hamming, Hanning, etc.
    window = np.hamming(int(window_length * sample_rate))
    stride = int(window_step * sample_rate)

    # Split the signal into frames with overlapping windows
    frames = librosa.util.frame(audio, frame_length=len(window), hop_length=stride, axis=0)
    n_frames = frames.shape[1]

    # Step 2: Take the Discrete Time Fourier Transform (DTFT) of the windowed signal.
    # Step 3: Calculate the square of the DTFT of the windowed signal.
    power_frames = np.abs(np.fft.rfft(frames, n=1024, axis=1)) ** 2

    # Step 4: Compute Mel filterbank energies
    mel_fb = librosa.filters.mel(sr=sample_rate, n_fft=1024, n_mels=n_filters)
    mel_spec = np.matmul(mel_fb, power_frames.T).T

    # Step 5: Calculate the logarithm of the mel-scaled filterbank energies.
    log_mel_spec = np.log(mel_spec + 1e-8)

    # Step 6: Take the DWT of the log-filterbank energies to calculate MFDWCs
    mfcc_dwt = []
    for i in range(n_frames):
        cA = log_mel_spec[i]
        cA_dwt = []
        for j in range(n_levels):
            cA, cD = pywt.dwt(cA, 'db1')
            cA_dwt.append(cA)
        mfcc_dwt.append(cA_dwt)

    return np.array(mfcc_dwt)


def mel_dwt2(
    audio, n_fft, n_mels, sample_rate, hop_length, fmin, fmax, center=False
):
    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)

    if torch.min(audio) < -1.0:
        print('min value is ', torch.min(audio))
    if torch.max(audio) > 1.0:
        print('max value is ', torch.max(audio))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel_fb = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[str(fmax) + '_' + str(audio.device)] = (
            torch.from_numpy(mel_fb).float().to(audio.device)
        )
        hann_window[str(audio.device)] = torch.hann_window(n_fft).to(audio.device)

    audio = torch.nn.functional.pad(
        audio.unsqueeze(1),
        (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
        mode='reflect',
    )
    audio = audio.squeeze(1)

    spec = torch.stft(
        audio,
        n_fft,
        hop_length=hop_length,
        window=hann_window[str(audio.device)],
        center=center,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    mel = torch.matmul(mel_basis[str(fmax) + '_' + str(audio.device)], spec)
    mel = spectral_normalize_torch(mel).numpy()

    #dwt = pywt.dwt(mel, 'db4')
    dwt = pywt.dwt2(mel[0], 'sym4')

    return dwt


if __name__ == '__main__':

    filename = "cough.wav"
    sample_rate, audio = loadwav(filename)
    audio = audio / MAX_WAV_VALUE

    max_length = 22050 * 4
    n_fft = 1024
    n_mels = 80 
    hop_length = 256
    sample_rate = 22050
    fmin = 0
    fmax = 8000

    mel_spec = mel_spectrogram_hifi(
            audio,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            sample_rate=sample_rate,
            fmin=fmin,
            fmax=fmax,
        )

    print(mel_spec.shape)
    plt.figure()
    plt.imshow(mel_spec.reshape(80,344))
    
    wavelet = 'morl'
    n_scales = 514
    fmin = 0
    fmax = 8000
    level = 2

    #wavelet_spec = dwt_mel_spectrogram_hifi(audio, level, n_mels, sample_rate, fmin, fmax)
    wavelet_spec = mel_dwt2(
            audio,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            sample_rate=sample_rate,
            fmin=fmin,
            fmax=fmax,
            )

    print(wavelet_spec[0].shape)
    print(wavelet_spec[1][0].shape)
    print(wavelet_spec[1][1].shape)
    print(wavelet_spec[1][2].shape)


    f, axarr = plt.subplots(2,2)
    
    axarr[0,0].imshow(wavelet_spec[0].reshape(43,-1))
    axarr[0,1].imshow(wavelet_spec[1][0].reshape(43,-1))
    axarr[1,0].imshow(wavelet_spec[1][1].reshape(43,-1))
    axarr[1,1].imshow(wavelet_spec[1][2].reshape(43,-1))

    plt.show()