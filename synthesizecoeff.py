import sys
import librosa
import numpy as np
import soundfile as sf
import pywt
from HiFiGanWrapper import HiFiGanWrapper
import matplotlib.pyplot as plt
import audio2mel
import os
import scipy
from scipy import ndimage


def save_mel_image(mel, output_path):
    plt.figure(figsize=(10, 4))
    mel_spec_2d = np.squeeze(mel)  # Remove extra dimension
    plt.imshow(mel_spec_2d, origin="lower", aspect="auto", cmap="viridis")
    plt.xlabel("Time")
    plt.ylabel("Mel-frequency bins")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def load_audio(audio_path, sr=22050):
    audio, _ = librosa.load(audio_path, sr=sr)
    return audio

def save_audio(audio, save_path, sr=22050):
    sf.write(save_path, audio, samplerate=sr)

def extract_mel(audio):
    input_audio = audio2mel.Audio2Mel(
        [{'file_path': 'dummy.wav', 'class_id': 0}],  # Pass a dummy file path, it won't be used
        22050 * 4, 1024, 80, 256, 22050, 0, 8000
    )
    
    mel_spec = audio2mel.mel_spectrogram_hifi(
        audio,
        n_fft=input_audio.n_fft,
        n_mels=input_audio.n_mels,
        hop_length=input_audio.hop_length,
        sample_rate=input_audio.sample_rate,
        fmin=input_audio.fmin,
        fmax=input_audio.fmax,
    )

    return mel_spec


def main(input_wav_path, output_mel_path, output_wavelet_path, mel_img_path, mel_wavelet_img_path):
    # Load input wav file
    audio = load_audio(input_wav_path)

    # Extract Mel-spectrogram
    mel_spec = extract_mel(audio)

    mother = 'sym4'

    # Extract wavelet coefficients
    wavelet_coeffs = pywt.dwt2(mel_spec[0], mother)
    der_coeffs = np.array(np.gradient(wavelet_coeffs[0]))
    thresh_coeffs = pywt.threshold(wavelet_coeffs[0], 0.1, mode='soft')
    transformed = (wavelet_coeffs[0] - wavelet_coeffs[1][2]) + der_coeffs[1]

    # Plot wavelet coefficients and save as image
    print(mel_spec.shape)
    print(wavelet_coeffs[0].shape)
    print(wavelet_coeffs[1][0].shape)
    print(wavelet_coeffs[1][1].shape)
    print(wavelet_coeffs[1][2].shape)
    print(der_coeffs.shape)
    print(thresh_coeffs.shape)

    save_mel_image(mel_spec,mel_img_path)
    save_mel_image(wavelet_coeffs[0],os.path.join(mel_wavelet_img_path,"wave1.jpg"))
    save_mel_image(wavelet_coeffs[1][0],os.path.join(mel_wavelet_img_path,"wave2.jpg"))
    save_mel_image(wavelet_coeffs[1][1],os.path.join(mel_wavelet_img_path,"wave3.jpg"))
    save_mel_image(wavelet_coeffs[1][2],os.path.join(mel_wavelet_img_path,"wave4.jpg"))
    save_mel_image(transformed,os.path.join(mel_wavelet_img_path,"wavetransform.jpg"))
    save_mel_image(der_coeffs[0],os.path.join(mel_wavelet_img_path,"wavehorizontalderivative.jpg"))
    save_mel_image(der_coeffs[1],os.path.join(mel_wavelet_img_path,"waveverticalderivative.jpg"))
    save_mel_image(thresh_coeffs,os.path.join(mel_wavelet_img_path,"wavethresh.jpg"))

    # Inverse transforms
    spec = pywt.waverec2([wavelet_coeffs[0]], mother)
    spectransform = pywt.waverec2([transformed], mother)
    save_mel_image(spec,os.path.join(mel_wavelet_img_path,"specrecon.jpg"))
    save_mel_image(spectransform,os.path.join(mel_wavelet_img_path,"spectransform.jpg"))
    print(spec.shape)
    # Add extra 1st dimension to match their shapes
    spec = np.expand_dims(spec, axis=0)
    # Upsample can also use CV2
    # Compute zoom factors for each dimension
    zoom_factors = (1, 80 / spec.shape[1], 344 / spec.shape[2])
    # Apply the zoom function
    spec_resized = ndimage.zoom(spec, zoom_factors, order=1)
    print(spec_resized.shape)

    # Generate Foley Sounds (Originating from Jack Foley - Father of Sound Effects and Loving Husband to Beatrice (Lastname Unknown))
    # Initialize HiFi-GAN
    hifi_gan = HiFiGanWrapper('./checkpoint/hifigan/g_00935000', './checkpoint/hifigan/hifigan_config.json')

    # Generate audio from Mel-spectrogram
    audio_from_mel = hifi_gan.generate_audio_by_hifi_gan(mel_spec)
    save_audio(audio_from_mel, output_mel_path)

    # Generate audio from Reconstructed Mel-spectrogram
    audio_from_spec = hifi_gan.generate_audio_by_hifi_gan(spec_resized)
    save_audio(audio_from_spec, output_wavelet_path)


if __name__ == '__main__':
    input_wav_path = sys.argv[1]
    output_mel_path = sys.argv[2]
    output_wavelet_path = sys.argv[3]
    mel_img_path = sys.argv[4]
    mel_wavelet_img_path = sys.argv[5]

    main(input_wav_path, output_mel_path, output_wavelet_path, mel_img_path, mel_wavelet_img_path)
    # python3 synthesize.py Test/dog.wav Test/output_mel_generated_wav_file.wav Test/output_wavelet_generated_wav_file.wav Test/output_mel_generated_mel_file.jpg Test/output_wavelet_generated_mel_file.jpg
