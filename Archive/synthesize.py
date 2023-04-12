import sys
import librosa
import numpy as np
import soundfile as sf
import pywt
from HiFiGanWrapper import HiFiGanWrapper
import matplotlib.pyplot as plt
import audio2mel

# def save_mel_image(mel_spec, output_path):
#     plt.figure(figsize=(10, 4))
#     mel_spec_2d = np.squeeze(mel_spec)  # Remove extra dimension
#     plt.imshow(mel_spec_2d, origin="lower", aspect="auto", cmap="viridis")
#     plt.xlabel("Time")
#     plt.ylabel("Mel-frequency bins")
#     plt.colorbar(format="%+2.0f dB")
#     plt.tight_layout()
#     plt.savefig(output_path)
#     plt.close()

def save_mel_image(mel_spec, wavelet_coeffs, output_path):
# ????????????
    pass

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

def extract_mel_wavelet_coeffs(mel_spec, wavelet='sym4', level=2):
    coeffs = pywt.wavedec2(mel_spec[0], wavelet, level=level)
    return coeffs

def reconstruct_mel_from_wavelet(coeffs_list, wavelet='sym4'):
    mel_reconstructed = []
    for coeffs in coeffs_list:
        reconstructed_band = pywt.waverec(coeffs, wavelet)
        mel_reconstructed.append(reconstructed_band)
    return np.array(mel_reconstructed)

def main(input_wav_path, output_mel_path, output_wavelet_path, mel_img_path, mel_wavelet_img_path):
    # Load input wav file
    audio = load_audio(input_wav_path)

    # Extract Mel-spectrogram
    mel_spec = extract_mel(audio)
    # save_mel_image(mel_spec, mel_img_path)

    # # Initialize HiFi-GAN
    # hifi_gan = HiFiGanWrapper('./checkpoint/hifigan/g_00935000', './checkpoint/hifigan/hifigan_config.json')

    # # Generate audio from Mel-spectrogram
    # audio_from_mel = hifi_gan.generate_audio_by_hifi_gan(mel_spec)
    # save_audio(audio_from_mel, output_mel_path)

    # Extract wavelet features
    wavelet_features = extract_mel_wavelet_coeffs(mel_spec)

    # Plot mel spectrogram and wavelet coefficients
    save_mel_image(mel_spec, wavelet_features, mel_wavelet_img_path)
    # melwave = reconstruct_mel_from_wavelet(wavelet_features)

    # save_mel_image(melwave, mel_wavelet_img_path)

    # # Generate audio from wavelet features
    # audio_from_mel = hifi_gan.generate_audio_by_hifi_gan(melwave)
    # save_audio(audio_from_mel, output_wavelet_path)

if __name__ == '__main__':
    input_wav_path = sys.argv[1]
    output_mel_path = sys.argv[2]
    output_wavelet_path = sys.argv[3]
    mel_img_path = sys.argv[4]
    mel_wavelet_img_path = sys.argv[5]

    main(input_wav_path, output_mel_path, output_wavelet_path, mel_img_path, mel_wavelet_img_path)
    # python3 synthesize.py Test/dog.wav Test/output_mel_generated_wav_file.wav Test/output_wavelet_generated_wav_file.wav Test/output_mel_generated_mel_file.jpg Test/output_wavelet_generated_mel_file.jpg
