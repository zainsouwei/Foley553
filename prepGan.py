import os
import shutil
from glob import glob
import torch
from vqvae import VQVAE

# Change this path to the location of your DCASEFoleySoundSynthesisDevSet folder
input_folder = "path/to/DCASEFoleySoundSynthesisDevSet"
output_folder = "processed_wav_files"

os.makedirs(output_folder, exist_ok=True)

vqvae_checkpoint = "path/to/vqvae_checkpoint.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VQVAE()
model.load_state_dict(torch.load(vqvae_checkpoint, map_location=device))
model = model.to(device)
model.eval()

for category_folder in os.listdir(input_folder):
    category_path = os.path.join(input_folder, category_folder)
    if os.path.isdir(category_path):
        for wav_file in glob(os.path.join(category_path, "*.wav")):
            new_filename = f"{category_folder}_{os.path.basename(wav_file)}"
            shutil.copy(wav_file, os.path.join(output_folder, new_filename))

            # Load your .wav file as a PyTorch tensor here
            wav_tensor = load_wav_as_tensor(wav_file)

            class_id = get_class_id(category_folder)

            with torch.no_grad():
                out, _ = model(wav_tensor, label_condition=class_id)
                npy_output_file = os.path.join(output_folder, f"{os.path.splitext(new_filename)[0]}.npy")
                torch.save(out.cpu(), npy_output_file)
