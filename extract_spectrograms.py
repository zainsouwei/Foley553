import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import lmdb
from tqdm import tqdm
import audio2mel
from datasets import get_dataset_filelist
from vqvae import VQVAE
from datasets import CodeRow


def extract(lmdb_env, loader, model, device, output_dir):
    index = 0

    with lmdb_env.begin(write=True) as txn:
        pbar = tqdm(loader)

        for img, _, _, filename in pbar:
            img = img.to(device)

            out, latent_loss = model(img)
            out = out.detach().cpu().numpy()

            for file, spec in zip(filename, out):
                folder_name = os.path.basename(os.path.dirname(file))
                spec_filename = folder_name + os.path.splitext(os.path.basename(file))[0] + '.npy'
                spec_filepath = os.path.join(output_dir, spec_filename)
                np.save(spec_filepath, spec)
                index += 1
                pbar.set_description(f'saved: {index}')

        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--vqvae_checkpoint', type=str, default='./checkpoint/vqvae/vqvae.pth'
    )
    parser.add_argument('--name', type=str, default='/home/erssmith/Foley553/hifi-gan-master/ft_dataset')

    args = parser.parse_args()

    device = 'cuda'

    train_file_list = get_dataset_filelist()

    train_set = audio2mel.Audio2Mel(
        train_file_list, 22050 * 4, 1024, 80, 256, 22050, 0, 8000
    )

    loader = DataLoader(train_set, batch_size=128, sampler=None, num_workers=2)

    model = VQVAE()
    model.load_state_dict(torch.load(args.vqvae_checkpoint, map_location='cpu'))
    model = model.to(device)
    model.eval()

    map_size = 100 * 1024 * 1024 * 1024

    env = lmdb.open(args.name, map_size=map_size)

    extract(env, loader, model, device)