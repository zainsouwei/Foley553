# DCASE2023 - Task 7 - Baseline systems

The code of this repository is mostly from [liuxubo717/sound_generation](https://github.com/liuxubo717/sound_generation). If you use this code, please cite the original repository following: 
```
@article{liu2021conditional,
  title={Conditional Sound Generation Using Neural Discrete Time-Frequency Representation Learning},
  author={Liu, Xubo and Iqbal, Turab and Zhao, Jinzheng and Huang, Qiushi and Plumbley, Mark D and Wang, Wenwu},
  journal={arXiv preprint arXiv:2107.09998},
  year={2021}
}
```
For the neural vocoder, we brought the generator model code from [jik876/hifi-gan](https://github.com/jik876/hifi-gan).

## Set up

* Clone the repository: 

  ```
  git clone https://github.com/DCASE2023-Task7-Foley-Sound-Synthesis/dcase2023_task7_baseline.git
  ```

* Create conda environment with dependencies: 

  ```
  conda env create -f dcase_baseline_env.yaml -n dcase_task7
  ```

* Activate conda environment:  

  ```
  conda activate dcase_task7
  ```

* Download the development dataset and move it to the root folder. The dataset path must be `./DCASEFoleySoundSynthesisDevSet` (in same folder as training script. dataset found here: https://drive.google.com/drive/folders/1GzfZvYVdbgDXnykOR93C3LCchPYBPh5I
## Usage:

1: (Stage 1) Train a multi-scale VQ-VAE to extract the Discrete T-F Representation (DTFR) of sound. The pre-trained model will be saved to `checkpoint/vqvae/`.

```
python train_vqvae.py --epoch 800
python train_vqvae.py --epoch 800 --checkpoint_file /path/to/checkpoint.pt --start_epoch 10
```

2: Extract DTFR for stage 2 training. 

```
python extract_code.py --vqvae_checkpoint [VQ-VAE CHECKPOINT]
```

3: (Stage 2) Train a PixelSNAIL model on the extracted DTFR of sound. The pre-trained model will be saved to `checkpoint/pixelsnail-final/`.

```
python train_pixelsnail.py --epoch 1500
```

4: Inference sounds. The synthesized sound samples will be saved to `./synthesized`

```
python inference.py --vqvae_checkpoint [VQ-VAE CHECKPOINT] --pixelsnail_checkpoint [PIXELSNAIL CHECKPOINT] --number_of_synthesized_sound_per_class [NUMBER OF SOUND SAMPLES]
```

Update and push conda env to file for others to use to update their env:
 conda env export --name dcase_task7 --file dcase_baseline_env.yml

