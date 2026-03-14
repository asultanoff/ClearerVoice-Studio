# ClearerVoice-Studio: Target Speaker Extraction Algorithms


## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Usage](#2-usage)
- [3. Task: Audio-only Speaker Extraction Conditioned on a Reference Speech](#3-audio-only-speaker-extraction-conditioned-on-a-reference-speech)
- [4. Task: Audio-visual Speaker Extraction Conditioned on Face (Lip) Recording](#4-audio-visual-speaker-extraction-conditioned-on-face-or-lip-recording)
- [5. Task: Audio-visual Speaker Extraction Conditioned on Body Gestures](#5-audio-visual-speaker-extraction-conditioned-on-body-gestures)
- [6. Task: Neuro-steered Speaker Extraction Conditioned on EEG Signals](#6-neuro-steered-speaker-extraction-conditioned-on-eeg-signals)


## 1. Introduction

This repository provides training scripts for various target speaker extraction algorithms, including audio-only, audio-visual, and neuro-steered speaker extraction.

## 2. Usage

### Step-by-Step Guide

1. **Clone the Repository**

``` sh
git clone https://github.com/modelscope/ClearerVoice-Studio.git
```

2. **Create Conda Environment**

``` sh
cd ClearerVoice-Studio/train/target_speaker_extraction/
conda create -n clear_voice_tse python=3.9
conda activate clear_voice_tse
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

Alternative with `uv`:

``` sh
cd ClearerVoice-Studio/train/target_speaker_extraction/
uv sync
uv run bash train.sh
```

You can also launch training directly without the shell wrapper:

``` sh
uv run python train.py --config config/config_custom_speech_SpEx-plus_onthefly.yaml --checkpoint_dir checkpoints/log_custom --train_from_last_checkpoint 0
```

To automatically push the final checkpoint directory to Hugging Face Hub after training, either set `hf_repo_id` in the config/CLI or provide `HF_REPO` in a `.env` file:

``` sh
cat > .env <<'EOF'
HF_REPO=your-org/your-model
HF_TOKEN=hf_...
EOF

uv run python train.py --config config/config_custom_speech_SpEx-plus_onthefly.yaml --checkpoint_dir checkpoints/log_custom
```

The repo is created as private by default; override with `--hf_private 0` if needed.

3. **Download Dataset**
> Follow the download links or preprocessing scripts provided under each task section.

4. **Modify Dataset Paths** 
> Update the paths to your datasets in the configuration files. For example, modify the "audio_direc" and "ref_direc" in "config/config_YGD_gesture_seg_2spk.yaml"

5. **Modify Train Configuration** 
> Adjust the settings in the "train.sh" file. For example, set "n_gpu=1" for single-GPU training, or "n_gpu=2" for two-GPU distributed training

6. **Start Training**

``` sh
bash train.sh
```

7. **Visualize Training Progress using Tensorboard**

``` sh
tensorboard --logdir ./checkpoints/
```

8. **Optionally Evaluate Checkpoints**

``` sh
bash evaluate_only.sh
```






## 3. Audio-only speaker extraction conditioned on a reference speech

### Support datasets for training: 

* WSJ0-2mix [[Download](https://github.com/gemengtju/Tutorial_Separation/blob/master/generation/wsj0-2mix/create-speaker-mixtures.zip)]

### Support models for training: 

* SpEx+ (Non-causal) [[Paper: SpEx+: A Complete Time Domain Speaker Extraction Network](https://arxiv.org/abs/2005.04686)]

### Non-causal (Offline) WSJ0-2mix benchmark: 

| Dataset | Speakers | Model| Config | Checkpoint | SI-SDRi (dB) | SDRi (dB) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| WSJ0-2mix | 2-mix | SpEx+ | [Paper](https://arxiv.org/abs/2005.04686) | - | 16.9 | 17.2 |
| WSJ0-2mix | 2-mix | SpEx+ | [This repo](./config/config_wsj0-2mix_speech_SpEx-plus_2spk.yaml) | [This repo](https://huggingface.co/alibabasglab/log_wsj0-2mix_speech_SpEx-plus_2spk/) | 17.1 | 17.5 |


## 4. Audio-visual speaker extraction conditioned on face or lip recording

### Support datasets for training: 

* VoxCeleb2 [[Download](https://huggingface.co/datasets/alibabasglab/VoxCeleb2-mix)]
* LRS2 [[Download](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html)]

### Support models for training: 

* AV-ConvTasNet (Causal/Non-causal) [[Paper: Time Domain Audio Visual Speech Separation](https://arxiv.org/abs/1904.03760)]
* AV-DPRNN (aka USEV) (Non-causal) [[Paper: Universal Speaker Extraction With Visual Cue](https://ieeexplore.ieee.org/document/9887809)]
* AV-TFGridNet (Non-causal) [[Paper: Scenario-Aware Audio-Visual TF-GridNet for Target Speech Extraction](https://arxiv.org/abs/2310.19644)]
* AV-TFGridNet-ISAM (Non-causal) [[Paper: Plug-and-Play Co-Occurring Face Attention for Robust Audio-Visual Speaker Extraction](https://arxiv.org/abs/2505.20635)]
* AV-Mossformer2 (Non-causal) [[Paper: ClearerVoice-Studio: Bridging Advanced Speech Processing Research and Practical Deployment](https://arxiv.org/abs/2506.19398)]




### Non-causal (Offline) VoxCeleb2-mix benchmark: 

 Dataset | Speakers | Model| Config | Checkpoint | SI-SDRi (dB) | SDRi (dB) 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| VoxCeleb2 | 2-mix | [AV-ConvTasNet](https://arxiv.org/abs/1904.03760) | [Paper](https://arxiv.org/abs/1904.03760) | - | 10.6 | 10.9
| VoxCeleb2 | 2-mix | [MuSE](https://arxiv.org/abs/2010.07775) | [Paper](https://arxiv.org/abs/2010.07775) | - | 11.7 | 12.0
| VoxCeleb2 | 2-mix | [reentry](https://ieeexplore.ieee.org/document/9721129) | [Paper](https://ieeexplore.ieee.org/document/9721129) | - | 12.6 | 12.9
| VoxCeleb2 | 2-mix | [AV-DPRNN](https://ieeexplore.ieee.org/document/9887809) | [This repo](./config/config_VoxCeleb2_lip_dprnn_2spk.yaml) | [This repo](https://huggingface.co/alibabasglab/log_VoxCeleb2_lip_dprnn_2spk/)| 11.5 | 11.8
| VoxCeleb2 | 2-mix | [AV-TFGridNet](https://arxiv.org/abs/2310.19644) | [This repo](./config/config_VoxCeleb2_lip_tfgridnet_2spk.yaml) | [This repo](https://huggingface.co/alibabasglab/log_VoxCeleb2_lip_tfgridnet_2spk/)| 13.7 | 14.1
| VoxCeleb2 | 2-mix | [AV-TFGridNet-ISAM](https://arxiv.org/abs/2505.20635) | [This repo](./config/config_VoxCeleb2_lip_tfgridnet-isam_2spk.yaml) | [This repo](https://huggingface.co/alibabasglab/log_VoxCeleb2_lip_tfgridnet-isam_2spk/)| 14.5 | 14.8
| VoxCeleb2 | 2-mix | [AV-Mossformer2](https://arxiv.org/abs/2506.19398) | [This repo](./config/config_VoxCeleb2_lip_mossformer2_2spk.yaml) | [This repo](https://huggingface.co/alibabasglab/log_VoxCeleb2_lip_mossformer2_2spk/)| 14.6 | 14.9
| VoxCeleb2 | 3-mix | [AV-ConvTasNet](https://arxiv.org/abs/1904.03760) | [Paper](https://arxiv.org/abs/1904.03760) | - | 9.8 | 10.2
| VoxCeleb2 | 3-mix | [MuSE](https://arxiv.org/abs/2010.07775) | [Paper](https://arxiv.org/abs/2010.07775) | - | 11.6 | 12.2
| VoxCeleb2 | 3-mix | [reentry](https://ieeexplore.ieee.org/document/9721129) | [Paper](https://ieeexplore.ieee.org/document/9721129) | - | 12.6 | 13.1
| VoxCeleb2 | 3-mix | [AV-DPRNN](https://ieeexplore.ieee.org/document/9887809) | [This repo](./config/config_VoxCeleb2_lip_dprnn_3spk.yaml) | [This repo](https://huggingface.co/alibabasglab/log_VoxCeleb2_lip_dprnn_3spk/)| 10.5 | 11.0
| VoxCeleb2 | 3-mix | [AV-TFGridNet](https://arxiv.org/abs/2310.19644) | [This repo](./config/config_VoxCeleb2_lip_tfgridnet_3spk.yaml) | [This repo](https://huggingface.co/alibabasglab/log_VoxCeleb2_lip_tfgridnet_3spk/)| 14.2 | 14.6
| VoxCeleb2 | 3-mix | [AV-TFGridNet-ISAM](https://arxiv.org/abs/2505.20635) | [This repo](./config/config_VoxCeleb2_lip_tfgridnet-isam_3spk.yaml) | [This repo](https://huggingface.co/alibabasglab/log_VoxCeleb2_lip_tfgridnet-isam_3spk/)| 15.6 | 15.9
| VoxCeleb2 | 3-mix | [AV-Mossformer2](https://arxiv.org/abs/2506.19398) | [This repo](./config/config_VoxCeleb2_lip_mossformer2_3spk.yaml) | [This repo](https://huggingface.co/alibabasglab/log_VoxCeleb2_lip_mossformer2_3spk/)| 15.5 | 16.0


### Non-causal (Offline) LRS2-mix benchmark: 

 Dataset | Speakers | Model| Config | Checkpoint | SI-SDRi (dB) | SDRi (dB) 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LRS2 | 2-mix | [AV-ConvTasNet](https://arxiv.org/abs/1904.03760) | [This repo](./config/config_LRS2_lip_convtasnet_2spk.yaml)| [This repo](https://huggingface.co/alibabasglab/log_LRS2_lip_convtasnet_2spk/) | 11.6 | 11.9
| LRS2 | 2-mix | [AV-DPRNN](https://ieeexplore.ieee.org/document/9887809) | [This repo](./config/config_LRS2_lip_dprnn_2spk.yaml) | [This repo](https://huggingface.co/alibabasglab/log_LRS2_lip_dprnn_2spk/) | 12.0 | 12.4 
| LRS2 | 2-mix | [AV-TFGridNet](https://arxiv.org/abs/2310.19644) | [This repo](./config/config_LRS2_lip_tfgridnet_2spk.yaml) | [This repo](https://huggingface.co/alibabasglab/log_LRS2_lip_tfgridnet_2spk/)| 15.1 | 15.4 
| LRS2 | 2-mix | [AV-Mossformer2](https://arxiv.org/abs/2506.19398)| [This repo](./config/config_LRS2_lip_mossformer2_2spk.yaml) | [This repo](https://huggingface.co/alibabasglab/log_LRS2_lip_mossformer2_2spk/)| 15.5 | 15.8 
| LRS2 | 3-mix | [AV-ConvTasNet](https://arxiv.org/abs/1904.03760) | [This repo](./config/config_LRS2_lip_convtasnet_3spk.yaml) | [This repo](https://huggingface.co/alibabasglab/log_LRS2_lip_convtasnet_3spk/)| 10.8 | 11.3
| LRS2 | 3-mix | [AV-DPRNN](https://ieeexplore.ieee.org/document/9887809) | [This repo](./config/config_LRS2_lip_dprnn_3spk.yaml) | [This repo](https://huggingface.co/alibabasglab/log_LRS2_lip_dprnn_3spk/)| 10.6 | 11.1 
| LRS2 | 3-mix | [AV-TFGridNet](https://arxiv.org/abs/2310.19644) | [This repo](./config/config_LRS2_lip_tfgridnet_3spk.yaml) | [This repo](https://huggingface.co/alibabasglab/log_LRS2_lip_tfgridnet_3spk/)| 15.0 | 15.4 
| LRS2 | 3-mix | [AV-Mossformer2](https://arxiv.org/abs/2506.19398) | [This repo](./config/config_LRS2_lip_mossformer2_3spk.yaml) | [This repo](https://huggingface.co/alibabasglab/log_LRS2_lip_mossformer2_3spk/)| 16.2 | 16.6 



## 5. Audio-visual speaker extraction conditioned on body gestures

### Support datasets for training: 

* YGD [[Download](https://huggingface.co/datasets/alibabasglab/YGD-mix)] [[Paper: Robots Learn Social Skills: End-to-End Learning of Co-Speech Gesture Generation for Humanoid Robots](https://arxiv.org/abs/1810.12541)]

### Support models for training: 

* SEG (Non-causal) [[Paper: Speaker Extraction with Co-Speech Gestures Cue](https://ieeexplore.ieee.org/document/9774925)]

### Non-causal (Offline) YGD-mix benchmark: 

 Dataset | Speakers | Model| Config | Checkpoint | SI-SDRi (dB) | SDRi (dB) 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| YGD | 2-mix | [DPRNN-GSR](https://ieeexplore.ieee.org/document/9774925) | [Paper](https://ieeexplore.ieee.org/document/9774925) | - | 6.2 | 8.1 
| YGD | 2-mix | [SEG](https://ieeexplore.ieee.org/document/9774925) | [Paper](https://ieeexplore.ieee.org/document/9774925) | - | 9.1 | 10.0 
| YGD | 2-mix | [SEG](https://ieeexplore.ieee.org/document/9774925) | [This repo](./config/config_YGD_gesture_seg_2spk.yaml) | [This repo](https://huggingface.co/alibabasglab/log_YGD_gesture_seg_2spk/)| 9.5 | 10.4 
| YGD | 3-mix | [DPRNN-GSR](https://ieeexplore.ieee.org/document/9774925) | [Paper](https://ieeexplore.ieee.org/document/9774925) | - | 1.8 | 3.5 
| YGD | 3-mix | [SEG](https://ieeexplore.ieee.org/document/9774925) | [Paper](https://ieeexplore.ieee.org/document/9774925) | - | 5.0 | 5.3 
| YGD | 3-mix | [SEG](https://ieeexplore.ieee.org/document/9774925) | [This repo](./config/config_YGD_gesture_seg_3spk.yaml) | [This repo](https://huggingface.co/alibabasglab/log_YGD_gesture_seg_3spk/)| 4.9 | 5.6 

## 6. Neuro-steered speaker extraction conditioned on EEG signals

### Support datasets for training: 
* KUL [[Download](https://huggingface.co/datasets/alibabasglab/KUL-mix)] [[Paper: Auditory-Inspired Speech Envelope Extraction Methods for Improved EEG-Based Auditory Attention Detection in a Cocktail Party Scenario](https://ieeexplore.ieee.org/document/7478117?signout=success)]

### Support models for training: 
* NeuroHeed (Non-causal) [[Paper: Neuro-Steered Speaker Extraction Using EEG Signals](https://ieeexplore.ieee.org/document/10683957)]

### Non-causal (Offline) KUL-mix benchmark: 

 Dataset | Speakers | Model | Config | Checkpoint | SI-SDRi (dB) | SDRi (dB) 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| KUL | 2-mix | [NeuroHeed](https://ieeexplore.ieee.org/document/10683957) | [Paper](https://ieeexplore.ieee.org/document/10683957) | - | 14.3 | 15.5 
| KUL | 2-mix | [NeuroHeed](https://ieeexplore.ieee.org/document/10683957) | [This repo](./config/config_KUL_eeg_neuroheed_2spk.yaml) | [This repo](https://huggingface.co/alibabasglab/log_KUL_eeg_neuroheed_2spk/)| 13.4 | 15.0 

### Causal (online) KUL-mix benchmark: 

 Dataset | Speakers | Model | Config | Checkpoint | SI-SDRi (dB) | SDRi (dB) 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| KUL | 2-mix | [NeuroHeed](https://ieeexplore.ieee.org/document/10683957) | [Paper](https://ieeexplore.ieee.org/document/10683957) | - | 11.2 | 11.8 
